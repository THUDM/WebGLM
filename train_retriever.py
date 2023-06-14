from transformers import RobertaTokenizer, RobertaModel, AutoModelWithLMHead, AutoTokenizer, Trainer, AutoModel, BertLMHeadModel
from datasets.load import load_dataset, load_from_disk
import torch, os, sys, time, random, json, argparse
from rouge_score.rouge_scorer import RougeScorer

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler

class QuestionReferenceDensity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.question_encoder = AutoModel.from_pretrained("facebook/contriever-msmarco")
        self.reference_encoder = AutoModel.from_pretrained("facebook/contriever-msmarco")

        total = sum([param.nelement() for param in self.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
    
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
        
    
    def forward(self, question, pos, neg):
        global args
        
        q = self.question_encoder(**question)
        r_pos = self.reference_encoder(**pos)
        r_neg = self.reference_encoder(**neg)
        cls_q = self.mean_pooling(q[0], question["attention_mask"])
        cls_q /= args.temp
        cls_r_pos = self.mean_pooling(r_pos[0], pos["attention_mask"])
        cls_r_neg = self.mean_pooling(r_neg[0], neg["attention_mask"])
        
        l_pos = torch.matmul(cls_q, torch.transpose(cls_r_pos, 0, 1))

        l_neg = torch.matmul(cls_q, torch.transpose(cls_r_neg, 0, 1))

        return l_pos, l_neg
        
    @staticmethod
    def loss(l_pos, l_neg):
        return torch.nn.functional.cross_entropy(torch.cat([l_pos, l_neg], dim=1), torch.arange(0, len(l_pos), dtype=torch.long, device=args.device))
    
    @staticmethod
    def num_correct(l_pos, l_neg):
        return ((torch.diag(l_pos) > torch.diag(l_neg))==True).sum()

    @staticmethod
    def acc(l_pos, l_neg):
        return ((torch.diag(l_pos) > torch.diag(l_neg))==True).sum() / len(l_pos)


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(WarmupLinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return (1 - self.ratio) * step / float(max(1, self.warmup))

        return max(
            0.0,
            1.0 + (self.ratio - 1) * (step - self.warmup) / float(max(1.0, self.total - self.warmup)),
        )


def move_dict_to_device(obj, device):
    for key in obj:
        obj[key] = obj[key].to(device)

def collate(data):
    question = tokenizer([item["question"] for item in data], return_tensors="pt", padding=True, truncation=True)
    positive_reference = tokenizer([item["positive_reference"] for item in data], return_tensors="pt", padding=True, truncation=True)
    negative_reference = tokenizer([item["negative_reference"] for item in data], return_tensors="pt", padding=True, truncation=True)

    for key in question: question[key] = question[key].to(args.device)
    for key in positive_reference: positive_reference[key] = positive_reference[key].to(args.device)
    for key in negative_reference: negative_reference[key] = negative_reference[key].to(args.device)

    return question, positive_reference, negative_reference

def eval():
    # print("EVAL ...")
    model.eval()
    with torch.no_grad():
        total_acc = 0
        for q, pos, neg in eval_loader:
            results = model(q, pos, neg)
            # print(results)
            # exit()
            tot_cr = model.num_correct(*results)
            total_acc += tot_cr

        print("EVALUATION, Acc: %10.6f"%(total_acc / len(eval_set)))
    
def save(name):
    os.makedirs(log_dir, exist_ok=True)
    model.question_encoder.save_pretrained(os.path.join(log_dir, name, "query_encoder"))
    model.reference_encoder.save_pretrained(os.path.join(log_dir, name, "reference_encoder"))

def train(max_epoch = 10, eval_step = 200, save_step = 400, print_step = 50):
    step = 0
    for epoch in range(0, max_epoch):
        print("EPOCH %d"%epoch)
        for q, pos, neg in train_loader:
            model.train()
            step += 1
            opt.zero_grad()
            results = model(q, pos, neg)
            loss = model.loss(*results)
            
            if step % print_step == 0:
                print("Step %4d, Loss, Acc: %10.6f, %10.6f"%(step, loss, model.acc(*results)))
            
            loss.backward()
            opt.step()
            
            scheduler.step()
            model.zero_grad()
            if step % eval_step == 0:
                eval()
                pass
            if step % save_step == 0:
                save("step-%d"%(step))
            

        save("step-%d-epoch-%d"%(step, epoch))
        # eval()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--max_epoch", type=int, default=3)
    args.add_argument("--eval_step", type=int, default=40)
    args.add_argument("--save_step", type=int, default=40)
    args.add_argument("--print_step", type=int, default=40)
    args.add_argument("--device", type=str, default="cuda")
    args.add_argument("--temp", type=float, default=0.05)
    args.add_argument("--train_batch_size", type=int, default=64)
    args.add_argument("--eval_batch_size", type=int, default=32)
    args.add_argument("--lr", type=float, default=1e-6)
    args.add_argument("--warmup", type=int, default=100)
    args.add_argument("--total", type=int, default=1000)
    args.add_argument("--ratio", type=float, default=0.0)
    args.add_argument("--save_dir", type=str, default="./retriever_runs")
    args.add_argument("--train_data_dir", type=str, required=True)
    
    args = args.parse_args()
    
    log_dir = os.path.join(args.save_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())))
    
    train_set = load_from_disk(os.path.join(args.train_data_dir, "train"))
    eval_set = load_from_disk(os.path.join(args.train_data_dir, "eval"))
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, collate_fn=collate)
    eval_loader = DataLoader(eval_set, batch_size=args.eval_batch_size, collate_fn=collate)

    model = QuestionReferenceDensity()
    model = model.to(args.device)
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler_args = {
        "warmup": args.warmup,
        "total": args.total,
        "ratio": args.ratio,
    }
    scheduler = WarmupLinearScheduler(opt, **scheduler_args)
    temp = args.temp
    
    train(max_epoch=args.max_epoch, eval_step=args.eval_step, save_step=args.save_step, print_step=args.print_step)

