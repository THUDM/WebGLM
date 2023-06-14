from transformers import BigBirdTokenizer, BigBirdForQuestionAnswering
import torch
import json
from tqdm import tqdm

PUNCTUATION_SET_TO_EXCLUDE = set(''.join(['‘', '’', '´', '`', '.', ',', '-', '"', '\'', '[', ']', '{', '}', '(', ')', '!', '?']))

def get_sub_answers(answers, begin=0, end=None):
    return [" ".join(x.split(" ")[begin:end]) for x in answers if len(x.split(" ")) > 1]

def expand_to_aliases(given_answers, ignore_prefix=False, ignore_suffix=False):
    if ignore_prefix:
        given_answers = given_answers + get_sub_answers(given_answers, begin=1)
    if ignore_suffix:
        given_answers = given_answers + get_sub_answers(given_answers, end=-1)
    answers = []
    for answer in given_answers:
        alias = answer.replace('_', ' ').lower()
        alias = ''.join(c if c not in PUNCTUATION_SET_TO_EXCLUDE else ' ' for c in alias)
        answers.append(' '.join(alias.split()).strip())
    return set(answers)


def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score = torch.argmax(scores).item()

    return best_start_idx[best_score % top_k], best_end_idx[best_score // top_k]

def extract(extractor, tokenizer, example):
    encoding = tokenizer(example["question"], example["predict"], return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    input_ids = encoding['input_ids'].to("cuda")

    with torch.no_grad():
        start_scores, end_scores = extractor(input_ids=input_ids).to_tuple()

    start_score, end_score = get_best_valid_start_end_idx(start_scores[0], end_scores[0], top_k=8, max_size=16)

    all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
    answer_tokens = all_tokens[start_score: end_score + 1]

    example["output"] = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
    
    answers = expand_to_aliases(example["answer"], ignore_prefix=True, ignore_suffix=True)
    predictions = expand_to_aliases([example["output"]], ignore_prefix=True)
    
    example["match"] = len(list(answers & predictions)) > 0

    return example


def eval(model, args):
    ds = [json.loads(data_str) for data_str in open(args.evaluate_task_data_path).readlines()]
    
    for ix, sample in enumerate(tqdm(ds)):
        output = model.query(sample['question'])
        ds[ix]['predict'] = output['answer']
    
    print('Start Extracting Answer...')
    
    tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-base-trivia-itc")
    extractor = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-base-trivia-itc").to("cuda")
    
    scores = {}
    acc = {}
    
    for sample in tqdm(ds):
        example = {}
        match = extract(extractor, tokenizer, sample)['match']
        labels = sample['labels']
        for label in labels:
            if label not in scores:
                scores[label] = [0, 0]
            scores[label][1] += 1
            if match:
                scores[label][0] += 1
    
    for split, data in scores.items():
        acc[split] = data[0] / data[1]
    
    return acc