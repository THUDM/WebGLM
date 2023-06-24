from .retriever import ReferenceRetiever
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re, os

class WebGLM:
    def __init__(self, webglm_ckpt_path, retriever_ckpt_path, device=None, filter_max_batch_size=400, searcher_name="serpapi") -> None:
        self.device = device
        self.ref_retriever = ReferenceRetiever(retriever_ckpt_path, device, filter_max_batch_size, searcher_name)
        self.tokenizer = AutoTokenizer.from_pretrained(webglm_ckpt_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(webglm_ckpt_path, trust_remote_code=True)
        self.model = self.model.half()
        if device:
            self.model.to(device)
        self.model.eval()
    
    def query(self, question):
        refs = self.ref_retriever.query(question)
        if not refs:
            return { "references": [], "answer": "" }
        prompt = ''
        for ix, ref in enumerate(refs):
            txt = ref["text"]
            prompt += f'Reference [{ix+1}]: {txt}' '\\'
        prompt += f'Question: {question}\\Answer: [gMASK]'
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = self.tokenizer.build_inputs_for_generation(inputs, max_gen_length=1024)
        if self.device:
            inputs = inputs.to(self.device)
        outputs = self.model.generate(**inputs, max_length=1024, eos_token_id = self.tokenizer.eop_token_id, pad_token_id=self.tokenizer.eop_token_id)
        f = re.findall(r"<\|startofpiece\|>(.+)<\|endofpiece\|>", self.tokenizer.decode(outputs[0].tolist()))
        assert len(f) > 0
        return { "answer": f[0].strip(), "references": refs}
    
    def stream_query(self, question):
        refs = self.ref_retriever.query(question)
        if not refs:
            yield { "references": [], "answer": "" }
            return
        yield { "references": refs }
        prompt = ''
        for ix, ref in enumerate(refs):
            txt = ref["text"]
            prompt += f'Reference [{ix+1}]: {txt}' '\\'
        prompt += f'Question: {question}\\Answer: [gMASK]'
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = self.tokenizer.build_inputs_for_generation(inputs, max_gen_length=1024)
        if self.device:
            inputs = inputs.to(self.device)
        outputs = self.model.generate(**inputs, max_length=1024, eos_token_id = self.tokenizer.eop_token_id, pad_token_id=self.tokenizer.eop_token_id)
        f = re.findall(r"<\|startofpiece\|>(.+)<\|endofpiece\|>", self.tokenizer.decode(outputs[0].tolist()))
        assert len(f) > 0
        yield { "answer": f[0].strip() }


def load_model(args):
    webglm_ckpt_path = args.webglm_ckpt_path or os.getenv("WEBGLM_CKPT") or 'THUDM/WebGLM'
    retiever_ckpt_path = args.retriever_ckpt_path or os.getenv("WEBGLM_RETRIEVER_CKPT")
    if not retiever_ckpt_path:
        print('Retriever checkpoint not specified, please specify it with --retriever_ckpt_path or $WEBGLM_RETRIEVER_CKPT')
        exit(1)
    if args.serpapi_key:
        os.environ["SERPAPI_KEY"] = args.serpapi_key
    
    print('WebGLM Initializing...')
    
    webglm = WebGLM(webglm_ckpt_path, retiever_ckpt_path, args.device, args.filter_max_batch_size, args.searcher)
    
    print('WebGLM Loaded')
    
    return webglm