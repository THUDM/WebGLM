import torch
from transformers import AutoTokenizer, AutoModel
import os

from typing import Optional, Union, List, Dict, Tuple, Iterable, Callable, Any

class ContrieverScorer:
    def __init__(self, retriever_ckpt_path, device=None, max_batch_size=400) -> None:
        query_encoder_path = os.path.join(retriever_ckpt_path, 'query_encoder')
        reference_encoder_path = os.path.join(retriever_ckpt_path, 'reference_encoder')
            
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        self.query_encoder = AutoModel.from_pretrained(query_encoder_path)
        self.reference_encoder = AutoModel.from_pretrained(reference_encoder_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if not device else device
        self.query_encoder = self.query_encoder.to(self.device).eval()
        self.reference_encoder = self.reference_encoder.to(self.device).eval()
        assert max_batch_size > 0
        self.max_batch_size = max_batch_size

    def get_query_embeddings(self, sentences: List[str]) -> torch.Tensor:
        # Tokenization and Inference
        torch.cuda.empty_cache()
        with torch.no_grad():
            inputs = self.tokenizer(sentences, padding=True,
                                    truncation=True, return_tensors='pt')
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            outputs = self.query_encoder(**inputs)
            # Mean Pool
            token_embeddings = outputs[0]
            mask = inputs["attention_mask"]
            token_embeddings = token_embeddings.masked_fill(
                ~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(
                dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings
    
    def get_embeddings(self, sentences: List[str]) -> torch.Tensor:
        # Tokenization and Inference
        torch.cuda.empty_cache()
        with torch.no_grad():
            inputs = self.tokenizer(sentences, padding=True,
                                    truncation=True, return_tensors='pt')
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            outputs = self.reference_encoder(**inputs)
            # Mean Pool
            token_embeddings = outputs[0]
            mask = inputs["attention_mask"]
            token_embeddings = token_embeddings.masked_fill(
                ~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(
                dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings

    def score_documents_on_query(self, query: str, documents: List[str]) -> torch.Tensor:
        query_embedding = self.get_query_embeddings([query])[0]
        document_embeddings = self.get_embeddings(documents)
        return query_embedding@document_embeddings.t()

    def select_topk(self, query: str, documents: List[str], k=1):
        """
        Returns:
            `ret`: `torch.return_types.topk`, use `ret.values` or `ret.indices` to get value or index tensor
        """
        scores = []
        for i in range((len(documents) + self.max_batch_size - 1) // self.max_batch_size):
            scores.append(self.score_documents_on_query(query, documents[self.max_batch_size*i:self.max_batch_size*(i+1)]).to('cpu'))
        scores = torch.concat(scores)
        return scores.topk(min(k, len(scores)))


class ReferenceFilter:
    def __init__(self, retriever_ckpt_path, device=None, max_batch_size=400) -> None:
        self.scorer = ContrieverScorer(retriever_ckpt_path, device, max_batch_size)

    def produce_references(self, query, paragraphs: List[Dict[str, str]], topk=5) -> List[Dict[str, str]]:
        """Individually calculate scores of each sentence, and return `topk`. paragraphs should be like a list of {title, url, text}."""
        # paragraphs = self._pre_filter(paragraphs)
        texts = [item['text'] for item in paragraphs]
        topk = self.scorer.select_topk(query, texts, topk)
        indices = list(topk.indices.detach().cpu().numpy())
        return [paragraphs[idx] for idx in indices]


