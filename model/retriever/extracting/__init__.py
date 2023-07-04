from .extracting_by_bs4 import extracting as bs4
from .html2text import html2text

from typing import List, Dict
import re

class Extractor:
    def __init__(self) -> None:
        pass
    
    def _pre_filter(self, paragraphs):
        # sorted_paragraphs = sorted(paragraphs, key=lambda x: len(x))
        # if len(sorted_paragraphs[-1]) < 10:
        #     return []
        ret = []
        for item in paragraphs:
            item = item.strip()
            item = re.sub(r"\[\d+\]", "", item) 
            if len(item) < 50:
                continue
            if len(item) > 1200:
                item = item[:1200] + "..."
            ret.append(item)
        return ret
    
    def extract_by_bs4(self, html) -> List[str]:
        return self._pre_filter(bs4(html))
    
    def extract_by_html2text(self, html) -> List[str]:
        return self._pre_filter(html2text(html).split("\n"))