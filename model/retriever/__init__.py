import json
from .searching import create_searcher
from .fetching import Fetcher
from .extracting import Extractor
from .filtering import ReferenceFilter

from typing import Optional, Union, List, Dict, Tuple, Iterable, Callable, Any

class ReferenceRetiever():
    def __init__(self, retriever_ckpt_path, device=None, filter_max_batch_size=400, searcher="serpapi") -> None:
        self.searcher = create_searcher(searcher)
        self.fetcher = Fetcher()
        self.extractor = Extractor()
        self.filter = ReferenceFilter(retriever_ckpt_path, device, filter_max_batch_size)

    def query(self, question) -> List[Dict[str, str]]:
        print("[System] Searching ...")
        search_results = self.searcher.search(question)
        urls = [result.url for result in search_results]
        titles = {result.url: result.title for result in search_results}
        print("[System] Count of available urls: ", len(urls))
        if len(urls) == 0:
            print("[System] No available urls. Please check your network connection.")
            return None
            
        print("[System] Fetching ...")
        fetch_results = self.fetcher.fetch(urls)
        cnt = sum([len(fetch_results[key]) for key in fetch_results])
        print("[System] Count of available fetch results: ", cnt)
        if cnt == 0:
            print("[System] No available fetch results. Please check playwright or your network.")
            return None
            
        print("[System] Extracting ...")
        data_list = []
        for url in fetch_results:
            extract_results = self.extractor.extract_by_html2text(fetch_results[url])
            for value in extract_results:
                data_list.append({
                    "url": url,
                    "title": titles[url],
                    "text": value
                })
        print("[System] Count of paragraphs: ", len(data_list))
        if len(data_list) == 0:
            print("[System] No available paragraphs. The references provide no useful information.")
            return None
        
        print("[System] Filtering ...")
        return self.filter.produce_references(question, data_list, 5)