from .playwright_based_crawl_new import get_raw_pages
from .import playwright_based_crawl_new

import asyncio
    
from typing import List, Dict

class Fetcher:
    def __init__(self) -> None:
        self.loop = asyncio.get_event_loop()
        # TODO delete loop -> loop.close()

    
    def _pre_handle_urls(self, urls: List[str]) -> List[str]:
        urls_new = []
        for url in urls:
            if url in urls_new or "http://%s"%url in urls_new or "https://%s"%url in urls_new:
                continue
            if not url.startswith("http"):
                url = "http://%s" % url
            urls_new.append(url)
        return urls_new

    def fetch(self, urls: List[str]) -> Dict[str, List[str]]:
        
        urls = self._pre_handle_urls(urls)
        
        self.loop.run_until_complete(get_raw_pages(urls, close_browser=True))
        responses = [playwright_based_crawl_new.results[url] for url in urls] 

        ret = dict()
        for url, resp in zip(urls, responses):
            if not resp[1]:
                pass
            else:
                ret[url] = resp[1]

        return ret
