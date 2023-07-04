from playwright.sync_api import sync_playwright
from .searcher import *
from typing import List, Dict, Tuple, Optional

import json

def get_bing_search_raw_page(question: str):
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto(f"https://www.bing.com/search?q={question}")
        except:
            page.goto(f"https://www.bing.com")
            page.fill('input[name="q"]', question)
            page.press('input[name="q"]', 'Enter')
        try:
            page.wait_for_load_state('networkidle', timeout=3000)
        except:
            pass
        # page.wait_for_load_state('networkidle')
        search_results = page.query_selector_all('.b_algo h2')
        for result in search_results:
            title = result.inner_text()
            a_tag = result.query_selector('a')
            if not a_tag: continue
            url = a_tag.get_attribute('href')
            if not url: continue
            # print(title, url)
            results.append({
                'title': title,
                'url': url
            })
        browser.close()
    return results

def query_bing(question, max_tries=3):
    cnt = 0
    while cnt < max_tries:
        cnt += 1
        results = get_bing_search_raw_page(question)
        if results:
            return results
    print('No Bing Result')
    return None


if __name__ == '__main__':
    
    with open('crawl.json', 'w', encoding='utf-8') as f:
        json.dump(query_bing('how to cook a steak'), f, ensure_ascii=False, indent=4)
        
    exit(0)


class Searcher(SearcherInterface):
    def __init__(self) -> None:
        pass

    def _parse(self, result) -> List[SearchResult]:
        if not result:
            return None
        ret = []
        for item in result:
            ret.append(SearchResult(item['title'], item['url'], None))
        return ret

    def search(self, query) -> List[SearchResult]:
        return self._parse(query_bing(query))



if __name__ == '__main__':
    
    print(json.dumps(query_bing('how to cook a cake?'), ensure_ascii=False, indent=4))