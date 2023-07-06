import json, os
import requests
from .searcher import *
from typing import List, Dict
    


def serp_api(query: str, api_key: str):
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key
    }
    resp = requests.get("https://serpapi.com/search", params=params)
    if resp.status_code != 200:
        raise Exception("Serpapi returned %d\n%s"%(resp.status_code, resp.text))
    result = resp.json()
    ret = []
    for item in result['organic_results']:
        if "title" not in item or "link" not in item or "snippet" not in item:
            continue
        ret.append(SearchResult(item['title'], item['link'], item['snippet']))
    return ret



def dump_results(results: List[SearchResult]):
    return json.dumps([result.dump() for result in results])


class Searcher(SearcherInterface):
    def __init__(self) -> None:
        self.SERPAPI_KEY = os.getenv("SERPAPI_KEY")
        if not self.SERPAPI_KEY:
            print("[Error] SERPAPI_KEY is not set, please set it to use serpapi")
            exit(1)

    def _parse(self, result) -> List[SearchResult]:
        if not result:
            return None
        ret = []
        for item in result:
            ret.append(SearchResult(item['ref'], item['url'], item['snip']))
        return ret

    def search(self, query) -> List[SearchResult]:
        return serp_api(query, self.SERPAPI_KEY)
