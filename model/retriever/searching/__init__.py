import json, os
import requests

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
if not SERPAPI_KEY:
    print("[Error] SERPAPI_KEY is not set, please set it to use serpapi")
    exit(0)
    


def serp_api(query: str):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY
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

class SearchResult:
    def __init__(self, title, url, snip) -> None:
        self.title = title
        self.url = url
        self.snip = snip

    def dump(self):
        return {
            "title": self.title,
            "url": self.url,
            "snip": self.snip
        }

    def __str__(self) -> str:
        return json.dumps(self.dump())


def dump_results(results: list[SearchResult]):
    return json.dumps([result.dump() for result in results])


class Searcher:
    def __init__(self) -> None:
        pass

    def _parse(self, result) -> list[SearchResult]:
        if not result:
            return None
        ret = []
        for item in result:
            ret.append(SearchResult(item['ref'], item['url'], item['snip']))
        return ret

    def search(self, query) -> list[SearchResult]:
        return serp_api(query)
