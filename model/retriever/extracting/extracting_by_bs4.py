from bs4 import BeautifulSoup
import asyncio
import multiprocessing
import json
import os
import sys

def extracting(html: str) -> list[str]:
    html = html.replace("\n", " ")
    soup = BeautifulSoup(html, 'html.parser')
    raw = soup.find('body')
    if raw:
        raw = raw.get_text("\n")
    else:
        raw = soup.get_text("\n")
    paragraphs = []
    for item in raw.split("\n"):
        item = item.strip()
        if not item:
            continue
        paragraphs.append(item)
    return paragraphs