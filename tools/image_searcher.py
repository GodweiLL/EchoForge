"""Image searcher tool — uses Google Custom Search API to search for images."""

import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


@tool
def search_images(query: str, num: int = 5) -> list[dict]:
    """使用 Google 搜索图片，返回图片 URL 列表，可用作 generate_image 或 generate_video 的参考图。

    Args:
        query: 搜索关键词，建议用英文以获得更好结果。
        num: 返回结果数量，最多 10，默认 5。

    Returns:
        图片信息列表，每项包含 url / title / source / context_url。
    """
    num = max(1, min(10, num))
    resp = requests.get(
        _ENDPOINT,
        params={
            "key": os.environ["GOOGLE_API_KEY"],
            "cx": os.environ["GOOGLE_SEARCH_ENGINE_ID"],
            "q": query,
            "searchType": "image",
            "num": num,
        },
        timeout=15,
    )
    if not resp.ok:
        raise RuntimeError(f"Google 搜图失败 {resp.status_code}: {resp.text}")

    items = resp.json().get("items", [])
    return [
        {
            "url": item["link"],
            "title": item.get("title", ""),
            "source": item.get("displayLink", ""),
            "context_url": item.get("image", {}).get("contextLink", ""),
        }
        for item in items
    ]
