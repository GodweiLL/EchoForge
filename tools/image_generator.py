"""Image generator tool — uses Volcano Engine (火山引擎) doubao-seedream to generate images.

Supports three modes:
  1. Text-to-image: only prompt provided
  2. Single reference image: prompt + one image URL
  3. Multiple reference images: prompt + list of image URLs
"""

import os
from typing import Optional, Union

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


@tool
def generate_image(
    prompt: str,
    image: Optional[Union[str, list[str]]] = None,
    size: str = "2560x1440",
) -> str:
    """调用火山引擎生成图片，支持三种模式：
    1. 文生图：仅传 prompt
    2. 单张参考图生图：传 prompt + 单个图片 URL
    3. 多张参考图生图：传 prompt + 图片 URL 列表

    Args:
        prompt: 图片生成的文字描述。
        image: 参考图片，可为单个 URL 字符串或 URL 列表，不传则为纯文生图。
        size: 图片尺寸，格式为 '宽x高'，像素总数须 >= 3686400。
              例如 '2560x1440'(16:9 横屏)、'1440x2560'(9:16 竖屏)、'2048x2048'(1:1 方图)。
              宽高比范围 [1/16, 16]，最大 4096x4096。默认 '2560x1440'。

    Returns:
        生成图片的 URL。
    """
    api_key = os.environ["VOLC_API_KEY"]
    base_url = os.environ.get("VOLC_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
    model = os.environ.get("VOLC_IMAGE_MODEL", "doubao-seedream-5-0-260128")

    payload: dict = {
        "model": model,
        "prompt": prompt,
        "sequential_image_generation": "disabled",
        "response_format": "url",
        "size": size,
        "stream": False,
        "watermark": True,
    }

    if image is not None:
        payload["image"] = image

    response = requests.post(
        f"{base_url}/images/generations",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(f"API 错误 {response.status_code}: {response.text}")
    response.raise_for_status()

    data = response.json()
    return data["data"][0]["url"]
