"""Image generator tool — uses Volcano Engine (火山引擎) doubao-seedream to generate images."""

import base64
import os
import uuid
from pathlib import Path
from typing import Optional, Union

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

_OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "images"


def _to_api_ref(path_or_url: str) -> str:
    """将本地路径或远程 URL 统一转为 base64 data URL。"""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = requests.get(path_or_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "image/jpeg")
        mime = ct.split(";")[0].strip()
        return f"data:{mime};base64,{base64.b64encode(resp.content).decode()}"
    p = Path(path_or_url)
    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


@tool
def generate_image(
    prompt: str,
    image: Optional[Union[str, list[str]]] = None,
    size: str = "2560x1440",
) -> str:
    """调用火山引擎生成图片，支持三种模式：
    1. 文生图：仅传 prompt
    2. 单张参考图生图：传 prompt + 单个图片本地路径
    3. 多张参考图生图：传 prompt + 图片本地路径列表

    Args:
        prompt: 图片生成的文字描述。
        image: 参考图片，可为单个本地路径或路径列表，不传则为纯文生图。
        size: 图片尺寸，格式为 '宽x高'，像素总数须 >= 3686400。
              例如 '2560x1440'(16:9 横屏)、'1440x2560'(9:16 竖屏)。默认 '2560x1440'。

    Returns:
        生成图片保存的本地路径。
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
        payload["image"] = (
            [_to_api_ref(i) for i in image] if isinstance(image, list) else _to_api_ref(image)
        )

    resp = requests.post(
        f"{base_url}/images/generations",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if not resp.ok:
        raise RuntimeError(f"生图失败 {resp.status_code}: {resp.text}")

    url = resp.json()["data"][0]["url"]

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_resp = requests.get(url, timeout=60)
    img_resp.raise_for_status()

    local_path = _OUTPUT_DIR / f"{uuid.uuid4().hex}.jpg"
    local_path.write_bytes(img_resp.content)

    return str(local_path)
