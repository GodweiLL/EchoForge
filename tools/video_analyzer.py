"""Video analyzer tool — uses Gemini via OpenRouter to analyze local video files."""

import base64
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

_MIME_FALLBACK = "video/mp4"
_SUPPORTED_MIME = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".mov": "video/mov",
    ".webm": "video/webm",
}

_PROMPT = "请详细分析这段视频的内容（细化到秒）和其中的音频（若有）。"


@tool
def analyze_video(video_path: str) -> str:
    """分析本地视频文件，包括视频画面（细化到秒）和音频内容。返回大模型对视频的详细分析报告。

    Args:
        video_path: 本地视频文件的绝对或相对路径。
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    ext = path.suffix.lower()
    mime_type = _SUPPORTED_MIME.get(ext, _MIME_FALLBACK)

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    data_url = f"data:{mime_type};base64,{b64}"

    api_key = os.environ["OPENROUTER_API_KEY"]
    base_url = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    model = os.environ.get("MODEL_VIDEO", "google/gemini-3-flash-preview-20251217")

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {"type": "video_url", "video_url": {"url": data_url}},
                    ],
                }
            ],
        },
        timeout=120,
    )
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
