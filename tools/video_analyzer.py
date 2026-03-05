"""Video analyzer tool — uses Gemini via OpenRouter to analyze local video files or YouTube URLs."""

import base64
import os
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from yt_dlp import YoutubeDL

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


def _download_video(url: str) -> tuple[str, str]:
    """用 yt-dlp 将 URL 下载到临时目录，返回 (视频文件路径, 临时目录路径)。"""
    tmpdir = tempfile.mkdtemp(prefix="echoforge_")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(tmpdir, "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)
    return video_path, tmpdir


def _cleanup(tmpdir: str) -> None:
    """删除临时目录及其中所有文件。"""
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@tool
def analyze_video(video_path: str) -> str:
    """分析本地视频文件或 YouTube 视频链接，包括视频画面（细化到秒）和音频内容。返回大模型对视频的详细分析报告。

    Args:
        video_path: 本地视频文件的绝对或相对路径，或 YouTube / 其他平台视频 URL。
    """
    tmpdir = None

    try:
        # 若为 URL，先用 yt-dlp 下载到临时目录
        if video_path.startswith("http://") or video_path.startswith("https://"):
            video_path, tmpdir = _download_video(video_path)

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

    finally:
        # 无论成功还是失败，都清理临时文件
        if tmpdir:
            _cleanup(tmpdir)
