"""Video generator tool — uses Volcano Engine (火山引擎) doubao-seedance to generate videos.

Async workflow: submit task → poll until done → return video URL.
"""

import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

_VOLC_VIDEO_MODEL = "doubao-seedance-1-5-pro-251215"
_TASK_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['VOLC_API_KEY']}",
        "Content-Type": "application/json",
    }


@tool
def generate_video(
    prompt: str,
    image_url: Optional[str] = None,
    duration: int = 5,
    resolution: str = "720p",
    ratio: str = "16:9",
) -> str:
    """调用火山引擎 doubao-seedance 生成视频（异步，自动轮询直到完成）。

    Args:
        prompt: 视频内容的文字描述。
        image_url: 参考图片 URL，提供则为图生视频，不提供则为文生视频。
        duration: 视频时长（秒），范围 2-12，默认 5。
        resolution: 分辨率，可选 '480p' / '720p' / '1080p'，默认 '720p'。
        ratio: 画面比例，可选 '16:9' / '9:16' / '1:1' / '4:3'，默认 '16:9'。

    Returns:
        生成视频的 URL。
    """
    full_prompt = f"{prompt} --dur {duration} --rs {resolution} --rt {ratio}"

    content = [{"type": "text", "text": full_prompt}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})

    # 提交任务
    resp = requests.post(
        _TASK_URL,
        headers=_headers(),
        json={"model": _VOLC_VIDEO_MODEL, "content": content},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"提交任务失败 {resp.status_code}: {resp.text}")

    task_id = resp.json()["id"]
    print(f"任务已提交，task_id: {task_id}")

    # 轮询任务状态
    while True:
        time.sleep(5)
        query = requests.get(
            f"{_TASK_URL}/{task_id}",
            headers=_headers(),
            timeout=30,
        )
        if not query.ok:
            raise RuntimeError(f"查询任务失败 {query.status_code}: {query.text}")

        result = query.json()
        status = result.get("status")
        print(f"任务状态: {status}")

        if status == "succeeded":
            return result["content"]["video_url"]
        elif status in ("failed", "cancelled"):
            raise RuntimeError(f"任务失败: {result}")
