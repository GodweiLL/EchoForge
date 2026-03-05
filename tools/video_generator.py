"""Video generator tool — uses Volcano Engine (火山引擎) doubao-seedance to generate videos."""

import base64
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

_VOLC_VIDEO_MODEL = "doubao-seedance-1-5-pro-251215"
_TASK_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
_OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "videos"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['VOLC_API_KEY']}",
        "Content-Type": "application/json",
    }


def _image_to_data_url(path: str) -> str:
    p = Path(path)
    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


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
        image_url: 参考图片的本地路径（由 generate_image 返回），提供则为图生视频，不提供则为文生视频。
        duration: 视频时长（秒），范围 2-12，默认 5。
        resolution: 分辨率，可选 '480p' / '720p' / '1080p'，默认 '720p'。
        ratio: 画面比例，可选 '16:9' / '9:16' / '1:1' / '4:3'，默认 '16:9'。

    Returns:
        生成视频保存的本地路径。
    """
    duration = int(max(2, min(12, duration)))
    full_prompt = f"{prompt} --dur {duration} --rs {resolution} --rt {ratio}"

    content = [{"type": "text", "text": full_prompt}]
    if image_url:
        ref = _image_to_data_url(image_url) if not image_url.startswith("http") else image_url
        content.append({"type": "image_url", "image_url": {"url": ref}})

    resp = requests.post(
        _TASK_URL,
        headers=_headers(),
        json={"model": _VOLC_VIDEO_MODEL, "content": content},
        timeout=30,
    )
    if not resp.ok:
        err_msg = resp.json().get("error", {}).get("message", resp.text)
        raise RuntimeError(
            f"提交失败 HTTP {resp.status_code}: {err_msg}\n"
            f"实际发送参数: dur={duration} rs={resolution} rt={ratio}\n"
            f"prompt 后缀: ...{full_prompt[-100:]}"
        )

    task_id = resp.json()["id"]
    print(f"[video] 已提交 task_id={task_id} | dur={duration} {ratio} {resolution}")

    while True:
        time.sleep(5)
        query = requests.get(f"{_TASK_URL}/{task_id}", headers=_headers(), timeout=30)
        if not query.ok:
            raise RuntimeError(f"查询失败 HTTP {query.status_code}: {query.text}")

        result = query.json()
        status = result.get("status")

        if status == "succeeded":
            video_url = result["content"]["video_url"]
            _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            vid_resp = requests.get(video_url, timeout=300)
            vid_resp.raise_for_status()
            local_path = _OUTPUT_DIR / f"{uuid.uuid4().hex}.mp4"
            local_path.write_bytes(vid_resp.content)
            print(f"[video] 完成 task_id={task_id} → {local_path}")
            return str(local_path)
        elif status in ("failed", "cancelled"):
            err = result.get("error", {})
            raise RuntimeError(f"任务{status} task_id={task_id}: {err.get('message', result)}")
