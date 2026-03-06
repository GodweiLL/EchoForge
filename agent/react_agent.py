"""EchoForge ReAct Agent — multimodal media assistant powered by Gemini via OpenRouter."""

import base64
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from agent.prompts import SYSTEM_PROMPT
from tools import concat_videos, generate_image, generate_video, analyze_video, check_subjects, search_images

load_dotenv()

_TOOLS = [analyze_video, check_subjects, search_images, generate_image, generate_video, concat_videos]

_MIME_MAP = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mov": "video/mov",
    ".webm": "video/webm",
}


def build_agent():
    llm = ChatOpenAI(
        model=os.environ["MODEL"],
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ["OPENROUTER_API_BASE"],
    )
    return create_agent(
        llm,
        _TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )


def build_user_message(text: str, video_path: Optional[str] = None) -> HumanMessage:
    """构建用户消息，可选附带本地视频（base64 编码后随消息发送给模型）。"""
    if not video_path:
        return HumanMessage(content=text)

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    mime = _MIME_MAP.get(path.suffix.lower(), "video/mp4")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    return HumanMessage(content=[
        {"type": "text", "text": text},
        {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{b64}"}},
    ])
