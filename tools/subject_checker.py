"""Subject checker tool — uses Gemini via OpenRouter to audit subject completeness for a storyboard."""

import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

_PROMPT_TMPL = """你是一个专业的视频创作审查员。

## 剧本 / 分镜方案
{storyboard}

## 当前已规划的主体列表（将为其生成参考图）
{subjects}

## 任务
审查上述主体列表是否足以覆盖剧本中出现的所有需要保持视觉一致性的主体（人物、动物、标志性物体、特定场景等）。

请输出：
1. **已覆盖的主体**：列出已规划主体中与剧本匹配的部分。
2. **缺失的主体**：列出剧本中出现但未在主体列表中的主体，说明其在哪个分镜出现、为何需要参考图。
3. **建议**：对每个缺失主体，给出应如何补充（描述、生图提示词方向）。
4. **结论**：当前主体列表是否足够？若不足请明确指出需要补充哪些。
"""


@tool
def check_subjects(storyboard: str, subjects: list[str]) -> str:
    """审查为完成分镜方案，已规划的主体参考图是否足够。

    Args:
        storyboard: 完整的分镜方案描述，包含每个镜头的内容。
        subjects: 当前已规划要生成参考图的主体列表。
    """
    api_key = os.environ["OPENROUTER_API_KEY"]
    base_url = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    model = os.environ.get("MODEL_VIDEO", "google/gemini-3-flash-preview-20251217")

    subjects_text = "\n".join(f"- {s}" for s in subjects) if subjects else "（无）"
    prompt = _PROMPT_TMPL.format(storyboard=storyboard, subjects=subjects_text)

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
