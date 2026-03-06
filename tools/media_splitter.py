"""Media tools — uses ffmpeg to split and concatenate video files."""

import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


@tool
def split_media(
    video_path: str,
    output_dir: Optional[str] = None,
) -> str:
    """使用 ffmpeg 将视频文件分离为独立的视频轨和音频轨，输出到指定文件夹。

    Args:
        video_path: 本地视频文件的绝对或相对路径。
        output_dir: 输出文件夹路径，默认在视频同级目录下创建 <视频名>_split 文件夹。
    """
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    out_dir = Path(output_dir).resolve() if output_dir else path.parent / f"{path.stem}_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_out = out_dir / f"{path.stem}_video{path.suffix}"
    audio_out = out_dir / f"{path.stem}_audio.mp3"

    # 提取视频轨（去掉音频）
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-an", "-c:v", "copy", str(video_out)],
        check=True,
        capture_output=True,
    )

    # 提取音频轨（去掉视频）
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-vn", "-c:a", "libmp3lame", str(audio_out)],
        check=True,
        capture_output=True,
    )

    return (
        f"分离完成，输出目录: {out_dir}\n"
        f"  视频轨: {video_out.name}\n"
        f"  音频轨: {audio_out.name}"
    )


def _get_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


@tool
def concat_videos(
    video_paths: list[str],
    output_path: str,
    transitions: Optional[list[str]] = None,
    transition_duration: float = 1.0,
) -> str:
    """使用 ffmpeg 将多个视频按顺序拼接，可为每两个片段之间单独设置转场特效。

    Args:
        video_paths: 要拼接的视频路径列表（至少 2 个），按顺序排列。
        output_path: 输出视频文件路径（含文件名）。
        transitions: 转场列表，长度须为 len(video_paths)-1，每个元素对应相邻两段之间的转场。
                     "none" 或 None 元素表示硬切，其余可选：
                     fade / fadeblack / fadewhite / dissolve /
                     wipeleft / wiperight / wipeup / wipedown /
                     slideleft / slideright / slideup / slidedown /
                     circleopen / circleclose / radial / pixelize /
                     diagtl / diagtr / diagbl / diagbr
                     不传则所有片段硬切。
        transition_duration: 所有转场的时长（秒），默认 1.0。
    """
    if len(video_paths) < 2:
        raise ValueError("至少需要提供 2 个视频文件")

    # URL 自动下载到临时文件
    _tmp_files = []
    resolved = []
    for p in video_paths:
        if p.startswith("http://") or p.startswith("https://"):
            suffix = Path(p.split("?")[0]).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            urllib.request.urlretrieve(p, tmp.name)
            _tmp_files.append(tmp.name)
            resolved.append(tmp.name)
        else:
            if not Path(p).exists():
                raise FileNotFoundError(f"视频文件不存在: {p}")
            resolved.append(p)
    video_paths = resolved

    n = len(video_paths)
    td = transition_duration

    # 补全 transitions 列表
    if transitions is None:
        transitions = [None] * (n - 1)
    if len(transitions) != n - 1:
        raise ValueError(f"transitions 长度须为 {n - 1}，当前为 {len(transitions)}")
    # 统一 "none" 字符串为 None
    transitions = [None if (t is None or t == "none") else t for t in transitions]

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # 全部硬切：用 concat demuxer（最快）
    if all(t is None for t in transitions):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for p in video_paths:
                f.write(f"file '{Path(p).resolve()}'\n")
            list_file = f.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", str(out)],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg 失败 (exit {e.returncode}):\n{e.stderr}") from e
        finally:
            Path(list_file).unlink(missing_ok=True)
            for f in _tmp_files:
                Path(f).unlink(missing_ok=True)
        return f"拼接完成，输出文件: {out}"

    # 含转场：用 filter_complex
    durations = [_get_duration(p) for p in video_paths]
    inputs = []
    for p in video_paths:
        inputs += ["-i", p]

    parts = []
    for i in range(n):
        parts.append(f"[{i}:v]setsar=1,settb=AVTB[v{i}]")

    cumulative = 0.0
    v_prev = "v0"
    a_prev = "0:a"

    for i in range(1, n):
        t = transitions[i - 1]
        v_out = "vout" if i == n - 1 else f"vx{i}"
        a_out = "aout" if i == n - 1 else f"ax{i}"

        if t is None:
            # 硬切：concat filter
            cumulative += durations[i - 1]
            parts.append(f"[{v_prev}][v{i}]concat=n=2:v=1:a=0[{v_out}]")
            parts.append(f"[{a_prev}][{i}:a]concat=n=2:v=0:a=1[{a_out}]")
        else:
            # 转场：xfade + acrossfade
            cumulative += durations[i - 1] - td
            parts.append(
                f"[{v_prev}][v{i}]xfade=transition={t}:duration={td:.3f}:offset={cumulative:.3f}[{v_out}]"
            )
            parts.append(f"[{a_prev}][{i}:a]acrossfade=d={td:.3f}[{a_out}]")

        v_prev = v_out
        a_prev = a_out

    try:
        subprocess.run(
            ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", "; ".join(parts),
                "-map", "[vout]", "-map", "[aout]",
                str(out),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 失败 (exit {e.returncode}):\n{e.stderr}") from e
    finally:
        for f in _tmp_files:
            Path(f).unlink(missing_ok=True)
    return f"拼接完成，输出文件: {out}"
