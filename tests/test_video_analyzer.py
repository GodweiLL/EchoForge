import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import analyze_video

VIDEO_PATH = str(Path(__file__).parent.parent / "FRIEND OR BEST FRIEND？ [d_DCMhf9pWA].webm")

result = analyze_video.invoke({
    "video_path": VIDEO_PATH,
    "prompt": "请详细分析这段视频的内容和其中的音频。",
})
print(result)
