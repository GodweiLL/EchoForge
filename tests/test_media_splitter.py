import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import split_media

VIDEO_PATH = str(Path(__file__).parent.parent / "FRIEND OR BEST FRIEND？ [d_DCMhf9pWA].webm")

result = split_media.invoke({"video_path": VIDEO_PATH})
print(result)
