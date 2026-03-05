import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import generate_video

result = generate_video.invoke({
    "prompt": "无人机以极快速度穿越复杂障碍或自然奇观，带来沉浸式飞行体验",
    "image_url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/seepro_i2v.png",
    "duration": 5,
    "resolution": "720p",
    "ratio": "16:9",
})
print(result)
