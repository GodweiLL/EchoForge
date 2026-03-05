import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import generate_image

# ── 1. 文生图 ──────────────────────────────────────────────
print("=== 文生图 ===")
result = generate_image.invoke({
    "prompt": "星际穿越，黑洞，黑洞里冲出一辆快支离破碎的复古列车，电影大片，末日既视感",
    "size": "2560x1440",
})
print(result)

# ── 2. 单张参考图生图 ───────────────────────────────────────
print("\n=== 单张参考图生图 ===")
result = generate_image.invoke({
    "prompt": "生成狗狗趴在草地上的近景画面",
    "image": "https://ark-project.tos-cn-beijing.volces.com/doc_image/seedream4_imageToimage.png",
    "size": "2560x1440",
})
print(result)

# ── 3. 多张参考图生图 ───────────────────────────────────────
print("\n=== 多张参考图生图 ===")
result = generate_image.invoke({
    "prompt": "将图1的服装换为图2的服装",
    "image": [
        "https://ark-project.tos-cn-beijing.volces.com/doc_image/seedream4_imagesToimage_1.png",
        "https://ark-project.tos-cn-beijing.volces.com/doc_image/seedream4_imagesToimage_2.png",
    ],
    "size": "2560x1440",
})
print(result)
