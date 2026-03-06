import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import search_images

results = search_images.invoke({"query": "bubble tea cup close up", "num": 3})
for url in results:
    print(url)
