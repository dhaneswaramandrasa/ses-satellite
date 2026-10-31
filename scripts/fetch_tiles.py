"""Download Sentinel-2 cloudless 2016 tiles (EOX, CC BY 4.0) covering the study area and stitch them into one image.

python scripts/fetch_tiles.py  ->  data/raw/s2_malang_z15.jpg (+ .json with its Web Mercator bounds)
"""
import io
import json
import math
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
S, W, N, E = -8.05, 112.57, -7.90, 112.70
Z = 15
URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/{z}/{y}/{x}.jpg"


def tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


x0, y0 = (int(v) for v in tile(W, N, Z))
x1, y1 = (int(v) for v in tile(E, S, Z))
img = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
for x in range(x0, x1 + 1):
    for y in range(y0, y1 + 1):
        req = urllib.request.Request(URL.format(z=Z, x=x, y=y), headers={"User-Agent": "ses-satellite-demo"})
        for attempt in range(6):
            try:
                raw = urllib.request.urlopen(req, timeout=30).read()
                break
            except Exception:
                time.sleep(2 + attempt * 2)
        img.paste(Image.open(io.BytesIO(raw)).convert("RGB"), ((x - x0) * 256, (y - y0) * 256))
        time.sleep(0.05)
img.save(ROOT / "data/raw/s2_malang_z15.jpg", quality=90)
(ROOT / "data/raw/s2_malang_z15.json").write_text(json.dumps({"z": Z, "x0": x0, "y0": y0, "x1": x1, "y1": y1}))
print(img.size)
