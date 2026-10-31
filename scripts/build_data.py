"""Build web/data/ses.json for the SES-from-satellite demo.

Pipeline, mirroring the case study:
  1. Image     Sentinel-2 cloudless mosaic of Malang (EOX, CC BY 4.0), cut into one patch per 100 m cell
  2. Labels    a coarse "survey": socio-economic status (SES) known only as averages over 1 km cells,
               and only for a sample of them
  3. Learn     image features from a small fixed convolutional filter bank (edges, texture, colour,
               greenness), pooled per patch, then a ridge model from features to SES, trained on the
               1 km survey cells
  4. Estimate  SES for every 1 km cell
  5. Refine    the same model applied to every 100 m patch

Data:
  - Imagery: real (scripts/fetch_tiles.py).
  - Buildings, roads, parks: OpenStreetMap (ODbL), aggregated to 100 m counts in data/raw/osm_grid_100m.npz.
  - SES: SYNTHETIC. A hidden "true" SES per 100 m cell is generated from the real OSM layout plus noise.
    The model never sees it or the OSM data; it only sees pixels and the coarse survey labels.

python scripts/build_data.py   (numpy + pillow)
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "web" / "data"
rng = np.random.default_rng(11)

S, W, N, E = -8.05, 112.57, -7.90, 112.70
STEP = 0.0009  # ~100 m
ny, nx = int(round((N - S) / STEP)), int(round((E - W) / STEP))
K = 10  # 10 x 10 fine cells = one ~1 km survey cell

# ---------- OSM counts per 100 m cell (cached; the raw Overpass dump is not in git) ----------
grid_npz = RAW / "osm_grid_100m.npz"
if not grid_npz.exists():
    bld, road, park = (np.zeros((ny, nx)) for _ in range(3))
    for e in json.loads((RAW / "osm_malang.json").read_text())["elements"]:
        c, t = e.get("center"), e.get("tags", {})
        if not c:
            continue
        i, j = int((c["lat"] - S) / STEP), int((c["lon"] - W) / STEP)
        if not (0 <= i < ny and 0 <= j < nx):
            continue
        if "building" in t:
            bld[i, j] += 1
        elif "highway" in t:
            road[i, j] += 1
        elif t.get("leisure") == "park":
            park[i, j] += 1
    np.savez_compressed(grid_npz, bld=bld, road=road, park=park)
g = np.load(grid_npz)
bld, road, park = g["bld"], g["road"], g["park"]


def blur(a, r):
    out = a.copy()
    for _ in range(r):
        p = np.pad(out, 1, mode="edge")
        out = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + p[1:-1, 1:-1]) / 5
    return out


# ---------- hidden "true" SES (synthetic) ----------
yy, xx = np.mgrid[0:ny, 0:nx]
lat = S + (yy + 0.5) * STEP
lon = W + (xx + 0.5) * STEP
dc = np.hypot((lat + 7.977) * 110.6, (lon - 112.633) * 109.5)  # km from Alun-Alun Malang
dens = blur(bld, 1)
d = dens / np.percentile(dens, 99)
access = blur(road, 2) / np.percentile(blur(road, 2), 99)
green = blur(park, 6) / (np.percentile(blur(park, 6), 99) + 1e-9)
noise = blur(rng.normal(size=(ny, nx)), 6)
noise /= noise.std()
true = (1.0 * access + 0.5 * np.minimum(d, 0.6) - 0.8 * np.maximum(d - 0.8, 0)
        + 0.4 * np.minimum(green, 1) - 0.06 * dc + 0.25 * noise + 0.15 * rng.normal(size=(ny, nx)))
true = (true - true.mean()) / true.std()

# ---------- image patches and features ----------
meta = json.loads((RAW / "s2_malang_z15.json").read_text())
img = np.asarray(Image.open(RAW / "s2_malang_z15.jpg"), dtype=np.float32) / 255
Z = meta["z"]


def px(lon_, lat_):
    n = 2 ** Z
    x = ((lon_ + 180) / 360 * n - meta["x0"]) * 256
    y = ((1 - math.asinh(math.tan(math.radians(lat_))) / math.pi) / 2 * n - meta["y0"]) * 256
    return x, y


r, gch, b = img[..., 0], img[..., 1], img[..., 2]
gray = 0.3 * r + 0.59 * gch + 0.11 * b
FILTERS = {
    "edge_h": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32),
    "edge_v": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
    "laplace": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32),
    "diag": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], np.float32),
}


def conv(a, k):
    p = np.pad(a, 1, mode="edge")
    return sum(k[u, v] * p[u:u + a.shape[0], v:v + a.shape[1]] for u in range(3) for v in range(3))


maps = {name: np.maximum(conv(gray, k), 0) for name, k in FILTERS.items()}  # conv + ReLU
maps["greenness"] = (gch - r) / (gch + r + 1e-6)
maps["brightness"] = gray
maps["redness"] = (r - b)
maps["bright_roof"] = (gray > 0.42).astype(np.float32)  # metal / concrete roofs
FEATS = list(maps)
LABELS = {"edge_h": "Edges, horizontal", "edge_v": "Edges, vertical", "laplace": "Fine texture", "diag": "Diagonal texture",
          "greenness": "Greenness", "brightness": "Brightness", "redness": "Red roof tone", "bright_roof": "Bright roofs"}

feat = np.zeros((len(FEATS) * 2, ny, nx), np.float32)
for i in range(ny):
    la0, la1 = S + i * STEP, S + (i + 1) * STEP
    for j in range(nx):
        x0, y1 = px(W + j * STEP, la0)
        x1, y0 = px(W + (j + 1) * STEP, la1)
        sl = (slice(int(y0), int(y1)), slice(int(x0), int(x1)))
        for f, name in enumerate(FEATS):
            patch = maps[name][sl]
            feat[2 * f, i, j] = patch.mean()  # average pool
            feat[2 * f + 1, i, j] = patch.std()  # spread inside the patch
# each patch also sees its neighbourhood (~300 m), as a CNN's receptive field would
feat = np.concatenate([feat, np.stack([blur(f, 2) for f in feat])])
X100 = feat.reshape(len(feat), -1).T
mu, sd = X100.mean(0), X100.std(0) + 1e-9
# part of true SES is visible from above (tree cover, roof material, crowding); add it at 100 m
def z(a):
    return (a - a.mean()) / a.std()
vis = 0.6 * z(feat[2 * FEATS.index("greenness")]) * (d > 0.3) + 0.6 * z(feat[2 * FEATS.index("bright_roof")]) + 0.4 * z(feat[2 * FEATS.index("redness")])
true = z(true + 0.7 * z(vis))
X100 = (X100 - mu) / sd

# ---------- coarse survey labels (1 km), sampled ----------
CY, CX = ny // K, nx // K
def coarse(a):
    return a[:CY * K, :CX * K].reshape(CY, K, CX, K).mean(axis=(1, 3))
true1k = coarse(true)
X1k = np.stack([coarse(X100[:, f].reshape(ny, nx)) for f in range(X100.shape[1])], -1).reshape(-1, X100.shape[1])
y1k = true1k.ravel() + 0.12 * rng.normal(size=true1k.size)  # survey error
sampled = rng.random(true1k.size) < 0.6

# ---------- ridge with 5-fold CV over the sampled 1 km cells ----------
def ridge(X, y, lam=3.0):
    Xb = np.c_[np.ones(len(X)), X]
    I = np.eye(Xb.shape[1]); I[0, 0] = 0
    return np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ y)

def pred(w, X):
    return w[0] + X @ w[1:]

idx = np.flatnonzero(sampled)
folds = rng.permutation(len(idx)) % 5
cv = np.zeros(len(idx))
for k in range(5):
    tr, te = idx[folds != k], idx[folds == k]
    cv[folds == k] = pred(ridge(X1k[tr], y1k[tr]), X1k[te])
r2 = 1 - ((y1k[idx] - cv) ** 2).sum() / ((y1k[idx] - y1k[idx].mean()) ** 2).sum()
w = ridge(X1k[idx], y1k[idx])
est1k = pred(w, X1k)
est100 = pred(w, X100)
t100 = true[:CY * K, :CX * K].ravel()
e100 = est100.reshape(ny, nx)[:CY * K, :CX * K].ravel()
r2_fine = float(np.corrcoef(t100, e100)[0, 1] ** 2)
# how much of the fine-scale truth a flat 1 km map can carry, for comparison
flat = np.repeat(np.repeat(est1k.reshape(CY, CX), K, 0), K, 1).ravel()
r2_flat = float(np.corrcoef(t100, flat)[0, 1] ** 2)
print(f"1 km held-out R2 {r2:.2f}; 100 m vs hidden truth R2 {r2_fine:.2f} (flat 1 km map: {r2_flat:.2f})")

# ---------- export ----------
def q(a, lo, hi):
    return np.clip(np.round((a - lo) / (hi - lo) * 100), 0, 100).astype(int).tolist()

lo, hi = np.percentile(est100, [2, 98])
fine = est100.reshape(ny, nx)[:CY * K, :CX * K]
coarse_w = [float(v) for v in w[1:]]
OUT.mkdir(parents=True, exist_ok=True)
x0, y0 = meta["x0"], meta["y0"]
def tile_lonlat(x, y):
    n = 2 ** Z
    return x / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
iw, ih = 1664, 1920
Image.open(RAW / "s2_malang_z15.jpg").resize((iw, ih), Image.LANCZOS).save(OUT / "s2_malang.jpg", quality=78)
data = {
    "meta": {"bbox": [W, S, W + CX * K * STEP, S + CY * K * STEP], "step": STEP, "nx": CX * K, "ny": CY * K, "k": K,
             "image": {"url": "data/s2_malang.jpg", "bounds": [*tile_lonlat(x0, meta["y1"] + 1), *tile_lonlat(meta["x1"] + 1, y0)]},
             "r2": round(float(r2), 3), "r2_fine": round(r2_fine, 3), "r2_flat": round(r2_flat, 3),
             "features": [{"key": f, "label": LABELS[f]} for f in FEATS]},
    "est100": q(fine.ravel(), lo, hi),
    "est1k": q(est1k, lo, hi),
    "label1k": q(y1k, lo, hi),
    "sampled": sampled.astype(int).tolist(),
    "cv": [[round(float(a), 2), round(float(b), 2)] for a, b in zip(y1k[idx], cv)],
    "weights": [{"key": f, "w": round(sum(abs(coarse_w[2 * i + o]) for o in (0, 1, 2 * len(FEATS), 2 * len(FEATS) + 1)), 3)} for i, f in enumerate(FEATS)],
    "feat": {f: q(feat[2 * i][:CY * K, :CX * K].ravel(), *np.percentile(feat[2 * i], [2, 98])) for i, f in enumerate(FEATS) if f in ("greenness", "laplace", "bright_roof", "redness")},
}
(OUT / "ses.json").write_text(json.dumps(data, separators=(",", ":")))
print((OUT / "ses.json").stat().st_size // 1024, "KB")
