"""25강 그림: 변환 종류별 사각형 변형, 쌍선형 보간 도식, 리샘플링 방식 비교(PNG)"""
import math, os, sys
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")

# ---------------------------------------------------------------- 변환 종류별 사각형 변형
s = Svg(760, 222, "변환 종류별로 정사각형이 변하는 모습과 자유도, 최소 대응점 수")
SQ = [(-35, -35), (35, -35), (35, 35), (-35, 35)]


def rot(p, deg, sc=1.0):
    a = math.radians(deg)
    x, y = p
    return (sc * (x * math.cos(a) - y * math.sin(a)), sc * (x * math.sin(a) + y * math.cos(a)))


cases = [
    ("이동", "자유도 2 · 점 1", [(x + 18, y - 12) for x, y in SQ]),
    ("강체", "자유도 3 · 점 2", [(rot(p, 25)[0] + 6, rot(p, 25)[1]) for p in SQ]),
    ("유사", "자유도 4 · 점 2", [rot(p, 25, 0.72) for p in SQ]),
    ("어파인", "자유도 6 · 점 3", [(x + 0.45 * y, 0.1 * x + 0.8 * y) for x, y in SQ]),
    ("호모그래피", "자유도 8 · 점 4", [(-25, -30), (28, -40), (44, 34), (-42, 28)]),
]
for i, (name, sub, poly) in enumerate(cases):
    cx, cy = 76 + 152 * i, 92
    d = "M " + " L ".join(f"{cx + x:.1f} {cy + y:.1f}" for x, y in poly) + " Z"
    s.path(d, "s-ac f-acs", 2)
    d0 = "M " + " L ".join(f"{cx + x:.1f} {cy + y:.1f}" for x, y in SQ) + " Z"
    s.path(d0, "s-mu", 1.3)
    s.add(s.parts.pop().replace('/>', ' stroke-dasharray="4 3"/>'))
    s.circle(cx + poly[0][0], cy + poly[0][1], 4.5, "f-bd")
    s.text(cx, 180, name, "f-fg", 14, weight="600")
    s.text(cx, 202, sub, "f-mu", 12)
for i in range(1, 5):
    s.line(152 * i, 20, 152 * i, 205, "s-mu", 0.8, "2 4")
s.save(os.path.join(OUT, "25-transforms.svg"))

# ---------------------------------------------------------------- 쌍선형 보간 도식
s = Svg(560, 420, "쌍선형 보간: 네 화소 값으로 가운데 점의 값을 두 단계로 구함")
L, R, T, B = 150, 450, 50, 350
dx, dy = 0.6, 0.3
px, py = L + dx * (R - L), T + dy * (B - T)
v = {"tl": 0.30, "tr": 0.20, "bl": 0.26, "br": 0.10}
top = v["tl"] + dx * (v["tr"] - v["tl"])
bot = v["bl"] + dx * (v["br"] - v["bl"])
val = top + dy * (bot - top)
s.rect(L, T, R - L, B - T, "s-fg f-sf", 1.5)
s.line(L, py, R, py, "s-mu", 1.2, "5 4")
s.line(L, T, R, T, "s-ac", 2.5)
s.line(L, B, R, B, "s-ac", 2.5)
s.line(px, T, px, B, "s-bd", 2.5)
# 작은 직사각형 넓이 = 맞은편 모서리의 가중치
for (x0, x1, y0, y1, w) in [(L, px, T, py, dx * dy), (px, R, T, py, (1 - dx) * dy),
                            (L, px, py, B, dx * (1 - dy)), (px, R, py, B, (1 - dx) * (1 - dy))]:
    s.text((x0 + x1) / 2, (y0 + y1) / 2 + 5, f"{w:.2f}", "f-mu", 13)
for x, y in [(L, T), (R, T), (L, B), (R, B)]:
    s.circle(x, y, 6, "f-fg")
s.text(L - 12, T - 10, f"{v['tl']:.2f}", "f-fg", 14, "end", "600")
s.text(R + 12, T - 10, f"{v['tr']:.2f}", "f-fg", 14, "start", "600")
s.text(L - 12, B + 22, f"{v['bl']:.2f}", "f-fg", 14, "end", "600")
s.text(R + 12, B + 22, f"{v['br']:.2f}", "f-fg", 14, "start", "600")
s.circle(px, T, 5, "f-ac")
s.circle(px, B, 5, "f-ac")
s.text(px, T - 10, f"{top:.2f}", "f-ac", 14, weight="600")
s.text(px, B + 22, f"{bot:.3f}", "f-ac", 14, weight="600")
s.circle(px, py, 7, "f-bd")
s.text(px - 10, py + 22, f"P {val:.3f}", "f-bd", 14, "end", "600")
# 치수선
yd = 392
s.line(L, yd, R, yd, "s-mu", 1)
for x in (L, px, R):
    s.line(x, yd - 6, x, yd + 6, "s-mu", 1)
s.text((L + px) / 2, yd + 20, f"{dx:.1f}", "f-mu", 13)
s.text((px + R) / 2, yd + 20, f"{1 - dx:.1f}", "f-mu", 13)
xd = 105
s.line(xd, T, xd, B, "s-mu", 1)
for y in (T, py, B):
    s.line(xd - 6, y, xd + 6, y, "s-mu", 1)
s.text(xd - 10, (T + py) / 2 + 5, f"{dy:.1f}", "f-mu", 13, "end")
s.text(xd - 10, (py + B) / 2 + 5, f"{1 - dy:.1f}", "f-mu", 13, "end")
s.save(os.path.join(OUT, "25-bilinear.svg"))

# ---------------------------------------------------------------- 리샘플링 비교 (PNG, 글자 없음)
d = np.load(os.path.join(HERE, "..", "data", "part4_scene.npz"))
cls = d["cls"]
nir = d["refl"][3].astype(np.float32)
M = cv2.getRotationMatrix2D((149.5, 149.5), 15, 0.5)   # 15° 회전 + 0.5배(10 m → 20 m)
M[:, 2] += 74.5 - 149.5


def warp(img, interp, border):
    return cv2.warpAffine(img, M, (150, 150), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


c_near = warp(cls, cv2.INTER_NEAREST, 255)
c_bil = np.floor(warp(cls.astype(np.float32), cv2.INTER_LINEAR, np.nan) + 0.5)
c_bil = np.where(np.isnan(c_bil), 255, c_bil).astype(np.uint8)
n_bil = warp(nir, cv2.INTER_LINEAR, np.nan)
n_cub = warp(nir, cv2.INTER_CUBIC, np.nan)

r0, c0, S = 88, 30, 60                                   # 원본에서 자른 곳(행, 열, 크기)
cy, cx = r0 + S / 2 - 0.5, c0 + S / 2 - 0.5
oy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
ox = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
orr, occ = int(round(oy - 14.5)), int(round(ox - 14.5))

PAL = np.array([(20, 60, 120), (70, 150, 220), (30, 110, 50), (175, 205, 95), (205, 80, 70),
                (80, 80, 80), (215, 175, 125), (140, 120, 100), (255, 255, 255)], np.uint8)


def col(lbl):
    out = np.full(lbl.shape + (3,), 128, np.uint8)
    ok = lbl < 9
    out[ok] = PAL[lbl[ok]]
    return out


def gray(x):
    g = np.clip(np.nan_to_num(x, nan=0.21) / 0.42 * 255, 0, 255).astype(np.uint8)
    out = np.repeat(g[..., None], 3, -1)
    out[np.nan_to_num(x, nan=1) < 0] = (230, 40, 40)      # 음수 반사율은 빨강
    return out


def up(img, k):
    return np.kron(img, np.ones((k, k, 1), np.uint8)).astype(np.uint8)


so = np.s_[r0:r0 + S, c0:c0 + S]
oo = np.s_[orr:orr + 30, occ:occ + 30]
panels = [[up(col(cls[so]), 5), up(col(c_near[oo]), 10), up(col(c_bil[oo]), 10)],
          [up(gray(nir[so]), 5), up(gray(n_bil[oo]), 10), up(gray(n_cub[oo]), 10)]]
G = 12
canvas = np.full((2 * 300 + G, 3 * 300 + 2 * G, 3), 128, np.uint8)
for i, row in enumerate(panels):
    for j, p in enumerate(row):
        canvas[i * (300 + G):i * (300 + G) + 300, j * (300 + G):j * (300 + G) + 300] = p
Image.fromarray(canvas).convert("P", palette=Image.ADAPTIVE, colors=256).save(
    os.path.join(OUT, "25-resample.png"), optimize=True)
print("neg in crop", int((n_cub[oo] < 0).sum()), "false-looking classes in crop", np.unique(c_bil[oo]))
