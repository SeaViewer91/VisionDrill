"""21강 그림: 밴드 합성 네 가지(PNG), 스트레칭 전후 히스토그램(SVG), 감마 곡선(SVG)

자료: content/lessons/data/part4_scene.npz (4부 공통 예제, 가상 연안 장면 300×300 화소)
밴드 순서 B2 B3 B4 B8 B11 B12. DN은 make_part4.py의 규칙(반사율 0~0.5 → 12비트 0~4095)으로 만듦
"""
import os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
d = np.load(os.path.join(HERE, "..", "data", "part4_scene.npz"))
refl = d["refl"].astype(np.float32)
dn = np.round(refl / 0.5 * 4095).clip(0, 4095).astype(np.uint16)
IDX = {"B2": 0, "B3": 1, "B4": 2, "B8": 3, "B11": 4, "B12": 5}


def pct_stretch(x, lo=2, hi=98):
    """백분위 lo~hi%를 0~255로 선형으로 늘리고 바깥은 잘라 냄 (밴드마다 따로)"""
    x = x.astype(np.float64)
    a, b = np.percentile(x, [lo, hi])
    return np.round(np.clip((x - a) / (b - a) * 255, 0, 255)).astype(np.uint8)


def composite(bands, stretch=True):
    if stretch:
        return np.dstack([pct_stretch(dn[IDX[b]]) for b in bands])
    return np.dstack([(dn[IDX[b]] // 16).astype(np.uint8) for b in bands])   # 12비트 → 8비트 단순 축소


# ---------------------------------------------------------------- 1. 합성 네 가지 (PNG, 2×2)
panels = [
    composite(("B4", "B3", "B2"), stretch=False),   # 위 왼쪽: 트루컬러, 스트레칭 전
    composite(("B4", "B3", "B2")),                  # 위 오른쪽: 트루컬러, 2~98%
    composite(("B8", "B4", "B3")),                  # 아래 왼쪽: 표준 폴스컬러
    composite(("B12", "B8", "B4")),                 # 아래 오른쪽: SWIR 합성
]
G, N = 10, 300
canvas = np.zeros((2 * N + G, 2 * N + G, 4), np.uint8)   # 틈은 투명(다크 모드 배경이 비치게)
for k, p in enumerate(panels):
    r0, c0 = (k // 2) * (N + G), (k % 2) * (N + G)
    canvas[r0:r0 + N, c0:c0 + N, :3] = p
    canvas[r0:r0 + N, c0:c0 + N, 3] = 255
img = Image.fromarray(canvas, "RGBA").quantize(colors=256, method=Image.Quantize.FASTOCTREE)
png = os.path.join(OUT, "21-composites.png")
img.save(png, optimize=True)

# ---------------------------------------------------------------- 2. 스트레칭 전후 히스토그램 (B4)
b4 = dn[IDX["B4"]].astype(np.float64)
before = np.floor(b4 / 16)
lo, hi = np.percentile(b4, [2, 98])
after = np.round(np.clip((b4 - lo) / (hi - lo) * 255, 0, 255))
YMAX = 16000
s = Svg(680, 360, "적색 밴드 B4의 화면 값 히스토그램. 위는 12비트 DN을 16으로 나눈 것으로 0~120에 몰려 있고, 아래는 2~98% 백분위 스트레칭 후로 0~255 전체에 퍼짐")
X0, X1 = 70, 640
sx = lambda v: X0 + (X1 - X0) * v / 256
rows = [("스트레칭 전: DN ÷ 16", before, 30, 140), ("2~98% 백분위 스트레칭 후", after, 195, 305)]
for title, v, TOP, BOT in rows:
    h, edges = np.histogram(v, bins=64, range=(0, 256))
    sy = lambda c: BOT - (BOT - TOP) * min(c, YMAX) / YMAX
    for j, c in enumerate(h):
        if c == 0:
            continue
        s.rect(sx(edges[j]), sy(c), sx(edges[j + 1]) - sx(edges[j]), BOT - sy(c), "s-ac f-acs", 0.8)
        if c > YMAX:   # 잘린 막대: 물결 표시 대신 짧은 가로 틈과 실제 값
            s.line(sx(edges[j]) - 2, TOP + 8, sx(edges[j + 1]) + 2, TOP + 4, "s-bd", 1.5)
            s.text(sx(edges[j + 1]) + 6, TOP + 10, f"{c:,}", "f-bd", 11, anchor="start")
    s.line(X0, BOT, X1, BOT, "s-mu", 1.2)
    s.line(X0, TOP, X0, BOT, "s-mu", 1.2)
    for t in (0, 8000, 16000):
        s.line(X0 - 4, sy(t), X0, sy(t), "s-mu", 1.2)
        s.text(X0 - 8, sy(t) + 4, f"{t:,}", "f-mu", 11, anchor="end")
    for t in (0, 64, 128, 192, 255):
        s.line(sx(t), BOT, sx(t), BOT + 4, "s-mu", 1.2)
        s.text(sx(t), BOT + 17, f"{t}", "f-mu", 11)
    s.text(X1, TOP - 8, title, "f-fg", 13, anchor="end", weight="600")
s.text((X0 + X1) / 2, 350, "화면 값 (8비트, 0~255)", "f-mu", 12)
s.save(os.path.join(OUT, "21-stretch-hist.svg"))

# ---------------------------------------------------------------- 3. 감마 곡선
s = Svg(680, 330, "감마 보정 곡선. 가로축은 입력 값, 세로축은 출력 값(둘 다 0~1). 감마 1은 직선, 2.2는 어두운 쪽을 밝히고, 0.5는 어둡게 함")
X0, X1, YT, YB = 190, 470, 20, 290
mx = lambda v: X0 + (X1 - X0) * v
my = lambda v: YB - (YB - YT) * v
s.rect(X0, YT, X1 - X0, YB - YT, "s-mu f-bg", 1)
for t in (0.25, 0.5, 0.75):
    s.line(mx(t), YT, mx(t), YB, "s-mu", 0.5, "2 4")
    s.line(X0, my(t), X1, my(t), "s-mu", 0.5, "2 4")
for t in (0, 0.5, 1):
    s.text(mx(t), YB + 17, f"{t:g}", "f-mu", 11)
    s.text(X0 - 8, my(t) + 4, f"{t:g}", "f-mu", 11, anchor="end")
xs = np.linspace(0, 1, 101)
curves = [(2.2, "s-bd", "γ = 2.2 (밝게)", "f-bd"), (1.0, "s-mu", "γ = 1", "f-mu"), (0.5, "s-ac", "γ = 0.5 (어둡게)", "f-ac")]
for g, cls, lab, tcls in curves:
    ys = xs ** (1 / g)
    s.path("M " + " L ".join(f"{mx(a):.1f} {my(b):.1f}" for a, b in zip(xs, ys)), cls, 2.2)
# 예시 점: 입력 0.2 → γ 2.2 출력 0.48
x0, y0 = 0.2, 0.2 ** (1 / 2.2)
s.line(mx(x0), YB, mx(x0), my(y0), "s-bd", 1, "3 3")
s.line(X0, my(y0), mx(x0), my(y0), "s-bd", 1, "3 3")
s.circle(mx(x0), my(y0), 4, "f-bd")
s.text(mx(x0), YB + 17, "0.2", "f-bd", 11)
s.text(mx(x0) - 6, my(y0) - 8, f"{y0:.2f}", "f-bd", 12, anchor="end", weight="600")
# 곡선 이름: 그림 오른쪽 바깥, 곡선이 x=1 근처에서 만나는 것을 피해 각 곡선 중간 높이 옆에 둠
s.text(X1 + 14, my(0.62 ** (1 / 2.2)) - 40, curves[0][2], "f-bd", 12, anchor="start", weight="600")
s.line(mx(0.62), my(0.62 ** (1 / 2.2)), X1 + 10, my(0.62 ** (1 / 2.2)) - 44, "s-bd", 0.8)
s.text(X1 + 14, my(0.5) + 4, curves[1][2], "f-mu", 12, anchor="start", weight="600")
s.line(mx(0.75), my(0.75), X1 + 10, my(0.5), "s-mu", 0.8)
s.text(X1 + 14, my(0.62 ** 2) + 30, curves[2][2], "f-ac", 12, anchor="start", weight="600")
s.line(mx(0.62), my(0.62 ** 2), X1 + 10, my(0.62 ** 2) + 26, "s-ac", 0.8)
s.text((X0 + X1) / 2, YB + 36, "입력 (0~1로 맞춘 값)", "f-mu", 12)
s.text(X0 - 40, (YT + YB) / 2 - 6, "출력", "f-mu", 12, anchor="end")
s.save(os.path.join(OUT, "21-gamma.svg"))

print("png bytes", os.path.getsize(png), "B4 p2/p98", lo, hi)
