"""27강 그림: 5부 타일 예시(PNG), 퍼셉트론·다층 퍼셉트론 도식(SVG), 활성화 함수 곡선(SVG)

자료: content/lessons/data/make_part5.py의 load() (5부 공통 예제, 가상 Sentinel-2 10 m 타일 32×32, 밴드 B2 B3 B4 B8)
"""
import os, sys
import numpy as np
from PIL import Image
from scipy.stats import norm
from scipy import ndimage as ndi

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from svglib import Svg
from make_part5 import load

OUT = os.path.join(HERE, "..", "fig")

# ---------------------------------------------------------------- 1. 타일 예시 (PNG, 2줄 × 6클래스)
# 위 줄: 한 클래스만 있는 타일, 아래 줄: 경계 타일(다른 클래스가 30~49% 섞임). 폴스컬러(B8, B4, B3 → R, G, B)
D = load()
x, y, m = D["x_train"], D["y_train"], D["m_train"]
pure, mixed, partner = [], [], []
for c in range(6):
    idx = np.where(y == c)[0]
    frac = np.array([(m[i] != c).mean() for i in idx])
    # 얇은 구름 조각(크고 매끈한 밝은 얼룩)이 든 타일을 피하려고, 후보 가운데 청색 밴드를 흐린 값의
    # (최댓값 − 중앙값)이 가장 작은 타일을 고름
    blob = np.array([ndi.gaussian_filter(x[i][0], 3).max() - np.median(x[i][0]) for i in idx])
    pc = np.where(frac == 0)[0][:40]
    mc = np.where((frac >= 0.30) & (frac < 0.49))[0][:40]
    pure.append(int(idx[pc[blob[pc].argmin()]]))
    j = int(idx[mc[blob[mc].argmin()]])
    mixed.append(j)
    other = np.bincount(m[j].ravel(), minlength=6)
    other[c] = 0
    partner.append(D["classes"][int(other.argmax())])
tiles = [x[i] for i in pure + mixed]
fc = np.stack([t[[3, 2, 1]] for t in tiles])            # (12, 3, 32, 32)
lo = np.percentile(fc, 1, axis=(0, 2, 3))              # 밴드마다 하위 1%
hi = np.percentile(fc, 99, axis=(0, 2, 3))              # 밴드마다 상위 1%에서 자름
SC, G, N = 5, 8, 32 * 5
canvas = np.zeros((2 * N + G, 6 * N + 5 * G, 4), np.uint8)   # 틈은 투명
for k, t in enumerate(fc):
    v = np.clip((t - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1) ** 0.8   # 감마 0.8로 어두운 물을 조금 밝힘
    rgb = np.round(v * 255).astype(np.uint8)
    rgb = rgb.transpose(1, 2, 0).repeat(SC, 0).repeat(SC, 1)
    r0, c0 = (k // 6) * (N + G), (k % 6) * (N + G)
    canvas[r0:r0 + N, c0:c0 + N, :3] = rgb
    canvas[r0:r0 + N, c0:c0 + N, 3] = 255
img = Image.fromarray(canvas, "RGBA").quantize(colors=256, method=Image.Quantize.FASTOCTREE)
img.save(os.path.join(OUT, "27-tiles.png"), optimize=True)
print("tiles pure", pure, "mixed", mixed, "partner", partner,
      "mixed frac", [round(float((m[j] != y[j]).mean()), 2) for j in mixed])

# ---------------------------------------------------------------- 2. 퍼셉트론과 다층 퍼셉트론 도식
s = Svg(760, 300, "왼쪽은 퍼셉트론: 밴드 값 네 개의 가중합에 편향을 더하고 계단 함수로 0 또는 1을 냄. "
                  "오른쪽은 다층 퍼셉트론: 입력 4개, 은닉층 두 개(각 32유닛, ReLU), 출력 6개(로짓)")
# 왼쪽: 퍼셉트론
s.text(170, 30, "퍼셉트론", "f-fg", 14, weight="600")
ins = [("B2", 80), ("B3", 125), ("B4", 170), ("B8", 215)]
SX, SY, SR = 175, 147, 24
for lab, yy in ins:
    s.line(62, yy, SX - SR * 0.95, SY + (yy - SY) * 0.25, "s-mu", 1.2)
    s.circle(50, yy, 12, "s-fg f-bg")
    s.text(30, yy + 4, lab, "f-mu", 12, anchor="end")
s.text(112, 72, "가중치 w", "f-mu", 12)
s.circle(SX, SY, SR, "s-fg f-acs")
s.text(SX, SY + 5, "Σ + b", "f-fg", 13, weight="600")
s.line(SX + SR, SY, 222, SY, "s-fg", 1.5)
s.rect(222, SY - 24, 50, 48, "s-fg f-sf", 1.2, rx=4)
s.path(f"M 230 {SY + 12} L 247 {SY + 12} L 247 {SY - 12} L 264 {SY - 12}", "s-ac", 2)
s.text(247, SY + 44, "계단 함수", "f-mu", 12)
s.line(272, SY, 296, SY, "s-fg", 1.5)
s.text(302, SY + 5, "0/1", "f-fg", 13, anchor="start", weight="600")
s.line(345, 20, 345, 285, "s-mu", 1, "3 4")
# 오른쪽: 다층 퍼셉트론 (4 → 32 → 32 → 6)
s.text(555, 30, "다층 퍼셉트론(MLP)", "f-fg", 14, weight="600")
LX = [400, 500, 600, 700]


def ys(n, top=62, bot=222):
    return [top + (bot - top) * i / (n - 1) for i in range(n)]


layers = [ys(4, 87, 197), ys(5), ys(5), ys(6, 67, 217)]
# 은닉층은 5칸 중 4번째를 '⋮'로 (32개 중 일부만 그림)
drawn = [layers[0], [v for i, v in enumerate(layers[1]) if i != 3], [v for i, v in enumerate(layers[2]) if i != 3], layers[3]]
for a in range(3):
    for y1 in drawn[a]:
        for y2 in drawn[a + 1]:
            s.line(LX[a] + 10, y1, LX[a + 1] - 10, y2, "s-mu", 0.5)
for a, col in enumerate(drawn):
    cls = "s-fg f-acs" if a in (1, 2) else "s-fg f-bg"
    for yy in col:
        s.circle(LX[a], yy, 10, cls)
for a in (1, 2):
    s.text(LX[a], layers[a][3] + 5, "⋮", "f-fg", 16, weight="600")
labels = [("입력 4", "밴드 평균"), ("은닉층 1", "32 · ReLU"), ("은닉층 2", "32 · ReLU"), ("출력 6", "로짓")]
for a, (l1, l2) in enumerate(labels):
    s.text(LX[a], 256, l1, "f-fg", 12, weight="600")
    s.text(LX[a], 274, l2, "f-mu", 12)
s.save(os.path.join(OUT, "27-mlp.svg"))

# ---------------------------------------------------------------- 3. 활성화 함수 곡선
s = Svg(680, 300, "활성화 함수 곡선. 왼쪽은 비교용 시그모이드와 tanh(양 끝에서 평평해짐), "
                  "오른쪽은 ReLU, Leaky ReLU(α=0.1로 과장), GELU")


def panel(X0, X1, YT, YB, zlo, zhi, ylo, yhi, yticks, curves, legend):
    sx = lambda v: X0 + (X1 - X0) * (v - zlo) / (zhi - zlo)
    sy = lambda v: YB - (YB - YT) * (v - ylo) / (yhi - ylo)
    s.rect(X0, YT, X1 - X0, YB - YT, "s-mu f-bg", 0.8)
    s.line(X0, sy(0), X1, sy(0), "s-mu", 0.8)
    s.line(sx(0), YT, sx(0), YB, "s-mu", 0.8)
    for v in yticks:
        s.line(X0 - 4, sy(v), X0, sy(v), "s-mu", 1)
        s.text(X0 - 7, sy(v) + 4, f"{v:g}".replace("-", "−"), "f-mu", 11, anchor="end")
    for v in range(int(zlo), int(zhi) + 1, 1 if zhi - zlo <= 6 else 2):
        s.line(sx(v), YB, sx(v), YB + 4, "s-mu", 1)
        s.text(sx(v), YB + 17, f"{v:g}".replace("-", "−"), "f-mu", 11)
    s.text((X0 + X1) / 2, YB + 36, "입력 z", "f-mu", 12)
    zs = np.linspace(zlo, zhi, 241)
    for f, cls, dash in curves:
        pts = [(sx(z), sy(f(z))) for z in zs]
        d = "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts)
        s.add(f'<path d="{d}" class="{cls}" stroke-width="2.4" fill="none"'
              + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
    for k, (lab, cls, dash) in enumerate(legend):
        ly = YT + 20 + 20 * k
        s.line(X0 + 12, ly - 4, X0 + 38, ly - 4, cls, 2.4, dash)
        s.text(X0 + 44, ly, lab, "f-fg", 12, anchor="start")


sig = lambda z: 1 / (1 + np.exp(-z))
panel(55, 315, 20, 245, -4, 4, -1.2, 1.2, [-1, 0, 1],
      [(sig, "s-ac", None), (np.tanh, "s-bd", "7 4")],
      [("시그모이드", "s-ac", None), ("tanh", "s-bd", "7 4")])
gelu = lambda z: z * norm.cdf(z)
panel(400, 660, 20, 245, -3, 3, -0.8, 3.0, [0, 1, 2, 3],
      [(lambda z: np.maximum(0, z), "s-ac", None), (lambda z: np.where(z > 0, z, 0.1 * z), "s-bd", "7 4"),
       (gelu, "s-ok", "2 3")],
      [("ReLU", "s-ac", None), ("Leaky ReLU", "s-bd", "7 4"), ("GELU", "s-ok", "2 3")])
s.save(os.path.join(OUT, "27-activation.svg"))
print("ok")
