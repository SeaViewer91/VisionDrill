"""22강 그림: 3×3 합성곱 계산 도식(SVG), 광학 밴드 필터 비교(PNG), SAR 스페클 필터 비교(PNG)

PNG는 4부 공통 자료(data/part4_scene.npz)로 만듦. 글자 없이 회색조 팔레트 PNG, 패널 사이 틈은 투명.
"""
import os, sys
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
D = np.load(os.path.join(HERE, "..", "data", "part4_scene.npz"))

# ---------------------------------------------------------------- 1. 3×3 합성곱 계산 도식
I = np.array([[49, 51, 54, 58, 60],
              [50, 52, 55, 61, 62],
              [53, 59, 250, 64, 65],
              [55, 57, 60, 63, 66],
              [56, 58, 61, 64, 67]], float)
MEAN = np.array([[I[r:r + 3, c:c + 3].mean() for c in range(3)] for r in range(3)])
MED = np.array([[np.median(I[r:r + 3, c:c + 3]) for c in range(3)] for r in range(3)])

s = Svg(740, 262, "5×5 입력에 3×3 평균 커널을 밀며 곱해 더한 출력과, 같은 창의 미디언 필터 출력")
C = 38


def box(x, y, w, h):
    """채움 없는 강조 테두리"""
    s.path(f"M {x:.1f} {y:.1f} h {w:.1f} v {h:.1f} h {-w:.1f} Z", "s-ac", 2.8)


def grid(x0, y0, vals, fmt, hl=None, hl_cls="s-ac f-acs", bold=None):
    n, m = vals.shape
    for r in range(n):
        for c in range(m):
            inside = hl and hl[0] <= r <= hl[1] and hl[2] <= c <= hl[3]
            s.rect(x0 + c * C, y0 + r * C, C, C, "s-mu f-acs" if inside else "s-mu f-bg", 1)
            cls = "f-bd" if bold and (r, c) == bold else "f-fg"
            s.text(x0 + c * C + C / 2, y0 + r * C + C / 2 + 5, fmt(vals[r, c]), cls, 13,
                   weight="600" if bold and (r, c) == bold else None)
    if hl:
        r0, r1, c0, c1 = hl
        box(x0 + c0 * C, y0 + r0 * C, (c1 - c0 + 1) * C, (r1 - r0 + 1) * C)


# 입력
xi, yi = 20, 52
s.text(xi + 2.5 * C, yi - 14, "입력 5×5", "f-fg", 13, weight="600")
grid(xi, yi, I, lambda v: f"{v:.0f}", hl=(1, 3, 1, 3), bold=(2, 2))
cy = yi + 2.5 * C                                   # 입력 세로 가운데
# 커널
xk, yk = 268, cy - 1.5 * C
s.text(xi + 5 * C + (xk - xi - 5 * C) / 2, cy + 6, "∗", "f-fg", 22)
s.text(xk + 1.5 * C, yk - 14, "평균 커널", "f-fg", 13, weight="600")
grid(xk, yk, np.full((3, 3), 1 / 9), lambda v: "1/9")
# 평균 출력
xo = 434
s.text(xk + 3 * C + (xo - xk - 3 * C) / 2, cy + 6, "=", "f-fg", 22)
s.text(xo + 1.5 * C, yk - 14, "평균 필터 출력", "f-fg", 13, weight="600")
grid(xo, yk, MEAN, lambda v: f"{v:.1f}")
box(xo + C, yk + C, C, C)
s.text(xo + 1.5 * C, yk + 3 * C + 22, "튀는 값이 9칸에 퍼짐", "f-bd", 12)
# 구분선 + 미디언 출력
xs = 584
s.line(xs, yk - 30, xs, yk + 3 * C + 30, "s-mu", 1.2, "4 4")
xm = 606
s.text(xm + 1.5 * C, yk - 14, "미디언 필터 출력", "f-fg", 13, weight="600")
grid(xm, yk, MED, lambda v: f"{v:.0f}")
box(xm + C, yk + C, C, C)
s.text(xm + 1.5 * C, yk + 3 * C + 22, "튀는 값이 사라짐", "f-ok", 12)
s.save(os.path.join(OUT, "22-conv.svg"))

# ---------------------------------------------------------------- PNG 공통: 회색조 팔레트 + 투명 틈
GAP, SC = 8, 2                   # 패널 사이 틈(화소), 확대 배율(최근접)


def to_u8(a, lo, hi):
    return np.clip((a - lo) / (hi - lo) * 254, 0, 254).round().astype(np.uint8)


def mosaic(rows, path):
    ph, pw = rows[0][0].shape[0] * SC, rows[0][0].shape[1] * SC
    nr, nc = len(rows), len(rows[0])
    canvas = np.full((nr * ph + (nr - 1) * GAP, nc * pw + (nc - 1) * GAP), 255, np.uint8)
    for i, row in enumerate(rows):
        for j, p in enumerate(row):
            big = np.kron(p, np.ones((SC, SC), np.uint8))
            canvas[i * (ph + GAP):i * (ph + GAP) + ph, j * (pw + GAP):j * (pw + GAP) + pw] = big
    im = Image.fromarray(canvas, "P")
    pal = []
    for v in range(255):
        g = round(v * 255 / 254)
        pal += [g, g, g]
    pal += [0, 0, 0]
    im.putpalette(pal)
    im.info["transparency"] = 255
    im.save(path, optimize=True, transparency=255)
    return canvas.shape


# ---------------------------------------------------------------- 2. 광학(B8 근적외) 필터 비교
b8 = D["refl"][3].astype(np.float64)
rng = np.random.default_rng(22)
sp = b8.copy()
u = rng.random(b8.shape)
sp[u < 0.025] = 0.0
sp[(u >= 0.025) & (u < 0.05)] = 0.5                 # 소금·후추 잡음 5%
unsharp = b8 + 1.0 * (b8 - ndi.gaussian_filter(b8, 1))
crop = (slice(66, 210), slice(62, 206))
disp = lambda a: to_u8(a[crop], 0.0, 0.42)
rows = [[disp(b8), disp(ndi.gaussian_filter(b8, 1)), disp(ndi.gaussian_filter(b8, 2)), disp(unsharp)],
        [disp(sp), disp(ndi.uniform_filter(sp, 3)), disp(ndi.median_filter(sp, 3)), disp(ndi.median_filter(sp, 7))]]
print("optical", mosaic(rows, os.path.join(OUT, "22-optical.png")))

# ---------------------------------------------------------------- 3. SAR 스페클 필터 비교
z = D["sar"].astype(np.float64)
x = D["sar_clean"].astype(np.float64)


def lee(img, w, looks=1):
    """리 필터(단순형): 국소 평균 m, 국소 변동계수 Ci, 스페클 변동계수 Cu = 1/sqrt(looks)"""
    m = ndi.uniform_filter(img, w)
    v = np.maximum(ndi.uniform_filter(img * img, w) - m * m, 1e-12)
    ci2 = v / np.maximum(m * m, 1e-12)
    wgt = np.clip(1 - (1.0 / looks) / ci2, 0, 1)
    return m + wgt * (img - m)


crop = (slice(20, 164), slice(0, 144))
dB = lambda a: to_u8(10 * np.log10(np.maximum(a[crop], 1e-6)), -28, 2)
rows = [[dB(z), dB(ndi.uniform_filter(z, 5)), dB(ndi.median_filter(z, 5)), dB(lee(z, 5))],
        [dB(x), dB(ndi.uniform_filter(z, 9)), dB(ndi.median_filter(z, 9)), dB(lee(z, 9))]]
print("sar", mosaic(rows, os.path.join(OUT, "22-sar.png")))
