"""26강 그림: 핀홀 투영과 GSD, 스테레오 시차와 깊이 오차, DEM·DSM·nDSM 비교(PNG)

자료: content/lessons/data/part4_scene.npz (4부 공통 예제)의 dem(맨땅, DTM에 해당)·dsm
"""
import os, sys
import numpy as np
from PIL import Image
from matplotlib import colormaps

sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")


def dim_v(s, x, y1, y2, cls="s-mu"):
    s.line(x, y1, x, y2, cls, 1.2)
    s.line(x - 5, y1, x + 5, y1, cls, 1.2)
    s.line(x - 5, y2, x + 5, y2, cls, 1.2)


def dim_h(s, x1, x2, y, cls="s-mu"):
    s.line(x1, y, x2, y, cls, 1.2)
    s.line(x1, y - 5, x1, y + 5, cls, 1.2)
    s.line(x2, y - 5, x2, y + 5, cls, 1.2)


# ---------------------------------------------------------------- 1. 핀홀 투영과 GSD
s = Svg(680, 315, "핀홀 투영: 투영 중심, 영상면, 초점거리 f, 거리 Z, 지상 점 P와 영상 좌표 x, 화소 크기 p와 GSD")
OX, OY, IY, GY = 220, 60, 110, 270          # 투영 중심, 영상면, 지면 높이(그림 좌표)
f, Z = IY - OY, GY - OY                      # 50, 210
ratio = Z / f
PX = 190                                     # 지상 점 P의 가로 거리 X
xi = PX / ratio                              # 영상 좌표 x
s.line(OX, OY, OX, GY, "s-mu", 1.2, "4 4")                      # 광축
s.line(40, GY, 520, GY, "s-fg", 2)                               # 지면
s.line(110, IY, 330, IY, "s-ac", 2.5)                            # 영상면(가상, 투영 중심 앞)
s.line(OX, OY, OX + PX, GY, "s-bd", 1.6)                         # P로 가는 광선
s.circle(OX + PX, GY, 5, "f-bd")
s.circle(OX + xi, IY, 4, "f-bd")
s.circle(OX, OY, 5, "f-fg")
# 화소 하나와 그 지상 투영(GSD)
p0, p1 = -16, -8
for q in (p0, p1):
    s.line(OX, OY, OX + q * ratio, GY, "s-ac", 1.1, "3 3")
s.line(OX + p0, IY, OX + p1, IY, "s-fg", 5)
s.line(OX + p0 * ratio, GY, OX + p1 * ratio, GY, "s-fg", 5)
# 치수
dim_v(s, 95, OY, IY)
s.text(84, (OY + IY) / 2 + 5, "f", "f-fg", 14, "end", "600")
dim_v(s, 60, OY, GY)
s.text(49, (OY + GY) / 2 + 5, "Z", "f-fg", 14, "end", "600")
s.line(55, OY, OX - 8, OY, "s-mu", 1, "2 3")
dim_h(s, OX, OX + xi, IY + 16)
s.text(OX + xi / 2, IY + 33, "x", "f-bd", 14, weight="600")
dim_h(s, OX, OX + PX, GY + 16)
s.text(OX + PX / 2, GY + 36, "X", "f-bd", 14, weight="600")
s.text(OX + PX + 12, GY - 8, "P", "f-bd", 14, "start", "600")
s.text(OX + (p0 + p1) / 2 - 22, IY - 8, "p", "f-fg", 13, weight="600")
s.text(OX + (p0 + p1) / 2 * ratio, GY + 36, "GSD", "f-fg", 13, weight="600")
s.text(OX + 12, OY - 8, "투영 중심", "f-fg", 13, "start")
s.text(335, IY + 5, "영상면", "f-ac", 13, "start")
s.text(530, 120, "x = f · X / Z", "f-fg", 15, "start", "600")
s.text(530, 160, "GSD = Z · p / f", "f-fg", 15, "start", "600")
s.text(530, 196, "닮은 삼각형", "f-mu", 12, "start")
s.save(os.path.join(OUT, "26-pinhole.svg"))

# ---------------------------------------------------------------- 2. 스테레오 시차와 깊이 오차
s = Svg(680, 320, "정류된 스테레오: 기선 B, 시차 d = xL - xR, 거리가 멀수록 깊이 오차가 제곱으로 커짐")
LX, RX, CY, IY, PY = 110, 250, 72, 112, 262
PXs = 200
fS, ZS = IY - CY, PY - CY
for cx in (LX, RX):
    s.line(cx - 55, IY, cx + 55, IY, "s-ac", 2.5)
    s.line(cx, CY, cx, IY + 14, "s-mu", 1.1, "3 3")
    s.line(cx, CY, PXs, PY, "s-bd", 1.5)
    s.circle(cx, CY, 5, "f-fg")
xl = (PXs - LX) * fS / ZS
xr = (PXs - RX) * fS / ZS
s.circle(LX + xl, IY, 4, "f-bd")
s.circle(RX + xr, IY, 4, "f-bd")
s.circle(PXs, PY, 5, "f-bd")
s.text(PXs, PY + 22, "P", "f-bd", 14, weight="600")
dim_h(s, LX, RX, 44)
s.text((LX + RX) / 2, 34, "B", "f-fg", 14, weight="600")
s.text(LX - 12, CY - 8, "왼쪽", "f-fg", 12, "end")
s.text(RX + 12, CY - 8, "오른쪽", "f-fg", 12, "start")
dim_v(s, 30, CY, PY)
s.text(42, (CY + PY) / 2 + 5, "Z", "f-fg", 14, "start", "600")
s.text(LX + xl + 4, IY + 32, "xL", "f-bd", 13, "start", "600")
s.text(RX + xr - 4, IY + 32, "xR", "f-bd", 13, "end", "600")
s.text(175, 300, "d = xL − xR", "f-fg", 14, weight="600")
# 오른쪽: 깊이 오차 곡선 (f·B = 500 화소·m, 시차 오차 0.5 화소)
AX0, AX1, AY0, AY1 = 385, 650, 262, 70
zmax, emax = 40.0, 1.6
sx = lambda z: AX0 + (AX1 - AX0) * z / zmax
sy = lambda e: AY0 - (AY0 - AY1) * e / emax
s.line(AX0, AY0, AX1, AY0, "s-mu", 1.2)
s.line(AX0, AY0, AX0, AY1 - 6, "s-mu", 1.2)
for z in (0, 10, 20, 30, 40):
    s.line(sx(z), AY0, sx(z), AY0 + 5, "s-mu", 1.2)
    s.text(sx(z), AY0 + 20, f"{z}", "f-mu", 12)
for e in (0.4, 0.8, 1.2, 1.6):
    s.line(AX0 - 5, sy(e), AX0, sy(e), "s-mu", 1.2)
    s.text(AX0 - 9, sy(e) + 4, f"{e:.1f}", "f-mu", 12, "end")
err = lambda z: z * z / 500 * 0.5
zs = np.linspace(0, 40, 81)
s.path("M " + " L ".join(f"{sx(z):.1f} {sy(err(z)):.1f}" for z in zs), "s-bd", 2.2)
for z in (10, 20, 40):
    s.circle(sx(z), sy(err(z)), 4.5, "f-bd")
s.text(sx(10) + 2, sy(err(10)) - 12, "0.1 m", "f-fg", 12, "middle")
s.text(sx(20) - 2, sy(err(20)) - 12, "0.4 m", "f-fg", 12, "end")
s.text(sx(40) - 10, sy(err(40)) + 4, "1.6 m", "f-fg", 12, "end")
s.text(AX0 + 6, AY1 - 12, "깊이 오차 (m)", "f-mu", 12, "start")
s.text((AX0 + AX1) / 2, AY0 + 40, "거리 Z (m)", "f-mu", 12)
s.save(os.path.join(OUT, "26-stereo.svg"))

# ---------------------------------------------------------------- 3. DEM · DSM · nDSM (PNG)
d = np.load(os.path.join(HERE, "..", "data", "part4_scene.npz"))
dem = d["dem"].astype(np.float64)
dsm = d["dsm"].astype(np.float64)
ndsm = dsm - dem


def shade(z, az=315, alt=45, cell=10.0):
    gy, gx = np.gradient(z, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, e = np.radians(az), np.radians(alt)
    hs = np.sin(e) * np.cos(slope) + np.cos(e) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(hs, 0, 1)


def elev_rgb(z):
    c = colormaps["gist_earth"](np.clip(z / 130.0, 0, 1) * 0.85 + 0.1)[..., :3]
    return c * (0.55 + 0.45 * shade(z))[..., None]


panels = [elev_rgb(dem), elev_rgb(dsm), colormaps["magma"](np.clip(ndsm / 40.0, 0, 1))[..., :3]]
S, GAP = 360, 20
canvas = np.zeros((S, 3 * S + 2 * GAP, 3), np.uint8)
canvas[:] = (255, 0, 255)                                   # 틈(투명 처리)
for i, p in enumerate(panels):
    im = Image.fromarray((p * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
    canvas[:, i * (S + GAP): i * (S + GAP) + S] = np.asarray(im)
img = Image.fromarray(canvas).quantize(256, method=Image.Quantize.MEDIANCUT)
pal = np.array(img.getpalette()[:768]).reshape(-1, 3)
key = int(np.argmin(((pal - [255, 0, 255]) ** 2).sum(1)))
img.save(os.path.join(OUT, "26-dsm-dtm.png"), optimize=True, transparency=key)

# 본문 수치
cls = d["cls"]
u = cls == 4
b = u & (ndsm > 2)
print("건물 화소", int(b.sum()), "시가지", int(u.sum()), f"비율 {b.sum() / u.sum():.3f}",
      f"평균 {ndsm[b].mean():.1f} 중앙 {np.median(ndsm[b]):.1f} 최대 {ndsm.max():.1f}")
print(f"산림 평균 {ndsm[cls == 2].mean():.2f}", "2 m 초과 화소", int((ndsm > 2).sum()))
print(f"DEM 최대 {dem.max():.1f} DSM 최대 {dsm.max():.1f}")
