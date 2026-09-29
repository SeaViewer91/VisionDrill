"""8강 그림: 이상치·레버리지·영향점 비교 도식, 공통 자료의 레버리지 대 잔차 산점도"""
import os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")


def fit(x, y):
    """단순회귀: 기울기, 절편, 레버리지, Cook's D"""
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H)
    e = y - X @ b
    p = 2
    s2 = e @ e / (len(x) - p)
    r = e / np.sqrt(s2 * (1 - h))
    D = r ** 2 / p * h / (1 - h)
    return b, h, D


# ---------------------------------------------------------------- 1. 세 종류의 튀는 점
rng = np.random.default_rng(8)
xb = np.linspace(1, 6, 12)
yb = 1.0 + 0.5 * xb + rng.normal(0, 0.25, xb.size)
extra = [("이상치", 3.5, 5.6), ("레버리지 큼", 11.0, None), ("영향점", 11.0, 2.6)]
b0, *_ = fit(xb, yb)
extra[1] = ("레버리지 큼", 11.0, b0[0] + b0[1] * 11.0 + 0.05)

s = Svg(680, 250, "튀는 점 세 종류: y 방향 이상치, 레버리지가 큰 점, 영향점이 회귀선에 주는 영향")
PW, GAP, L, TOP, BOT = 196, 30, 22, 44, 200
XMAX, YMAX = 12.0, 7.5
for k, (name, xe, ye) in enumerate(extra):
    ox = L + k * (PW + GAP)
    sx = lambda v, ox=ox: ox + PW * v / XMAX
    sy = lambda v: BOT - (BOT - TOP) * v / YMAX
    s.line(ox, BOT, ox + PW, BOT, "s-mu", 1.2)
    s.line(ox, BOT, ox, TOP - 6, "s-mu", 1.2)
    x = np.append(xb, xe); y = np.append(yb, ye)
    b1, h, D = fit(x, y)
    # 원래 선(점선)과 점을 넣은 선(실선)
    s.line(sx(0), sy(b0[0]), sx(XMAX), sy(b0[0] + b0[1] * XMAX), "s-mu", 1.6, "5 4")
    s.line(sx(0), sy(b1[0]), sx(XMAX), sy(b1[0] + b1[1] * XMAX), "s-bd", 2)
    for a, c in zip(xb, yb):
        s.circle(sx(a), sy(c), 3.6, "f-mu")
    s.circle(sx(xe), sy(ye), 6, "f-bd")
    s.text(ox + PW / 2, 24, name, "f-fg", 14, weight="600")
    s.text(ox + PW / 2, 222, f"h = {h[-1]:.2f}, D = {D[-1]:.2f}", "f-mu", 12)
    s.text(ox + PW / 2, 240, f"기울기 {b0[1]:.2f} → {b1[1]:.2f}", "f-bd", 12)
    print(name, "h", round(h[-1], 3), "D", round(D[-1], 3), "slope", round(b0[1], 3), "->", round(b1[1], 3), "mean h", round(2 / len(x), 3))
s.save(os.path.join(OUT, "08-three-points.svg"))

# ---------------------------------------------------------------- 2. 레버리지 대 스튜던트화 잔차 (공통 자료)
d = pd.read_csv(os.path.join(HERE, "..", "data", "part2_lst.csv"))
m = smf.ols("lst_c ~ imperv_pct + ndvi + elev_m + coast_km", d).fit()
inf = m.get_influence()
h = inf.hat_matrix_diag
r = inf.resid_studentized_internal
D = inf.cooks_distance[0]
n, p = m.model.exog.shape

s = Svg(680, 330, "레버리지 대 스튜던트화 잔차 산점도와 Cook의 거리 등고선. 공단 동 D43이 오른쪽 위에 홀로 떨어져 있음")
X0, X1, Y0, Y1 = 70, 640, 290, 24          # 그림 영역 (Y0 = 아래)
HMAX, RMIN, RMAX = 0.25, -3.0, 5.0
sx = lambda v: X0 + (X1 - X0) * v / HMAX
sy = lambda v: Y0 - (Y0 - Y1) * (v - RMIN) / (RMAX - RMIN)
# 축과 눈금
s.line(X0, Y0, X1, Y0, "s-mu", 1.2)
s.line(X0, Y0, X0, Y1, "s-mu", 1.2)
for v in (0, 0.05, 0.10, 0.15, 0.20, 0.25):
    s.line(sx(v), Y0, sx(v), Y0 + 5, "s-mu", 1.2)
    s.text(sx(v), Y0 + 19, f"{v:.2f}", "f-mu", 12)
for v in (-2, 0, 2, 4):
    s.line(X0 - 5, sy(v), X0, sy(v), "s-mu", 1.2)
    s.text(X0 - 9, sy(v) + 4, f"{v}", "f-mu", 12, anchor="end")
s.line(X0, sy(0), X1, sy(0), "s-mu", 1, "2 4")
for v in (-2, 2):
    s.line(X0, sy(v), X1, sy(v), "s-mu", 1, "2 4")
s.text((X0 + X1) / 2, Y0 + 38, "레버리지 h", "f-fg", 13)
s.text(X0 - 42, (Y0 + Y1) / 2 - 9, "내적", "f-fg", 13)
s.text(X0 - 42, (Y0 + Y1) / 2 + 9, "잔차", "f-fg", 13)
# 레버리지 관례 기준 2p/n
cut = 2 * p / n
s.line(sx(cut), Y0, sx(cut), Y1, "s-ac", 1.4, "6 4")
s.text(sx(cut) + 6, Y1 + 12, f"2p/n = {cut:.3f}", "f-ac", 12, anchor="start")
# Cook's D 등고선: r = ±sqrt(D p (1-h)/h)
for Dv, lab in ((0.5, "D = 0.5"), (1.0, "D = 1")):
    for sign in (1, -1):
        pts = []
        for hv in np.linspace(0.005, HMAX, 400):
            rv = sign * np.sqrt(Dv * p * (1 - hv) / hv)
            if RMIN <= rv <= RMAX:
                pts.append((sx(hv), sy(rv)))
        if len(pts) > 1:
            s.path("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts), "s-bd", 1.3 if Dv == 1 else 1.0)
    hl = HMAX - 0.022                       # 글자 왼쪽 끝의 h: 곡선이 이 지점에서 가장 높음
    rv = np.sqrt(Dv * p * (1 - hl) / hl)
    s.text(X1 - 4, sy(rv) - 5, lab, "f-bd", 12, anchor="end")
# 점: 크기 = Cook's D
order = np.argsort(D)
for k in order:
    rad = 2.6 + 9 * np.sqrt(D[k])
    cls = "f-bd" if d.dong[k] == "D43" else "f-mu"
    s.circle(sx(h[k]), sy(r[k]), round(rad, 1), cls)
k43 = int(np.where(d.dong == "D43")[0][0])
s.text(sx(h[k43]), sy(r[k43]) + 32, "D43", "f-bd", 13, weight="600")
k01 = int(np.where(d.dong == "D01")[0][0])
s.text(sx(h[k01]), sy(r[k01]) + 20, "D01", "f-mu", 12)
s.save(os.path.join(OUT, "08-leverage.svg"))
print("D43 h", round(h[k43], 3), "r", round(r[k43], 2), "D", round(D[k43], 3), "| D01 h", round(h[k01], 3), "r", round(r[k01], 2), "D", round(D[k01], 4))
