"""7강 그림: 잔차 대 적합값 플롯의 네 가지 전형, 공통 자료(part2_lst.csv) 기본 모형의 잔차 플롯과 Q-Q 플롯"""
import os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")

# ---------------------------------------------------------------- 1. 네 가지 전형 (도식)
rng = np.random.default_rng(12)
n = 60
fx = np.sort(rng.uniform(0, 1, n))
z = rng.normal(0, 1, n)
ok = 0.28 * z
funnel = (0.06 + 0.55 * fx) * z
curve = 1.6 * (fx - 0.5) ** 2 * 4 - 0.55 + 0.12 * z          # U자
curve -= curve.mean()
spike = 0.25 * rng.normal(0, 1, n)
spike[int(n * 0.7)] = 1.25                                  # 튀는 점 하나
panels = [("정상", "고르게 흩어진 띠", ok, None),
          ("깔때기", "이분산", funnel, None),
          ("곡선", "비선형", curve, None),
          ("튀는 점", "이상치", spike, int(n * 0.7))]

s = Svg(680, 230, "잔차 대 적합값 플롯의 네 가지 전형: 정상, 깔때기(이분산), 곡선(비선형), 튀는 점(이상치)")
PW, PH, TOP, GAP, LEFT = 145, 140, 36, 22, 20
for k, (title, sub, e, hi) in enumerate(panels):
    x0 = LEFT + k * (PW + GAP)
    yc = TOP + PH / 2
    my = lambda v: yc - (PH / 2 - 8) * v / 1.35
    mx = lambda v: x0 + 8 + (PW - 16) * v
    s.rect(x0, TOP, PW, PH, "s-mu f-bg", 1)
    s.line(x0, yc, x0 + PW, yc, "s-mu", 1.2, "4 3")
    for i, (a, b) in enumerate(zip(fx, e)):
        s.circle(mx(a), my(b), 3.4 if i != hi else 4.8, "f-bd" if i == hi else "f-ac")
    if title == "깔때기":
        s.path(f"M {mx(0):.1f} {my(0.15):.1f} L {mx(1):.1f} {my(1.25):.1f}", "s-bd", 1.4)
        s.path(f"M {mx(0):.1f} {my(-0.15):.1f} L {mx(1):.1f} {my(-1.25):.1f}", "s-bd", 1.4)
    if title == "곡선":
        xs = np.linspace(0, 1, 60)
        cv = 1.6 * (xs - 0.5) ** 2 * 4 - 0.55 - (1.6 * (fx - 0.5) ** 2 * 4 - 0.55).mean()
        s.path("M " + " L ".join(f"{mx(a):.1f} {my(b):.1f}" for a, b in zip(xs, cv)), "s-bd", 1.6)
    s.text(x0 + PW / 2, 24, title, "f-fg", 13, weight="600")
    s.text(x0 + PW / 2, TOP + PH + 20, sub, "f-mu", 12)
s.text(LEFT - 6, TOP + PH / 2 + 4, "0", "f-mu", 11, anchor="end")
s.text(340, TOP + PH + 44, "가로축: 적합값 · 세로축: 잔차", "f-mu", 12)
s.save(os.path.join(OUT, "07-patterns.svg"))

# ---------------------------------------------------------------- 2. 공통 자료의 잔차 플롯 + Q-Q 플롯
df = pd.read_csv(os.path.join(HERE, "..", "data", "part2_lst.csv"))
m = smf.ols("lst_c ~ imperv_pct + ndvi + elev_m + coast_km", df).fit()
e = m.resid.values
fv = m.fittedvalues.values
ri = m.get_influence().resid_studentized_internal
k43 = int(np.where(df.dong == "D43")[0][0])

s = Svg(680, 290, "공통 자료 기본 모형의 잔차 대 적합값 플롯(오른쪽으로 갈수록 퍼짐)과 정규 Q-Q 플롯(D43 한 점이 크게 벗어남)")
# 왼쪽: 잔차 대 적합값
X0, X1, Y0, Y1 = 60, 320, 240, 30
FLO, FHI, ELO, EHI = 22, 36, -4, 8
sx = lambda v: X0 + (X1 - X0) * (v - FLO) / (FHI - FLO)
sy = lambda v: Y0 - (Y0 - Y1) * (v - ELO) / (EHI - ELO)
s.line(X0, Y0, X1, Y0, "s-mu", 1.2); s.line(X0, Y0, X0, Y1, "s-mu", 1.2)
s.line(X0, sy(0), X1, sy(0), "s-mu", 1.2, "4 3")
for v in (22, 26, 30, 34):
    s.line(sx(v), Y0, sx(v), Y0 + 5, "s-mu", 1.2); s.text(sx(v), Y0 + 19, f"{v}", "f-mu", 11)
for v in (-4, 0, 4, 8):
    s.line(X0 - 5, sy(v), X0, sy(v), "s-mu", 1.2); s.text(X0 - 9, sy(v) + 4, f"{v}".replace("-", "−"), "f-mu", 11, anchor="end")
for i in range(len(e)):
    if i != k43:
        s.circle(sx(fv[i]), sy(e[i]), 3.4, "f-ac")
s.circle(sx(fv[k43]), sy(e[k43]), 4.8, "f-bd")
s.text(sx(fv[k43]) - 10, sy(e[k43]) + 4, "D43", "f-bd", 12, anchor="end", weight="600")
s.text((X0 + X1) / 2, Y0 + 40, "적합값(℃)", "f-mu", 12)
s.text(X0 - 40, 20, "잔차(℃)", "f-mu", 12, anchor="start")
assert sx(fv.min()) > X0 and sx(fv.max()) < X1 and sy(e.min()) < Y0 and sy(e.max()) > Y1

# 오른쪽: Q-Q (내적 스튜던트화 잔차 vs 정규 이론 분위수)
(osm, osr), _ = stats.probplot(ri, dist="norm")
order = np.argsort(ri)
X0, X1 = 410, 660
QLO, QHI, RLO, RHI = -3, 3, -3, 5
qx = lambda v: X0 + (X1 - X0) * (v - QLO) / (QHI - QLO)
qy = lambda v: Y0 - (Y0 - Y1) * (v - RLO) / (RHI - RLO)
s.line(X0, Y0, X1, Y0, "s-mu", 1.2); s.line(X0, Y0, X0, Y1, "s-mu", 1.2)
s.line(qx(-3), qy(-3), qx(3), qy(3), "s-fg", 1.3, "5 4")
for v in (-2, 0, 2):
    s.line(qx(v), Y0, qx(v), Y0 + 5, "s-mu", 1.2); s.text(qx(v), Y0 + 19, f"{v}".replace("-", "−"), "f-mu", 11)
for v in (-2, 0, 2, 4):
    s.line(X0 - 5, qy(v), X0, qy(v), "s-mu", 1.2); s.text(X0 - 9, qy(v) + 4, f"{v}".replace("-", "−"), "f-mu", 11, anchor="end")
for a, b in zip(osm[:-1], osr[:-1]):
    s.circle(qx(a), qy(b), 3.4, "f-ac")
assert order[-1] == k43
s.circle(qx(osm[-1]), qy(osr[-1]), 4.8, "f-bd")
s.text(qx(osm[-1]) - 10, qy(osr[-1]) + 4, "D43", "f-bd", 12, anchor="end", weight="600")
s.text((X0 + X1) / 2, Y0 + 40, "정규분포 이론 분위수", "f-mu", 12)
s.text(X0 - 40, 20, "내적 스튜던트화 잔차", "f-mu", 12, anchor="start")
assert qx(osm.min()) > X0 and qy(osr.max()) > Y1 and qy(osr.min()) < Y0
s.save(os.path.join(OUT, "07-lst-resid.svg"))
print("ok", round(fv.min(), 2), round(fv.max(), 2), round(e.min(), 2), round(e.max(), 2), round(ri.min(), 2), round(ri.max(), 2), round(osm.min(), 2))
