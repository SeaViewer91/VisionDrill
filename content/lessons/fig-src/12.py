"""12강 그림: 선형회귀 직선과 로지스틱 곡선(폭염 취약 동), 2차원 특징 공간의 직선 결정 경계
자료: content/lessons/data/part2_lst.csv (2부 공통 예제)"""
import os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
d = pd.read_csv(os.path.join(HERE, "..", "data", "part2_lst.csv"))
sig = lambda z: 1 / (1 + np.exp(-z))

# ---------------------------------------------------------------- 1. 직선 vs 시그모이드
ols = smf.ols("heat_warn ~ imperv_pct", d).fit()
lg = smf.logit("heat_warn ~ imperv_pct", d).fit(disp=0)
a0, a1 = ols.params.iloc[0], ols.params.iloc[1]
b0, b1 = lg.params.iloc[0], lg.params.iloc[1]
x50 = -b0 / b1

s = Svg(680, 300, "폭염 취약 여부(0/1)를 불투수면 비율로 예측한 선형회귀 직선과 로지스틱 곡선")
X0, X1, YT, YB = 70, 600, 30, 250          # 그림 영역
ylo, yhi = -0.4, 1.1
sx = lambda v: X0 + (X1 - X0) * v / 100
sy = lambda p: YB - (YB - YT) * (p - ylo) / (yhi - ylo)
# 축과 눈금
s.line(X0, YB, X1, YB, "s-mu", 1.2)
s.line(X0, YT, X0, YB, "s-mu", 1.2)
for v in (0, 0.5, 1):
    s.line(X0, sy(v), X1, sy(v), "s-mu", 0.8, "2 4")
    s.text(X0 - 8, sy(v) + 4, f"{v:g}", "f-mu", 12, anchor="end")
s.text(X0 - 8, sy(-0.3) + 4, "−0.3", "f-mu", 12, anchor="end")
s.line(X0 - 4, sy(-0.3), X0, sy(-0.3), "s-mu", 1.2)
for v in (0, 20, 40, 60, 80, 100):
    s.line(sx(v), YB, sx(v), YB + 4, "s-mu", 1.2)
    s.text(sx(v), YB + 18, f"{v}", "f-mu", 12)
s.text((X0 + X1) / 2, YB + 38, "불투수면 비율(%)", "f-mu", 12)
# 관측값 (0 또는 1)
for xv, yv in zip(d.imperv_pct, d.heat_warn):
    s.circle(sx(xv), sy(yv), 3.5, "f-bd" if yv else "f-mu")
# 선형회귀 직선
xs0 = (ylo - a0) / a1                          # 그림 아래 끝(−0.4)에서 시작
s.line(sx(xs0), sy(ylo), sx(100), sy(a0 + a1 * 100), "s-mu", 2, "6 4")
# 로지스틱 곡선
xs = np.linspace(0, 100, 201)
s.path("M " + " L ".join(f"{sx(v):.1f} {sy(sig(b0 + b1 * v)):.1f}" for v in xs), "s-ac", 2.5)
# 결정 경계 (p = 0.5)
s.line(sx(x50), YT, sx(x50), YB, "s-bd", 1.5, "5 4")
s.text(sx(x50) - 6, sy(0.78), f"p = 0.5 → {x50:.0f}%", "f-bd", 12, anchor="end", weight="600")
# 범례
s.line(90, 52, 120, 52, "s-ac", 2.5)
s.text(128, 56, "로지스틱 회귀 (0~1 안에서 휨)", "f-ac", 12, anchor="start", weight="600")
s.line(90, 74, 120, 74, "s-mu", 2, "6 4")
s.text(128, 78, "선형회귀 (0 아래로 내려감)", "f-mu", 12, anchor="start", weight="600")
s.save(os.path.join(OUT, "12-sigmoid.svg"))

# ---------------------------------------------------------------- 2. 2차원 결정 경계
g = smf.logit("heat_warn ~ imperv_pct + elev_m", d).fit(disp=0)
c0, c1, c2 = g.params.iloc[0], g.params.iloc[1], g.params.iloc[2]

s = Svg(680, 330, "불투수면 비율과 고도 두 특징 공간에서 로지스틱 회귀의 직선 결정 경계 두 개(p=0.5, p=0.2)")
X0, X1, YT, YB = 80, 560, 25, 275
xlo, xhi, elo, ehi = 10, 100, 0, 220
sx = lambda v: X0 + (X1 - X0) * (v - xlo) / (xhi - xlo)
sy = lambda e: YB - (YB - YT) * (e - elo) / (ehi - elo)
s.line(X0, YB, X1, YB, "s-mu", 1.2)
s.line(X0, YT, X0, YB, "s-mu", 1.2)
for v in (20, 40, 60, 80, 100):
    s.line(sx(v), YB, sx(v), YB + 4, "s-mu", 1.2)
    s.text(sx(v), YB + 18, f"{v}", "f-mu", 12)
for e in (0, 50, 100, 150, 200):
    s.line(X0 - 4, sy(e), X0, sy(e), "s-mu", 1.2)
    s.text(X0 - 8, sy(e) + 4, f"{e}", "f-mu", 12, anchor="end")
s.text((X0 + X1) / 2, YB + 40, "불투수면 비율(%)", "f-mu", 12)
s.text(X0 - 8, YT - 8, "고도(m)", "f-mu", 12, anchor="end")
for xv, ev, yv in zip(d.imperv_pct, d.elev_m, d.heat_warn):
    if yv:
        s.circle(sx(xv), sy(ev), 4, "f-bd")
    else:
        s.circle(sx(xv), sy(ev), 4, "s-mu f-bg")
# 경계선: c0 + c1*x + c2*e = logit(t)  →  x = (logit(t) - c0 - c2*e) / c1
for t, dash in ((0.5, None), (0.2, "6 4")):
    L = np.log(t / (1 - t))
    xa = (L - c0 - c2 * elo) / c1
    xb = (L - c0 - c2 * ehi) / c1
    s.line(sx(xa), sy(elo), sx(xb), sy(ehi), "s-ac", 2.2, dash)
    s.text(sx(xb) + (6 if t == 0.5 else -6), YT + 4, f"p = {t}", "f-ac", 12,
           anchor="start" if t == 0.5 else "end", weight="600")
# 범례 (그림 오른쪽 바깥)
s.circle(582, 120, 4, "f-bd")
s.text(592, 124, "취약 동", "f-fg", 12, anchor="start")
s.circle(582, 144, 4, "s-mu f-bg")
s.text(592, 148, "그 외", "f-fg", 12, anchor="start")
s.save(os.path.join(OUT, "12-boundary.svg"))
print("ok", round(x50, 2), [round(v, 5) for v in (c0, c1, c2)])
