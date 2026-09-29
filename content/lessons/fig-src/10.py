"""10강 그림: 인구밀도 로그 변환 전후 분포, 상호작용이 있을 때와 없을 때의 기울기"""
import csv, math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
from scipy import stats

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
d = pd.read_csv(os.path.join(HERE, "..", "data", "part2_lst.csv"))

# ---------------------------------------------------------------- 로그 변환 전후 히스토그램
pop = d.pop_density.values
lp = np.log(pop)
s = Svg(680, 250, "인구밀도의 로그 변환 전후 분포: 오른쪽 꼬리가 긴 분포가 거의 대칭이 됨")


def panel(x0, vals, edges, title, mean, med, fmt, ticks):
    """x0: 패널 왼쪽 끝. 너비 280, 막대 영역 y 60~200"""
    W, Y0, H = 280, 200, 120
    cnt, _ = np.histogram(vals, edges)
    top = max(cnt)
    lo, hi = edges[0], edges[-1]
    sx = lambda v: x0 + W * (v - lo) / (hi - lo)
    for c, a, b in zip(cnt, edges[:-1], edges[1:]):
        h = H * c / top
        if c:
            s.rect(sx(a), Y0 - h, sx(b) - sx(a), h, "f-acs s-mu", 1)
    s.line(x0, Y0, x0 + W, Y0, "s-mu", 1.2)
    for t, lab in ticks:
        s.line(sx(t), Y0, sx(t), Y0 + 5, "s-mu", 1.2)
        s.text(sx(t), Y0 + 19, lab, "f-mu", 11)
    s.text(x0 + W / 2, 30, title, "f-fg", 13, weight="600")
    # 평균·중앙값 선
    s.line(sx(med), Y0 - H - 8, sx(med), Y0, "s-ac", 2)
    s.line(sx(mean), Y0 - H - 8, sx(mean), Y0, "s-bd", 2, "5 3")
    return sx


ticks_raw = [(0, "0"), (20000, "2만"), (40000, "4만"), (60000, "6만")]
panel(40, pop, np.arange(0, 60001, 5000), f"인구밀도 (명/km²) · 왜도 {stats.skew(pop):.2f}",
      pop.mean(), np.median(pop), None, ticks_raw)
ticks_log = [(7, "7"), (8, "8"), (9, "9"), (10, "10"), (11, "11")]
panel(370, lp, np.arange(6.5, 11.01, 0.5), f"ln(인구밀도) · 왜도 {stats.skew(lp):.2f}",
      lp.mean(), np.median(lp), None, ticks_log)
# 범례 (왼쪽 패널 오른쪽 빈 공간)
s.line(210, 92, 236, 92, "s-ac", 2.5); s.text(242, 96, "중앙값", "f-ac", 12, anchor="start")
s.line(210, 114, 236, 114, "s-bd", 2.5, "5 3"); s.text(242, 118, "평균", "f-bd", 12, anchor="start")
s.save(os.path.join(OUT, "10-log.svg"))
print("log panel: mean", pop.mean(), "median", np.median(pop), "log mean", lp.mean(), "log median", np.median(lp))

# ---------------------------------------------------------------- 상호작용 두 직선
base = smf.ols("lst_c ~ imperv_pct + ndvi + elev_m + coast_km", d).fit()
inter = smf.ols("lst_c ~ imperv_pct * coast_km + ndvi + elev_m", d).fit()
fix = dict(ndvi=d.ndvi.mean(), elev_m=d.elev_m.mean())

s = Svg(680, 300, "상호작용항이 없으면 두 직선이 평행하고, 있으면 기울기가 달라짐")
XL, XH, YL, YH = 10, 95, 20, 38


def frame(x0, title):
    W, Y0, H = 270, 220, 160
    sx = lambda v: x0 + W * (v - XL) / (XH - XL)
    sy = lambda v: Y0 - H * (v - YL) / (YH - YL)
    s.line(x0, Y0, x0 + W, Y0, "s-mu", 1.2)
    s.line(x0, Y0, x0, Y0 - H, "s-mu", 1.2)
    for t in (20, 40, 60, 80):
        s.line(sx(t), Y0, sx(t), Y0 + 5, "s-mu", 1.2)
        s.text(sx(t), Y0 + 18, str(t), "f-mu", 11)
    for t in (20, 25, 30, 35):
        s.line(x0 - 5, sy(t), x0, sy(t), "s-mu", 1.2)
        s.text(x0 - 9, sy(t) + 4, str(t), "f-mu", 11, anchor="end")
    s.text(x0 + W / 2, Y0 + 40, "불투수면 비율 (%)", "f-mu", 12)
    s.text(x0 + W / 2, 30, title, "f-fg", 13, weight="600")
    s.text(x0 - 30, Y0 - H - 10, "LST (℃)", "f-mu", 11, anchor="start")
    return sx, sy


def lines(model, sx, sy):
    out = []
    for c, cls in ((5, "s-ac"), (35, "s-bd")):
        g = pd.DataFrame({"imperv_pct": [XL, XH], "coast_km": [c, c], **fix})
        a, b = model.predict(g)
        s.line(sx(XL), sy(a), sx(XH), sy(b), cls, 2.5)
        out.append((c, a, b, (b - a) / (XH - XL)))
    return out


sx, sy = frame(60, "상호작용항 없음")
r1 = lines(base, sx, sy)
s.text(sx(12), sy(36.2), f"두 선 기울기 모두 {r1[0][3]:.3f}", "f-fg", 12, anchor="start")
s.text(sx(12), sy(34.2), "해안 거리는 높이만 바꿈", "f-mu", 12, anchor="start")

sx, sy = frame(390, "상호작용항 있음")
r2 = lines(inter, sx, sy)
s.text(sx(12), sy(36.2), f"기울기 {r2[0][3]:.3f} → {r2[1][3]:.3f}", "f-fg", 12, anchor="start")
s.text(sx(12), sy(34.2), "해안에서 멀수록 가팔라짐", "f-mu", 12, anchor="start")
# 범례 (아래쪽 가운데)
s.line(250, 288, 274, 288, "s-ac", 2.5); s.text(280, 292, "해안 5 km", "f-ac", 12, anchor="start")
s.line(360, 288, 384, 288, "s-bd", 2.5); s.text(390, 292, "해안 35 km", "f-bd", 12, anchor="start")
s.save(os.path.join(OUT, "10-interaction.svg"))
print("base", r1, "\ninter", r2)
