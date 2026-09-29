"""9강 그림: 자기상관 있는 잔차와 없는 잔차, 더빈-왓슨 값의 범위, 행정동 회귀 잔차 지도

본문 수치도 이 스크립트의 자료로 계산함
- 저수지 수면적 예: 가상의 월별 수면적(km², 위성 수계 마스크 산출물)과 월 강수량(mm), 96개월
  수면적 = 2.3 − 0.0015×월 + 0.0022×강수 + AR(1) 오차(ρ = 0.7), 시드 403
- 행정동 예: 2부 공통 자료(data/part2_lst.csv)의 기본 모형 잔차
"""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")


# ---------------------------------------------------------------- 자료: 저수지 수면적 시계열
def reservoir(seed=403, rho=0.7, sig=0.06, c=-0.0015, b=0.0022):
    clim = np.array([25, 35, 55, 85, 100, 140, 300, 280, 150, 50, 45, 25.0])
    rng = np.random.default_rng(seed)
    n = 97
    rain = np.tile(clim, 9)[:n] * np.exp(rng.normal(-0.1, 0.45, n))
    u = np.zeros(n)
    e = rng.normal(0, sig, n)
    u[0] = e[0] / np.sqrt(1 - rho ** 2)
    for k in range(1, n):
        u[k] = rho * u[k - 1] + e[k]
    t = np.arange(n)
    area = 2.3 + c * t + b * rain + u
    # 첫 달은 시차 변수용으로만 쓰고 버림 → t = 1..96
    return dict(year=t[1:] / 12, rain=rain[1:], area=area[1:], rain_lag1=rain[:-1], area_lag1=area[:-1])


def ols_resid(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ b


def dw(e):
    return float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))


d = reservoir()
e1 = ols_resid(d["area"], [d["year"], d["rain"]])                                   # 정적 모형
e2 = ols_resid(d["area"], [d["year"], d["rain"], d["rain_lag1"], d["area_lag1"]])   # 시차 변수 모형
dw1, dw2 = dw(e1), dw(e2)

# ---------------------------------------------------------------- 그림 1: 잔차 시계열 두 개
s = Svg(680, 330, "월별 회귀 잔차 두 가지: 자기상관이 강한 잔차와 거의 없는 잔차")
X0, X1 = 60, 660
lim = 0.24
panels = [(e1, 30, f"정적 모형 잔차 (DW {dw1:.2f})", "s-bd", "f-bd"),
          (e2, 185, f"시차 변수 모형 잔차 (DW {dw2:.2f})", "s-ac", "f-ac")]
PH = 110
n = len(e1)
sx = lambda i: X0 + (X1 - X0) * i / (n - 1)
for e, top, title, scls, fcls in panels:
    mid = top + 12 + PH / 2
    sy = lambda v, mid=mid: mid - (PH / 2) * v / lim
    s.rect(X0, top + 12, X1 - X0, PH, "s-mu f-bg", 0.8)
    s.line(X0, mid, X1, mid, "s-mu", 1, "4 3")
    for v in (0.2, -0.2):
        s.text(X0 - 6, sy(v) + 4, f"{v:+.1f}", "f-mu", 11, anchor="end")
    s.text(X0 - 6, mid + 4, "0", "f-mu", 11, anchor="end")
    s.path("M " + " L ".join(f"{sx(i):.1f} {sy(v):.1f}" for i, v in enumerate(e)), scls, 1.3)
    for i, v in enumerate(e):
        s.circle(sx(i), sy(v), 2.2, fcls)
    s.text(X0, top + 4, title, fcls, 13, anchor="start", weight="600")
for yr in range(0, 9):
    x = sx(yr * 12 - 0.5) if yr else X0
    if yr:
        s.line(x, 297, x, 302, "s-mu", 1)
    s.text(sx(min(yr * 12 + 5.5, n - 1)), 318, f"{yr + 1}년차", "f-mu", 11) if yr < 8 else None
s.save(os.path.join(OUT, "09-resid-ts.svg"))

# ---------------------------------------------------------------- 그림 2: DW 값의 범위
s = Svg(680, 190, "더빈-왓슨 통계량의 범위: 0 근처는 양의 자기상관, 2 근처는 자기상관 없음, 4 근처는 음의 자기상관")
X0, X1, Y = 50, 630, 92
sx = lambda v: X0 + (X1 - X0) * v / 4
s.rect(sx(0), Y - 16, sx(1.5) - sx(0), 32, "s-mu f-bds", 1)
s.rect(sx(1.5), Y - 16, sx(2.5) - sx(1.5), 32, "s-mu f-sf", 1)
s.rect(sx(2.5), Y - 16, sx(4) - sx(2.5), 32, "s-mu f-acs", 1)
for v in range(5):
    s.line(sx(v), Y + 16, sx(v), Y + 22, "s-fg", 1.2)
    s.text(sx(v), Y + 38, str(v), "f-fg", 13)
s.text(sx(0.75), Y + 5, "양의 자기상관", "f-bd", 13, weight="600")
s.text(sx(2.0), Y + 5, "거의 없음", "f-fg", 13, weight="600")
s.text(sx(3.25), Y + 5, "음의 자기상관", "f-ac", 13, weight="600")
s.text(sx(0.75), Y + 62, "잔차가 이웃 달과 닮음 (r > 0)", "f-mu", 12)
s.text(sx(2.0), Y + 62, "r ≈ 0", "f-mu", 12)
s.text(sx(3.25), Y + 62, "부호가 번갈아 바뀜 (r < 0)", "f-mu", 12)
for v, lab, cls in ((dw1, f"정적 모형 {dw1:.2f}", "f-bd"), (dw2, f"시차 변수 모형 {dw2:.2f}", "f-ac")):
    s.path(f"M {sx(v):.1f} {Y - 20:.1f} L {sx(v) - 6:.1f} {Y - 30:.1f} L {sx(v) + 6:.1f} {Y - 30:.1f} Z", cls, 0)
    s.text(sx(v), Y - 38, lab, cls, 12, weight="600")
s.text(sx(2.0), 24, "DW ≈ 2(1 − r),  r = 이웃한 두 시점 잔차의 상관", "f-fg", 13)
s.save(os.path.join(OUT, "09-dw-range.svg"))

# ---------------------------------------------------------------- 그림 3: 행정동 잔차 지도
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "data", "part2_lst.csv"), encoding="utf-8")))
xk = np.array([float(r["x_km"]) for r in rows]); yk = np.array([float(r["y_km"]) for r in rows])
yv = np.array([float(r["lst_c"]) for r in rows])
cols = [np.array([float(r[c]) for r in rows]) for c in ("imperv_pct", "ndvi", "elev_m", "coast_km")]
res = ols_resid(yv, cols)
s = Svg(680, 470, "행정동 80개의 회귀 잔차 지도. 빨강은 실제가 예측보다 뜨거운 동, 파랑은 시원한 동, 원 크기는 잔차 크기")
MX0, MY0, W, H = 70, 40, 520, 390            # 40 km × 30 km
sx = lambda v: MX0 + W * v / 40
sy = lambda v: MY0 + H * (1 - v / 30)
s.rect(MX0, MY0, W, H, "s-mu f-bg", 1)
s.line(MX0, MY0, MX0, MY0 + H, "s-ac", 4)
s.text(MX0 - 10, MY0 + H / 2, "해안", "f-ac", 12, anchor="end")
for v in (0, 10, 20, 30, 40):
    s.text(sx(v), MY0 + H + 18, f"{v}", "f-mu", 11)
s.text(sx(20), MY0 + H + 34, "해안에서 떨어진 거리 (km)", "f-mu", 11)
for v in (0, 10, 20, 30):
    s.text(MX0 - 6, sy(v) + 4, f"{v}", "f-mu", 11, anchor="end") if v else None
rad = lambda v: 2 + 2.8 * abs(v)            # 원 반지름이 잔차 크기에 비례
order = np.argsort(-np.abs(res))
for i in order[::-1]:
    r = rad(res[i])
    s.circle(sx(xk[i]), sy(yk[i]), round(r, 1), "f-bd" if res[i] > 0 else "f-ac")
i43 = [k for k, r in enumerate(rows) if r["dong"] == "D43"][0]
s.text(sx(xk[i43]) - 22, sy(yk[i43]) + 33, "D43 (+6.7 ℃)", "f-bd", 12, anchor="end", weight="600")
# 범례
LX = 612
s.text(LX, 60, "잔차", "f-fg", 12, weight="600")
for k, (v, cls, lab) in enumerate(((3, "f-bd", "+3 ℃"), (1, "f-bd", "+1 ℃"), (-1, "f-ac", "−1 ℃"), (-3, "f-ac", "−3 ℃"))):
    cy = 88 + 32 * k
    s.circle(LX - 4, cy, round(rad(v), 1), cls)
    s.text(LX + 14, cy + 4, lab, "f-fg", 11, anchor="start")
s.line(sx(18) - 6, sy(15), sx(18) + 6, sy(15), "s-fg", 2)
s.line(sx(18), sy(15) - 6, sx(18), sy(15) + 6, "s-fg", 2)
s.text(sx(18) + 10, sy(15) + 4, "도심", "f-fg", 12, anchor="start", weight="600")
s.text(MX0 - 6, MY0 - 10, "남북 (km)", "f-mu", 11, anchor="start")
s.save(os.path.join(OUT, "09-resid-map.svg"))
print(f"DW 정적 {dw1:.3f}, 시차 {dw2:.3f}; 잔차 범위 {res.min():.2f}~{res.max():.2f}")
