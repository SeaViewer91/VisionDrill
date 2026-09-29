"""11강 그림: 릿지·라쏘 제약 영역, 릿지·라쏘 계수 경로, 교차검증 오차와 1-SE 규칙

경로와 교차검증은 2부 공통 자료(data/part2_lst.csv)의 기본 모형
lst_c ~ imperv_pct + ndvi + elev_m + coast_km 을 표준화해 scikit-learn으로 계산함
"""
import math, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import Lasso, Ridge, lasso_path
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
d = pd.read_csv(os.path.join(HERE, "..", "data", "part2_lst.csv"))
V = ["imperv_pct", "ndvi", "elev_m", "coast_km"]
NAMES = ["불투수면", "NDVI", "고도", "해안 거리"]
STY = [("s-ac", None, "f-ac"), ("s-ok", None, "f-ok"), ("s-bd", None, "f-bd"), ("s-mu", "6 4", "f-mu")]
X = d[V].values
y = d.lst_c.values
Z = (X - X.mean(0)) / X.std(0)


def dpath(s, d, cls, w, dash=None):
    s.path(d, cls, w)
    if dash:
        s.parts[-1] = s.parts[-1].replace("/>", f' stroke-dasharray="{dash}"/>')


def vtext(s, x, y, txt):
    s.text(x, y, txt, "f-mu", 11)
    s.parts[-1] = s.parts[-1].replace("<text ", f'<text transform="rotate(-90 {x:.1f} {y:.1f})" ')


def poly(pts):
    return "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts)


# ---------------------------------------------------------------- 1. 제약 영역 (원 vs 마름모)
bh = np.array([0.9, 2.0])                 # 최소제곱 추정값
A = np.array([[1.0, 0.55], [0.55, 1.0]])  # 잔차제곱합의 곡률 (X^T X 꼴)
t = 1.0                                    # 계수 예산
f = lambda b: (b - bh) @ A @ (b - bh)
mu = brentq(lambda m: np.linalg.norm(np.linalg.solve(A + m * np.eye(2), A @ bh)) - t, 0, 1e4)
b_ridge = np.linalg.solve(A + mu * np.eye(2), A @ bh)
cand = [np.array([s * (1 - abs(u)), u]) for u in np.linspace(-1, 1, 4001) for s in (1, -1)]
b_lasso = min(cand, key=f)
lev = {"ridge": f(b_ridge), "lasso": f(b_lasso)}
ev, Q = np.linalg.eigh(A)


def ellipse(c):
    th = np.linspace(0, 2 * math.pi, 181)
    u = np.stack([np.cos(th) * math.sqrt(c / ev[0]), np.sin(th) * math.sqrt(c / ev[1])])
    return (Q @ u).T + bh


s = Svg(680, 340, "릿지의 원형 제약 영역과 라쏘의 마름모 제약 영역에 잔차제곱합 등고선이 닿는 점")
PW, TOP, U = 300, 12, 54           # 패널 폭, 위 여백, 단위당 픽셀
for k, (kind, title) in enumerate([("ridge", "릿지: 원"), ("lasso", "라쏘: 마름모")]):
    ox = 30 + k * (PW + 40) + 1.4 * U       # β1 = 0 위치
    oy = TOP + 4.1 * U                      # β2 = 0 위치
    mx = lambda v: ox + U * v
    my = lambda v: oy - U * v
    s.line(mx(-1.4), my(0), mx(3.0), my(0), "s-mu", 1.2)
    s.line(mx(0), my(-1.2), mx(0), my(4.1), "s-mu", 1.2)
    s.text(mx(3.0) - 2, my(0) + 18, "β₁", "f-mu", 13, anchor="end")
    s.text(mx(0) - 8, my(4.1) + 12, "β₂", "f-mu", 13, anchor="end")
    if kind == "ridge":
        th = np.linspace(0, 2 * math.pi, 121)
        s.path(poly([(mx(math.cos(a)), my(math.sin(a))) for a in th]) + " Z", "s-ac f-acs", 1.8)
        bs = b_ridge
    else:
        s.path(poly([(mx(1), my(0)), (mx(0), my(1)), (mx(-1), my(0)), (mx(0), my(-1))]) + " Z", "s-ac f-acs", 1.8)
        bs = b_lasso
    c = lev[kind]
    for cc, w, dash in ((c * 0.25, 1.1, "4 3"), (c * 0.6, 1.1, "4 3"), (c, 1.8, None)):
        dpath(s, poly([(mx(a), my(b)) for a, b in ellipse(cc)]) + " Z", "s-bd", w, dash)
    s.circle(mx(bh[0]), my(bh[1]), 5, "f-fg")
    s.text(mx(bh[0]) + 9, my(bh[1]) - 6, "최소제곱", "f-fg", 12, anchor="start")
    s.circle(mx(bs[0]), my(bs[1]), 6, "f-ac")
    lab = f"({bs[0]:.2f}, {bs[1]:.2f})"
    s.text(mx(0) - 8, my(1.08), lab, "f-ac", 12, anchor="end", weight="600")
    s.text(30 + k * (PW + 40) + PW / 2, 332, title, "f-fg", 13, weight="600")
s.save(os.path.join(OUT, "11-geometry.svg"))
print("ridge", b_ridge.round(3), "lasso", b_lasso.round(3))

# ---------------------------------------------------------------- 2. 계수 경로 (릿지 / 라쏘)
r_al = np.logspace(-1, 3.5, 91)
r_co = np.array([Ridge(alpha=a).fit(Z, y).coef_ for a in r_al]).T
l_al = np.logspace(-3, 0.6, 145)
_, l_co, _ = lasso_path(Z, y - y.mean(), alphas=l_al[::-1])
l_co = l_co[:, ::-1]

s = Svg(680, 300, "λ를 키울 때 표준화 계수가 0으로 줄어드는 모습: 릿지와 라쏘")
PW, PH, TOP, LEFT, GAP = 270, 190, 58, 62, 58
YLO, YHI = -1.2, 2.8
# 범례
lx = 70
for (cls, dash, tcls), nm in zip(STY, NAMES):
    s.line(lx, 18, lx + 26, 18, cls, 2.5, dash)
    s.text(lx + 32, 23, nm, tcls, 12, anchor="start", weight="600")
    lx += 32 + 14 * len(nm) + 40
for k, (title, al, co, ticks) in enumerate([
        ("릿지", r_al, r_co, [0.1, 1, 10, 100, 1000]),
        ("라쏘", l_al, l_co, [0.001, 0.01, 0.1, 1])]):
    x0 = LEFT + k * (PW + GAP)
    y0 = TOP + PH
    la, lb = math.log10(al[0]), math.log10(al[-1])
    mx = lambda a: x0 + PW * (math.log10(a) - la) / (lb - la)
    my = lambda v: y0 - PH * (v - YLO) / (YHI - YLO)
    s.line(x0, y0, x0 + PW, y0, "s-mu", 1.2)
    s.line(x0, TOP, x0, y0, "s-mu", 1.2)
    s.line(x0, my(0), x0 + PW, my(0), "s-mu", 1, "2 3")
    for v in (-1, 0, 1, 2):
        s.line(x0 - 4, my(v), x0, my(v), "s-mu", 1.2)
        s.text(x0 - 8, my(v) + 4, f"{v}", "f-mu", 11, anchor="end")
    for a in ticks:
        s.line(mx(a), y0, mx(a), y0 + 4, "s-mu", 1.2)
        s.text(mx(a), y0 + 17, f"{a:g}", "f-mu", 11)
    for j, (cls, dash, _) in enumerate(STY):
        dpath(s, poly([(mx(a), my(v)) for a, v in zip(al, co[j])]), cls, 2.2, dash)
    s.text(x0 + PW / 2, TOP - 10, title, "f-fg", 13, weight="600")
    s.text(x0 + PW / 2, y0 + 36, "λ (로그 눈금, 오른쪽일수록 강함)", "f-mu", 11)
vtext(s, 20, TOP + PH / 2, "표준화 계수")
s.save(os.path.join(OUT, "11-path.svg"))

# ---------------------------------------------------------------- 3. 교차검증 오차와 1-SE 규칙
grid = np.round(np.logspace(-3, 0.5, 36), 5)
kf = KFold(10, shuffle=True, random_state=0)
res = []
for a in grid:
    fm = []
    for tr, te in kf.split(X):
        p = make_pipeline(StandardScaler(), Lasso(alpha=a, max_iter=200000)).fit(X[tr], y[tr])
        fm.append(np.mean((p.predict(X[te]) - y[te]) ** 2))
    fm = np.array(fm)
    res.append((a, fm.mean(), fm.std(ddof=1) / math.sqrt(len(fm))))
res = np.array(res)
i = res[:, 1].argmin()
thr = res[i, 1] + res[i, 2]
ok = np.where(res[:, 1] <= thr)[0]
j = ok[res[ok, 0].argmax()]
nz = lambda a: int((np.abs(Lasso(alpha=a, max_iter=200000).fit(Z, y).coef_) > 1e-10).sum())
print("cv min", res[i].round(4), "thr", round(thr, 4), "1se", res[j].round(4), "nz", nz(res[i, 0]), nz(res[j, 0]))

sel = res[res[:, 0] <= 1.3]
s = Svg(680, 290, "라쏘의 10겹 교차검증 오차와 1 표준오차 규칙으로 고른 λ")
x0, y0, PW, PH, TOP = 70, 240, 580, 200, 40
la, lb = -3, math.log10(1.3)
YLO, YHI = 2.0, 6.5
mx = lambda a: x0 + PW * (math.log10(a) - la) / (lb - la)
my = lambda v: y0 - PH * (v - YLO) / (YHI - YLO)
s.line(x0, y0, x0 + PW, y0, "s-mu", 1.2)
s.line(x0, TOP, x0, y0, "s-mu", 1.2)
for v in (2, 3, 4, 5, 6):
    s.line(x0 - 4, my(v), x0, my(v), "s-mu", 1.2)
    s.text(x0 - 8, my(v) + 4, f"{v}", "f-mu", 11, anchor="end")
for a in (0.001, 0.01, 0.1, 1):
    s.line(mx(a), y0, mx(a), y0 + 4, "s-mu", 1.2)
    s.text(mx(a), y0 + 17, f"{a:g}", "f-mu", 11)
s.text(x0 + PW / 2, y0 + 38, "λ (scikit-learn alpha, 로그 눈금)", "f-mu", 11)
vtext(s, 18, TOP + PH / 2, "검증 MSE")
s.line(x0, my(thr), x0 + PW, my(thr), "s-bd", 1.2, "5 4")
s.text(x0 + 6, my(thr) - 15, f"최소 + 1 SE = {thr:.2f}", "f-bd", 11, anchor="start")
for a, m, se in sel:
    s.line(mx(a), my(m - se), mx(a), my(m + se), "s-mu", 1)
s.path(poly([(mx(a), my(m)) for a, m, _ in sel]), "s-ac", 2)
for a, m, _ in sel:
    s.circle(mx(a), my(m), 2.8, "f-ac")
for idx, lab, cls, tcls in ((i, "최소", "s-fg", "f-fg"), (j, "1-SE", "s-bd", "f-bd")):
    a = res[idx, 0]
    s.line(mx(a), TOP, mx(a), y0, cls, 1.5, "3 3")
    s.text(mx(a), TOP - 20, f"{lab} λ = {a:.2f}", tcls, 12, weight="600")
    s.text(mx(a), TOP - 5, f"변수 {nz(a)}개", tcls, 11)
s.save(os.path.join(OUT, "11-cv.svg"))
print("ok")
