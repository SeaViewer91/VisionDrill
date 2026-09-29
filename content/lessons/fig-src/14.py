"""14강 그림: 다항식 차수별 적합 곡선, 편향²·분산·잡음 모의실험, 결정트리 학습곡선

모의실험: 해안에서 내륙으로 20 km 단면선의 가상 LST 곡선 f(x)에 잡음(σ = 1℃)을 더해
1 km 간격 20개 지점의 표본을 2,000벌 만들고, 차수 0~12 다항식을 최소제곱으로 맞춤.
편향²·분산은 단면선 전체(401점)에서 평균냄. 시험 MSE는 같은 401점에서 새 잡음으로 직접 잼.

학습곡선: 3부 공통 자료(data/part3_landcover.csv), 폴리곤 단위 5겹(GroupKFold).
각 겹의 학습 폴리곤 가운데 일부를 무작위로 20번 뽑아 결정트리를 맞춤(전체일 때는 1번).
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")

# ================================================================ 모의실험
L = 20.0
f = lambda x: 27 + 7 * np.exp(-((x - 8) / 3.5) ** 2) + 0.1 * x
n, sig, R = 20, 1.0, 2000
x = (np.arange(n) + 0.5) * L / n
xg = np.linspace(0, L, 401)
T = lambda v: v / L * 2 - 1                     # x를 −1~1로 바꿔 다항식 계산을 안정시킴


def coef(d, y, lam=0.0):
    X = np.vander(T(x), d + 1, increasing=True)
    P = np.eye(d + 1) * lam
    P[0, 0] = 0                                  # 절편은 벌점에서 뺌
    return np.linalg.solve(X.T @ X + P, X.T @ y)


rng = np.random.default_rng(14)


def run(d, lam=0.0):
    G = np.vander(T(xg), d + 1, increasing=True)
    X = np.vander(T(x), d + 1, increasing=True)
    preds = np.empty((R, len(xg)))
    tr = np.empty(R)
    te = np.empty(R)
    for r in range(R):
        y = f(x) + rng.normal(0, sig, n)
        c = coef(d, y, lam)
        preds[r] = G @ c
        tr[r] = np.mean((y - X @ c) ** 2)
        yt = f(xg) + rng.normal(0, sig, len(xg))
        te[r] = np.mean((yt - preds[r]) ** 2)
    m = preds.mean(0)
    b2 = np.mean((m - f(xg)) ** 2)
    var = np.mean(preds.var(0))
    return b2, var, b2 + var + sig ** 2, te.mean(), tr.mean()


DEG = list(range(13))
SIM = np.array([run(d) for d in DEG])
for d in DEG:
    print("deg", d, "bias2 %.3f var %.3f sum %.3f test %.3f train %.3f" % tuple(SIM[d]))
for lam in (1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0):
    print("ridge deg12 lam", lam, "bias2 %.3f var %.3f sum %.3f test %.3f train %.3f" % run(12, lam))


def dpath(s, pts, cls, w, dash=None, ylo=None, yhi=None, sy=None):
    """선 그리기. ylo·yhi를 주면 범위 밖 구간은 끊어서 그림(clipPath는 id가 필요해 쓰지 않음)"""
    segs, cur = [], []
    for a, b, v in pts:
        if ylo is not None and not (ylo <= v <= yhi):
            if len(cur) > 1:
                segs.append(cur)
            cur = []
            continue
        cur.append((a, b))
    if len(cur) > 1:
        segs.append(cur)
    d = " ".join("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in sg) for sg in segs)
    s.path(d, cls, w)
    if dash:
        s.parts[-1] = s.parts[-1].replace("/>", f' stroke-dasharray="{dash}"/>')


# ---------------------------------------------------------------- 그림 1: 차수별 적합 곡선
s = Svg(680, 250, "차수 1, 7, 12 다항식을 서로 다른 표본 15벌에 맞춘 곡선과 참 곡선")
rng2 = np.random.default_rng(140)
YS = [f(x) + rng2.normal(0, sig, n) for _ in range(15)]
YLO, YHI = 23.0, 39.0
PW, GAP, L0, T0, B0 = 180, 32, 48, 34, 204
for k, (d, title, tcls) in enumerate(((1, "1차: 편향이 큼", "f-bd"), (7, "7차: 균형", "f-ok"), (12, "12차: 분산이 큼", "f-ac"))):
    l = L0 + k * (PW + GAP)
    r = l + PW
    sx = lambda v, l=l: l + PW * v / L
    sy = lambda v: B0 - (B0 - T0) * (v - YLO) / (YHI - YLO)
    s.rect(l, T0, PW, B0 - T0, "s-mu f-bg", 1)
    xf = np.linspace(0, L, 161)                  # 그림 용량을 줄이려고 곡선은 161점으로 그림
    for yy in YS:
        c = coef(d, yy)
        p = np.polynomial.polynomial.polyval(T(xf), c)
        dpath(s, [(sx(a), sy(v), v) for a, v in zip(xf, p)], "s-ac", 0.9, None, YLO, YHI)
    dpath(s, [(sx(a), sy(v), v) for a, v in zip(xg, f(xg))], "s-fg", 2.4, "6 4")
    for a, v in zip(x, YS[0]):
        s.circle(sx(a), sy(v), 2.4, "f-bd")
    s.text((l + r) / 2, T0 - 12, title, tcls, 13, weight="600")
    for v in (0, 10, 20):
        s.line(sx(v), B0, sx(v), B0 + 4, "s-mu", 1.2)
        s.text(sx(v), B0 + 17, f"{v}", "f-mu", 11)
    if k == 0:
        for v in (25, 30, 35):
            s.line(l - 4, sy(v), l, sy(v), "s-mu", 1.2)
            s.text(l - 7, sy(v) + 4, f"{v}", "f-mu", 11, anchor="end")
s.text(L0 + 1.5 * PW + GAP, B0 + 38, "단면선 위치(km)   ·   세로축 LST(℃)   ·   점선: 참 곡선, 점: 표본 한 벌", "f-mu", 12)
s.save(os.path.join(OUT, "14-fits.svg"))

# ---------------------------------------------------------------- 그림 2: 편향²·분산·잡음
s = Svg(680, 300, "다항식 차수에 따른 편향 제곱, 분산, 잡음, 시험 오차, 학습 오차")
L1, R1, T1, B1 = 64, 600, 20, 250
YMAX = 7.2
sx = lambda d: L1 + (R1 - L1) * d / 12
sy = lambda v: B1 - (B1 - T1) * v / YMAX
s.line(L1, B1, R1, B1, "s-mu", 1.2)
s.line(L1, T1, L1, B1, "s-mu", 1.2)
for d in range(0, 13, 2):
    s.line(sx(d), B1, sx(d), B1 + 5, "s-mu", 1.2)
    s.text(sx(d), B1 + 19, f"{d}", "f-mu", 11)
for v in range(0, 8):
    s.line(L1 - 5, sy(v), L1, sy(v), "s-mu", 1.2)
    s.text(L1 - 9, sy(v) + 4, f"{v}", "f-mu", 11, anchor="end")
s.text((L1 + R1) / 2, B1 + 40, "다항식 차수(모델 복잡도) →", "f-fg", 12)
SER = [  # (열, 선, 굵기, 점선, 이름, 글자색)
    (2, "s-fg", 2.8, None, "시험 오차 = 편향² + 분산 + 잡음", "f-fg"),
    (0, "s-bd", 2.0, None, "편향²", "f-bd"),
    (1, "s-ac", 2.0, None, "분산", "f-ac"),
    (4, "s-ok", 1.8, "5 3", "학습 오차", "f-ok"),
]
dpath(s, [(sx(0), sy(1.0), 0), (sx(12), sy(1.0), 0)], "s-mu", 1.6, "2 3")
for col, cls, w, dash, name, tcls in SER:
    dpath(s, [(sx(d), sy(SIM[d, col]), 0) for d in DEG], cls, w, dash)
# 범례(차수 3~10 위쪽은 곡선이 지나지 않는 빈 곳)
LX, LY = sx(3.3), sy(6.6)
for i, (col, cls, w, dash, name, tcls) in enumerate(SER + [(None, "s-mu", 1.6, "2 3", "잡음 σ² = 1 (줄일 수 없는 오차)", "f-mu")]):
    yy = LY + i * 19
    dpath(s, [(LX, yy - 4, 0), (LX + 30, yy - 4, 0)], cls, w + 0.4, dash)
    s.text(LX + 40, yy, name, tcls, 12, anchor="start", weight="600" if col == 2 else None)
dmin = int(np.argmin(SIM[:, 2]))
s.circle(sx(dmin), sy(SIM[dmin, 2]), 4.5, "f-fg")
s.text(sx(dmin), sy(SIM[dmin, 2]) - 12, f"최소 {SIM[dmin, 2]:.2f}", "f-fg", 12)
s.text(sx(0.9), sy(0.35) - 30, "과소적합", "f-bd", 12)
s.text(sx(11.3), sy(0.35) - 30, "과적합", "f-ac", 12)
s.save(os.path.join(OUT, "14-bv-curve.svg"))

# ================================================================ 학습곡선 (3부 공통 자료)
dd = pd.read_csv(os.path.join(HERE, "..", "data", "part3_landcover.csv"))
F = ["B2", "B3", "B4", "B8", "B11", "B12", "ndvi", "ndwi"]
X = dd[F].values
y = dd.cls.values
g = dd.patch_id.values
gkf = GroupKFold(n_splits=5)

rows = []
for depth in list(range(1, 16)) + [None]:
    r = cross_validate(DecisionTreeClassifier(max_depth=depth, random_state=0), X, y, groups=g, cv=gkf, return_train_score=True)
    rows.append((depth or 99, r["train_score"].mean(), r["test_score"].mean(), r["test_score"].std(ddof=1) / np.sqrt(5)))
    print("depth", depth, "train %.3f val %.3f se %.3f" % rows[-1][1:])
rows = np.array(rows)
ib = rows[:, 2].argmax()
thr = rows[ib, 2] - rows[ib, 3]
print("best depth", rows[ib, 0], "1-SE threshold %.3f" % thr, "pick depth", rows[rows[:, 2] >= thr][0, 0])
full = DecisionTreeClassifier(random_state=0).fit(X, y)
print("unlimited tree depth", full.get_depth(), "leaves", full.get_n_leaves())
rk = cross_validate(DecisionTreeClassifier(random_state=0), X, y, cv=KFold(5, shuffle=True, random_state=0))
print("random 5-fold val %.3f" % rk["test_score"].mean())

rng3 = np.random.default_rng(0)
FR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LC = {3: [], None: []}
for frac in FR:
    acc = {3: [[], []], None: [[], []]}
    npol = []
    for tr, te in gkf.split(X, y, g):
        pats = np.unique(g[tr])
        for _ in range(1 if frac == 1.0 else 20):
            k = max(6, int(round(frac * len(pats))))
            sel = rng3.choice(pats, k, replace=False)
            m_ = np.isin(g, sel) & np.isin(np.arange(len(g)), tr)
            npol.append(k)
            for dep in (3, None):
                m = DecisionTreeClassifier(max_depth=dep, random_state=0).fit(X[m_], y[m_])
                acc[dep][0].append(m.score(X[m_], y[m_]))
                acc[dep][1].append(m.score(X[te], y[te]))
    for dep in (3, None):
        LC[dep].append((np.mean(npol), np.mean(acc[dep][0]), np.mean(acc[dep][1])))
    print("frac", frac, "polygons %.1f" % np.mean(npol),
          "| depth3 train %.3f val %.3f" % (np.mean(acc[3][0]), np.mean(acc[3][1])),
          "| unlimited train %.3f val %.3f" % (np.mean(acc[None][0]), np.mean(acc[None][1])))

# ---------------------------------------------------------------- 그림 3: 학습곡선 두 개
s = Svg(680, 270, "결정트리 깊이 3과 깊이 제한 없음의 학습곡선: 학습 폴리곤 수에 따른 학습·검증 정확도")
PW, GAP, L0, T0, B0 = 260, 66, 55, 36, 214
YLO, YHI = 0.6, 1.0
for k, (dep, title) in enumerate(((3, "깊이 3 (단순)"), (None, "깊이 제한 없음 (복잡)"))):
    l = L0 + k * (PW + GAP)
    sx = lambda v, l=l: l + PW * v / 90
    sy = lambda v: B0 - (B0 - T0) * (v - YLO) / (YHI - YLO)
    s.line(l, B0, l + PW, B0, "s-mu", 1.2)
    s.line(l, T0, l, B0, "s-mu", 1.2)
    for v in (0, 20, 40, 60, 80):
        s.line(sx(v), B0, sx(v), B0 + 4, "s-mu", 1.2)
        s.text(sx(v), B0 + 17, f"{v}", "f-mu", 11)
    for v in (0.6, 0.7, 0.8, 0.9, 1.0):
        s.line(l - 4, sy(v), l, sy(v), "s-mu", 1.2)
        s.text(l - 7, sy(v) + 4, f"{v:.1f}", "f-mu", 11, anchor="end")
    s.text(l + PW / 2, T0 - 16, title, "f-fg", 13, weight="600")
    P = np.array(LC[dep])
    dpath(s, [(sx(a), sy(b), 0) for a, b, _ in P], "s-ac", 2.2)
    dpath(s, [(sx(a), sy(c), 0) for a, _, c in P], "s-bd", 2.2)
    for a, b, c in P:
        s.circle(sx(a), sy(b), 3, "f-ac")
        s.circle(sx(a), sy(c), 3, "f-bd")
    # 오른쪽 끝의 틈 표시
    a, b, c = P[-1]
    gx = sx(a) + 12
    s.line(gx, sy(b), gx, sy(c), "s-mu", 1.2)
    s.line(gx - 4, sy(b), gx, sy(b), "s-mu", 1.2)
    s.line(gx - 4, sy(c), gx, sy(c), "s-mu", 1.2)
    s.text(gx + 5, (sy(b) + sy(c)) / 2 + 4, f"{b - c:.2f}", "f-mu", 11, anchor="start")
    if dep == 3:
        s.text(sx(30), sy(0.93), "학습", "f-ac", 12, weight="600")
        s.text(sx(30), sy(0.70), "검증", "f-bd", 12, weight="600")
    else:
        s.text(sx(30), sy(0.975), "학습", "f-ac", 12, weight="600")
        s.text(sx(30), sy(0.745), "검증", "f-bd", 12, weight="600")
s.text(L0 + PW + GAP / 2, B0 + 42, "학습에 쓴 폴리곤 수 (검증은 폴리곤 단위 5겹)   ·   세로축 정확도", "f-mu", 12)
s.save(os.path.join(OUT, "14-lcurve.svg"))
print("ok")
