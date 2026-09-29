"""5강 그림: 같은 피어슨 r의 서로 다른 산점도, 모란 지수 격자 세 개, LISA 지도"""
import os, sys
import numpy as np
from scipy import stats
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

OUT = os.path.join(os.path.dirname(__file__), "..", "fig")

# ---------------------------------------------------------------- 공통: 격자 인접·모란 지수
def weights(R, C, kind="rook"):
    """R×C 격자의 이진 인접 행렬. rook = 변 공유, queen = 변·꼭짓점 공유"""
    n = R * C
    W = np.zeros((n, n))
    for r in range(R):
        for c in range(C):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0 or (kind == "rook" and dr and dc):
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < R and 0 <= cc < C:
                        W[r * C + c, rr * C + cc] = 1
    return W


def moran(x, W):
    z = x - x.mean()
    return len(x) / W.sum() * (z @ W @ z) / (z @ z)


# ---------------------------------------------------------------- 1. 같은 r, 다른 모양
n, T = 15, 0.80
x = np.linspace(0, 1, n)
p = brentq(lambda p: stats.pearsonr(x, x ** p)[0] - T, 3, 10)
yb = x ** p                                                    # 단조 곡선
rng = np.random.default_rng(3)
e = rng.normal(size=n)
e -= np.polyval(np.polyfit(x, e, 1), x)
sa = brentq(lambda s: stats.pearsonr(x, x + s * e)[0] - T, 0.01, 5)
ya = x + sa * e                                                # 직선 + 잡음
rng = np.random.default_rng(23)
xc = np.append(rng.uniform(0.05, 0.45, n - 1), 0.95)
yc0 = rng.uniform(0.05, 0.45, n - 1)
h = brentq(lambda h: stats.pearsonr(xc, np.append(yc0, h))[0] - T, 0.3, 5)
yc = np.append(yc0, h)                                         # 무관한 점 + 이상치 1개

panels = [("직선 + 잡음", x, ya), ("휘어진 단조 관계", x, yb), ("무관한 점 + 이상치 1개", xc, yc)]
s = Svg(680, 270, "피어슨 r이 모두 0.80으로 같지만 모양이 다른 산점도 세 개와 각각의 스피어만 ρ")
PW, PH, TOP, GAP, LEFT = 190, 160, 40, 35, 25
for k, (title, px, py) in enumerate(panels):
    x0 = LEFT + k * (PW + GAP)
    y0 = TOP + PH
    xlo, xhi = px.min(), px.max()
    ylo, yhi = py.min(), py.max()
    mx = lambda v: x0 + 10 + (PW - 20) * (v - xlo) / (xhi - xlo)
    my = lambda v: y0 - 10 - (PH - 20) * (v - ylo) / (yhi - ylo)
    s.line(x0, y0, x0 + PW, y0, "s-mu", 1.2)
    s.line(x0, TOP, x0, y0, "s-mu", 1.2)
    for a, b in zip(px, py):
        s.circle(mx(a), my(b), 4.5, "f-ac")
    r = stats.pearsonr(px, py)[0]
    rho = stats.spearmanr(px, py)[0]
    s.text(x0 + PW / 2, 24, title, "f-fg", 13, weight="600")
    s.text(x0 + PW / 2, y0 + 30, f"피어슨 r = {r:.2f}", "f-fg", 13)
    s.text(x0 + PW / 2, y0 + 52, f"스피어만 ρ = {rho:.2f}", "f-bd", 13, weight="600")
s.save(os.path.join(OUT, "05-same-r.svg"))
print("fig1 rho", [round(stats.spearmanr(a, b)[0], 3) for _, a, b in panels])

# ---------------------------------------------------------------- 2. 모란 지수 격자 세 개
pos = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1]], float)
v = np.array([1] * 18 + [0] * 18)
np.random.default_rng(1).shuffle(v)
rnd = v.reshape(6, 6).astype(float)
neg = (np.indices((6, 6)).sum(0) % 2).astype(float)
W6 = weights(6, 6, "rook")
grids = [("양의 자기상관", pos), ("무작위", rnd), ("음의 자기상관", neg)]
s = Svg(680, 250, "양, 무작위, 음의 공간 자기상관을 보이는 6×6 격자와 각각의 모란 지수")
CS = 30
for k, (title, g) in enumerate(grids):
    x0 = 30 + k * 220
    y0 = 36
    for r in range(6):
        for c in range(6):
            s.rect(x0 + c * CS, y0 + r * CS, CS, CS, "s-mu f-ac" if g[r, c] else "s-mu f-sf", 1)
    I = moran(g.ravel(), W6)
    s.text(x0 + 3 * CS, 24, title, "f-fg", 13, weight="600")
    s.text(x0 + 3 * CS, y0 + 6 * CS + 26, f"I = {I:.2f}".replace("-", "−"), "f-bd", 14, weight="600")
s.save(os.path.join(OUT, "05-moran.svg"))
print("fig2 I", [round(moran(g.ravel(), W6), 3) for _, g in grids], "E", round(-1 / 35, 3))

# ---------------------------------------------------------------- 3. LISA
g = np.array([
    [35.2, 35.8, 34.6, 31.6, 31.2, 31.8],
    [35.5, 36.1, 35.0, 31.3, 31.7, 31.5],
    [34.9, 35.3, 31.9, 31.4, 31.5, 31.6],
    [31.7, 31.3, 31.5, 27.4, 27.0, 27.6],
    [31.2, 31.8, 31.5, 27.2, 35.1, 26.9],
    [31.4, 31.6, 31.3, 27.5, 26.8, 27.3]])
xv = g.ravel()
N = len(xv)
Wq = weights(6, 6, "queen")
Wr = Wq / Wq.sum(1, keepdims=True)                             # 행 표준화
z = xv - xv.mean()
m2 = (z @ z) / N
lag = Wr @ z
Ii = z * lag / m2                                              # 국지 모란 지수
rng = np.random.default_rng(2026)
P = 9999
pv = np.ones(N)
for i in range(N):
    if z[i] == 0:
        continue
    others = np.delete(z, i)
    k = int(Wq[i].sum())
    sims = np.array([z[i] * rng.choice(others, k, replace=False).mean() / m2 for _ in range(P)])
    ext = (sims >= Ii[i]) if Ii[i] >= 0 else (sims <= Ii[i])
    pv[i] = (ext.sum() + 1) / (P + 1)                          # 조건부 순열 유사 p값
quad = np.where(z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
sig = pv < 0.05

s = Svg(680, 262, "가상 지표면온도 6×6 격자(왼쪽)와 LISA로 찾은 유의한 군집과 이상 지점(오른쪽)")
CS = 36
for k in range(2):
    x0 = 60 + k * 330
    y0 = 36
    s.text(x0 + 3 * CS, 24, ["값(℃)", "LISA 분류 (p < 0.05)"][k], "f-fg", 13, weight="600")
    for r in range(6):
        for c in range(6):
            i = r * 6 + c
            cx, cy = x0 + c * CS, y0 + r * CS
            if k == 0:
                cls = "s-mu f-bds" if xv[i] >= 34 else ("s-mu f-acs" if xv[i] <= 28 else "s-mu f-bg")
                s.rect(cx, cy, CS, CS, cls, 1)
                s.text(cx + CS / 2, cy + CS / 2 + 4, f"{xv[i]:.1f}", "f-fg", 11)
            else:
                if not sig[i]:
                    s.rect(cx, cy, CS, CS, "s-mu f-sf", 1)
                    continue
                q = quad[i]
                fill = {"HH": "s-mu f-bd", "LL": "s-mu f-ac", "HL": "s-bd f-bds", "LH": "s-ac f-acs"}[q]
                tcls = {"HH": "f-bg", "LL": "f-bg", "HL": "f-bd", "LH": "f-ac"}[q]
                s.rect(cx, cy, CS, CS, fill, 2 if q in ("HL", "LH") else 1)
                s.text(cx + CS / 2, cy + CS / 2 + 5, q, tcls, 13, weight="700")
s.save(os.path.join(OUT, "05-lisa.svg"))
print("fig3 global I (queen, row-std)", round((z @ lag) / (z @ z), 3), "mean", xv.mean())
print("fig3 sig", [(i // 6, i % 6, quad[i], round(pv[i], 4)) for i in range(N) if sig[i]])
