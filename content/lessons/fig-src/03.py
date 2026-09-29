"""3강 그림: 반복 표본의 95% 신뢰구간, 표본 크기에 따른 표준오차"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

OUT = os.path.join(os.path.dirname(__file__), "..", "fig")
Z = 1.959963984540054

# ---------------------------------------------------------------- 신뢰구간 20개
TRUE, N, K = 0.88, 400, 20
rng = np.random.default_rng(4)                    # 20개 중 1개가 참값을 못 덮는 시드
p = rng.binomial(N, TRUE, K) / N
h = Z * np.sqrt(p * (1 - p) / N)
miss = (p - h > TRUE) | (p + h < TRUE)
assert miss.sum() == 1

s = Svg(680, 350, "참값 0.88인 모집단에서 표본을 20번 뽑아 만든 95% 신뢰구간 20개 중 1개가 참값을 덮지 못함")
X0, X1, V0, V1 = 70, 500, 0.80, 0.96
sx = lambda v: X0 + (X1 - X0) * (v - V0) / (V1 - V0)
Y0, DY = 36, 13
for i in range(K):
    y = Y0 + i * DY
    cls, dot = ("s-bd", "f-bd") if miss[i] else ("s-ac", "f-ac")
    s.line(sx(p[i] - h[i]), y, sx(p[i] + h[i]), y, cls, 2.2)
    s.circle(sx(p[i]), y, 3, dot)
yb = Y0 + (K - 1) * DY + 16                       # 축 위치
s.line(sx(TRUE), 22, sx(TRUE), yb, "s-fg", 1.4, "5 4")
s.text(sx(TRUE), 16, "참값 0.88 (고정)", "f-fg", 12, weight="600")
s.line(X0, yb, X1, yb, "s-mu", 1.2)
for v in (0.80, 0.84, 0.88, 0.92, 0.96):
    s.line(sx(v), yb, sx(v), yb + 5, "s-mu", 1.2)
    s.text(sx(v), yb + 19, f"{v:.2f}", "f-mu", 12)
s.text((X0 + X1) / 2, yb + 38, "전체 정확도", "f-mu", 12)
s.text(X0 - 12, Y0 + 4, "표본 1", "f-mu", 11, anchor="end")
s.text(X0 - 12, Y0 + (K - 1) * DY + 4, "표본 20", "f-mu", 11, anchor="end")
# 범례
LX, LY = 528, 70
s.line(LX, LY, LX + 30, LY, "s-ac", 2.2); s.circle(LX + 15, LY, 3, "f-ac")
s.text(LX + 40, LY + 4, "참값을 덮음", "f-fg", 12, anchor="start")
s.line(LX, LY + 26, LX + 30, LY + 26, "s-bd", 2.2); s.circle(LX + 15, LY + 26, 3, "f-bd")
s.text(LX + 40, LY + 30, "못 덮음", "f-bd", 12, anchor="start")
s.text(LX, LY + 66, "점 = 점추정치", "f-mu", 12, anchor="start")
s.text(LX, LY + 86, "선 = 95% 구간", "f-mu", 12, anchor="start")
s.text(LX, LY + 116, f"덮은 구간 {K - int(miss.sum())}/{K}", "f-fg", 12, anchor="start", weight="600")
s.save(os.path.join(OUT, "03-ci-cover.svg"))

# ---------------------------------------------------------------- 표준오차 곡선
P = 0.9
se = lambda n: math.sqrt(P * (1 - P) / n)
s = Svg(680, 250, "정확도 0.9일 때 표본 수에 따른 표준오차. 표본을 4배로 늘리면 표준오차가 절반이 됨")
X0, X1, Y0, Y1 = 70, 640, 205, 25
NMAX, SMAX = 2000, 0.06
sx = lambda n: X0 + (X1 - X0) * n / NMAX
sy = lambda v: Y0 - (Y0 - Y1) * v / SMAX
ns = np.linspace(25, NMAX, 300)
s.path("M " + " L ".join(f"{sx(n):.1f} {sy(se(n)):.1f}" for n in ns), "s-ac", 2.2)
s.line(X0, Y0, X1, Y0, "s-mu", 1.2); s.line(X0, Y0, X0, Y1, "s-mu", 1.2)
for n in (0, 500, 1000, 1500, 2000):
    s.line(sx(n), Y0, sx(n), Y0 + 5, "s-mu", 1.2)
    s.text(sx(n), Y0 + 19, f"{n}", "f-mu", 12)
s.text((X0 + X1) / 2, Y0 + 38, "표본 수 n", "f-mu", 12)
for v in (0, 0.02, 0.04, 0.06):
    s.line(X0 - 5, sy(v), X0, sy(v), "s-mu", 1.2)
    s.text(X0 - 9, sy(v) + 4, f"{v:.2f}", "f-mu", 12, anchor="end")
s.text(X0 + 8, Y1 - 6, "표준오차", "f-mu", 12, anchor="start")
labels = {100: (14, -8, "start", "0.030"), 400: (10, -10, "start", "0.015"), 1600: (0, -14, "middle", "0.0075")}
for n, (dx, dy, anc, txt) in labels.items():
    s.line(sx(n), sy(se(n)), sx(n), Y0, "s-mu", 1, "3 3")
    s.circle(sx(n), sy(se(n)), 4.5, "f-bd")
    s.text(sx(n) + dx, sy(se(n)) + dy, f"n={n}: {txt}", "f-bd", 12, anchor=anc, weight="600")
s.text(sx(1000), sy(0.042), "n을 4배로 늘릴 때마다 표준오차는 절반", "f-fg", 13, weight="600")
s.save(os.path.join(OUT, "03-se-curve.svg"))
print("ok", p.round(4).tolist(), int(miss.argmax()))
