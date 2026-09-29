"""4강 그림: 1·2종 오류와 검정력, 검정 횟수에 따른 가족별 오류율"""
import math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

OUT = os.path.join(os.path.dirname(__file__), "..", "fig")
phi = lambda x, m=0.0: math.exp(-0.5 * (x - m) ** 2) / math.sqrt(2 * math.pi)

# ---------------------------------------------------------------- 1·2종 오류
s = Svg(680, 270, "귀무가설 분포와 대립가설 분포, 기각 경계, 1종 오류 α, 2종 오류 β, 검정력")
D, C = 2.5, 1.96                                  # 참 효과(표준오차 단위), 양측 5% 기각 경계
XL, XR = -3.6, 6.4
X0, X1, Y0, H = 40, 640, 225, 150
sx = lambda v: X0 + (X1 - X0) * (v - XL) / (XR - XL)
sy = lambda p: Y0 - H * p / phi(0)
N = 400
xs = [XL + (XR - XL) * i / N for i in range(N + 1)]

def area(a, b, m, cls):
    pts = [x for x in xs if a <= x <= b]
    pts = [a] + pts + [b]
    d = f"M {sx(a):.1f} {Y0} " + " ".join(f"L {sx(x):.1f} {sy(phi(x, m)):.1f}" for x in pts) + f" L {sx(b):.1f} {Y0} Z"
    s.path(d, cls, 0)

area(XL, C, D, "f-acs")                           # β: H1인데 경계 안쪽
area(C, XR, 0, "f-bd")                            # α/2 오른쪽
area(XL, -C, 0, "f-bd")                           # α/2 왼쪽
for m, cls in ((0, "s-fg"), (D, "s-ac")):
    s.path("M " + " L ".join(f"{sx(x):.1f} {sy(phi(x, m)):.1f}" for x in xs), cls, 2)
s.line(X0, Y0, X1, Y0, "s-mu", 1.2)
for c in (-C, C):
    s.line(sx(c), 40, sx(c), Y0, "s-bd", 1.4, "5 4")
s.text(sx(C), 32, "기각 경계", "f-bd", 12)
s.text(sx(-C), 32, "기각 경계", "f-bd", 12)
s.text(sx(0), sy(phi(0)) - 8, "H₀: 차이 없음", "f-fg", 13, weight="600")
s.text(sx(D), sy(phi(0)) - 8, "H₁: 실제 차이 있음", "f-ac", 13, weight="600")
# β 라벨 (H1 곡선 아래 왼쪽)
s.text(sx(1.35), sy(phi(1.35, D)) + 30, "β", "f-ac", 14, weight="700")
# α/2 라벨: 지시선
s.line(sx(2.35), sy(phi(2.35)) - 2, sx(3.2), 196, "s-mu", 1)
s.text(sx(3.25), 200, "α/2", "f-bd", 12, anchor="start")
s.line(sx(-2.25), sy(phi(-2.25)) - 2, sx(-2.6), 196, "s-mu", 1)
s.text(sx(-2.65), 192, "α/2", "f-bd", 12, anchor="end")
s.text(sx(-2.65), 174, "1종 오류", "f-bd", 12, anchor="end")
s.text(sx(4.4), sy(phi(3.6, D)) - 20, "검정력 1−β", "f-ac", 12, anchor="start")
s.line(sx(4.35), sy(phi(3.6, D)) - 24, sx(3.3), sy(phi(3.3, D)) + 18, "s-mu", 1)
s.text(sx(0), Y0 + 20, "0", "f-mu", 12)
s.text(sx(D), Y0 + 20, "참 효과", "f-mu", 12)
s.text(X1, Y0 + 20, "검정통계량", "f-mu", 12, anchor="end")
s.save(os.path.join(OUT, "04-errors.svg"))

# ---------------------------------------------------------------- 검정 횟수와 FWER
s = Svg(680, 250, "검정 횟수 m에 따라 거짓 양성이 하나 이상 나올 확률 1−(1−α)^m이 커지는 곡선")
X0, X1, Y0, Y1 = 70, 640, 205, 25
M = 100
sx = lambda m: X0 + (X1 - X0) * (m - 1) / (M - 1)
sy = lambda p: Y0 - (Y0 - Y1) * p
f = lambda m: 1 - 0.95 ** m
for v in (0, 0.25, 0.5, 0.75, 1.0):
    s.line(X0, sy(v), X1, sy(v), "s-mu", 0.6, "2 4" if v else None)
    s.text(X0 - 8, sy(v) + 4, f"{v:g}", "f-mu", 12, anchor="end")
for m in (1, 20, 40, 60, 80, 100):
    s.text(sx(m), Y0 + 18, str(m), "f-mu", 12)
s.text((X0 + X1) / 2, Y0 + 38, "검정 횟수 m (α = 0.05, 서로 독립)", "f-mu", 12)
s.path("M " + " L ".join(f"{sx(m):.1f} {sy(f(m)):.1f}" for m in range(1, M + 1)), "s-ac", 2.5)
for m, dx, dy, anc in ((1, 16, -6, "start"), (10, 12, 14, "start"), (14, 12, 14, "start"), (50, 0, 22, "middle")):
    s.circle(sx(m), sy(f(m)), 4.5, "f-bd")
    s.text(sx(m) + dx, sy(f(m)) + dy, f"m={m}: {f(m):.2f}", "f-bd", 12, anchor=anc)
s.save(os.path.join(OUT, "04-fwer.svg"))
print("ok")
