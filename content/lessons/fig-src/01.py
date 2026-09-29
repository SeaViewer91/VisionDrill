"""1강 그림: 박스플롯 해부도, 오른쪽 꼬리 분포의 대표값 위치"""
import math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

OUT = os.path.join(os.path.dirname(__file__), "..", "fig")

# ---------------------------------------------------------------- 박스플롯 해부도
s = Svg(680, 195, "박스플롯 해부도: 사분위수, IQR, 수염, 울타리, 이상치")
X0, X1 = 40, 640          # 값 축 범위(가상의 값 0~12)
sx = lambda v: X0 + (X1 - X0) * v / 12
q1, med, q3 = 4.0, 5.0, 6.5
iqr = q3 - q1
lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr      # 0.25, 10.25
w_lo, w_hi = 1.2, 9.4                             # 울타리 안 가장 바깥 관측값
outs = [11.3]
y = 110
s.rect(sx(q1), y - 28, sx(q3) - sx(q1), 56, "s-fg f-acs", 1.8, 3)
s.line(sx(med), y - 28, sx(med), y + 28, "s-ac", 3)
s.line(sx(w_lo), y, sx(q1), y); s.line(sx(q3), y, sx(w_hi), y)
s.line(sx(w_lo), y - 14, sx(w_lo), y + 14); s.line(sx(w_hi), y - 14, sx(w_hi), y + 14)
for v in (lo_f, hi_f):
    s.line(sx(v), 50, sx(v), 170, "s-bd", 1.4, "5 4")
for v in outs:
    s.circle(sx(v), y, 6, "f-bd")
# 라벨
s.text(sx(q1), 66, "Q1", "f-fg", 13, weight="600")
s.text(sx(med), 66, "중앙값", "f-ac", 13, weight="600")
s.text(sx(q3), 66, "Q3", "f-fg", 13, weight="600")
s.line(sx(q1), 152, sx(q3), 152, "s-mu", 1.2)
s.line(sx(q1), 146, sx(q1), 158, "s-mu", 1.2); s.line(sx(q3), 146, sx(q3), 158, "s-mu", 1.2)
s.text((sx(q1) + sx(q3)) / 2, 174, "IQR = Q3 − Q1 (가운데 50%)", "f-mu", 12)
s.text((sx(w_lo) + sx(q1)) / 2, y - 10, "수염", "f-mu", 12)
s.text((sx(q3) + sx(w_hi)) / 2, y - 10, "수염", "f-mu", 12)
s.text(sx(lo_f), 42, "Q1 − 1.5×IQR", "f-bd", 12)
s.text(sx(hi_f), 42, "Q3 + 1.5×IQR", "f-bd", 12)
s.text(sx(outs[0]), y + 28, "이상치 후보", "f-bd", 12)
s.save(os.path.join(OUT, "01-boxplot.svg"))

# ---------------------------------------------------------------- 오른쪽 꼬리 분포
s = Svg(680, 235, "오른쪽 꼬리가 긴 분포에서 최빈값, 중앙값, 평균의 위치")
k, th = 2.0, 1.0                                   # 감마(형상 2, 척도 1)
pdf = lambda x: x ** (k - 1) * math.exp(-x / th) / (math.gamma(k) * th ** k)
mode, mean = (k - 1) * th, k * th
# 중앙값: 수치 적분으로 CDF = 0.5
acc, x, dx = 0.0, 0.0, 1e-4
while acc < 0.5:
    acc += pdf(x + dx / 2) * dx; x += dx
median = x
X0, X1, Y0, H = 50, 640, 220, 170
xmax = 8.0
sx = lambda v: X0 + (X1 - X0) * v / xmax
sy = lambda p: Y0 - H * p / pdf(mode)
pts = [(sx(i * xmax / 300), sy(pdf(i * xmax / 300))) for i in range(301)]
area = "M {:.1f} {:.1f} ".format(sx(0), Y0) + " ".join(f"L {a:.1f} {b:.1f}" for a, b in pts) + f" L {sx(xmax):.1f} {Y0} Z"
s.path(area, "f-acs", 0)
s.path("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts), "s-fg", 2)
s.line(X0, Y0, X1, Y0, "s-mu", 1.2)
for v, name, cls, dy in ((mode, "최빈값", "s-mu", 0), (median, "중앙값", "s-ac", 18), (mean, "평균", "s-bd", 36)):
    s.line(sx(v), sy(pdf(v)) - 4, sx(v), Y0, cls, 2, None if name != "최빈값" else "4 3")
    tcls = {"s-mu": "f-mu", "s-ac": "f-ac", "s-bd": "f-bd"}[cls]
    ly = 62 + dy * 1.4
    s.line(430, ly - 5, 460, ly - 5, cls, 2.5, None if name != "최빈값" else "4 3")
    s.text(470, ly, f"{name} {v:.2f}", tcls, 13, anchor="start", weight="600")
s.text(sx(4.3), 160, "긴 꼬리가 평균을 오른쪽으로 끌어당김", "f-mu", 12, anchor="start")
s.save(os.path.join(OUT, "01-skew.svg"))
print("ok", round(median, 3))
