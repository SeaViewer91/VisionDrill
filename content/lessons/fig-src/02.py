"""2강 그림: 포아송 분포(λ 세 개), 중심극한정리(치우친 분포에서 표본평균의 분포)"""
import math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

OUT = os.path.join(os.path.dirname(__file__), "..", "fig")

# ---------------------------------------------------------------- 포아송 분포
pois = lambda k, lam: math.exp(-lam) * lam ** k / math.factorial(k)
KMAX = 20
W, ROW_H, TOP = 680, 78, 18
s = Svg(W, TOP + 3 * ROW_H + 40, "포아송 분포: 평균 λ가 1, 4, 10일 때 개수별 확률 막대")
PX0, PX1 = 130, 660                      # 막대 영역
step = (PX1 - PX0) / (KMAX + 1)
cx = lambda k: PX0 + step * (k + 0.5)
PMAX = pois(1, 1)                        # 0.368, 세 줄 공통 배율
BAR_H = ROW_H - 18
for r, lam in enumerate((1, 4, 10)):
    base = TOP + (r + 1) * ROW_H
    s.line(PX0, base, PX1, base, "s-mu", 1)
    for k in range(KMAX + 1):
        h = BAR_H * pois(k, lam) / PMAX
        if h >= 0.5:
            s.rect(cx(k) - step * 0.36, base - h, step * 0.72, h, "f-ac", 0)
    s.line(cx(lam), base - BAR_H - 4, cx(lam), base, "s-bd", 1.4, "4 3")
    s.text(12, base - BAR_H / 2 + 2, f"λ = {lam}", "f-fg", 14, anchor="start", weight="600")
    s.text(12, base - BAR_H / 2 + 20, f"평균·분산 {lam}", "f-mu", 12, anchor="start")
axis = TOP + 3 * ROW_H
for k in range(0, KMAX + 1, 5):
    s.line(cx(k), axis, cx(k), axis + 5, "s-mu", 1)
    s.text(cx(k), axis + 20, str(k), "f-mu", 12)
s.text(PX1, axis + 36, "타일당 개수", "f-mu", 12, anchor="end")
s.save(os.path.join(OUT, "02-poisson.svg"))

# ---------------------------------------------------------------- 중심극한정리
TH = 1.2                                  # 원래 분포: 지수분포, 평균 1.2 ha
def gamma_pdf(x, n):                      # n개 평균의 분포 = 감마(형상 n, 척도 TH/n)
    if x <= 0:
        return 0.0
    sc = TH / n
    return math.exp((n - 1) * math.log(x) - x / sc - math.lgamma(n) - n * math.log(sc))
norm_pdf = lambda x, m, sd: math.exp(-0.5 * ((x - m) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))

s = Svg(680, 215, "중심극한정리: 오른쪽으로 치우친 분포에서 뽑은 표본의 평균은 n이 커질수록 종 모양이 됨")
XMAX = 3.2
PW, GAP, L0, Y0, PH = 196, 26, 20, 170, 120
for i, n in enumerate((1, 4, 36)):
    x0 = L0 + i * (PW + GAP)
    sx = lambda v, x0=x0: x0 + PW * v / XMAX
    sd = TH / math.sqrt(n)
    xs = [0.0005 + j * XMAX / 240 for j in range(241)]
    peak = max(max(gamma_pdf(x, n) for x in xs), norm_pdf(TH, TH, sd))
    sy = lambda p, peak=peak: Y0 - PH * p / peak
    pts = [(sx(x), sy(gamma_pdf(x, n))) for x in xs]
    area = f"M {sx(xs[0]):.1f} {Y0} " + " ".join(f"L {a:.1f} {b:.1f}" for a, b in pts) + f" L {sx(xs[-1]):.1f} {Y0} Z"
    s.path(area, "f-acs", 0)
    s.path("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts), "s-ac", 2)
    if n > 1:
        npts = [(sx(x), sy(norm_pdf(x, TH, sd))) for x in xs if norm_pdf(x, TH, sd) * PH / peak > 0.3]
        d = "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in npts)
        s.add(f'<path d="{d}" class="s-bd" stroke-width="1.6" stroke-dasharray="5 3" fill="none"/>')
    s.line(x0, Y0, x0 + PW, Y0, "s-mu", 1)
    for v in (0, 1.2, 2.4):
        s.line(sx(v), Y0, sx(v), Y0 + 4, "s-mu", 1)
        s.text(sx(v), Y0 + 18, f"{v:g}", "f-mu", 11)
    s.line(sx(TH), Y0 - PH - 2, sx(TH), Y0, "s-mu", 1, "3 3")
    s.text(x0 + PW - 4, 30, f"n = {n}", "f-fg", 14, anchor="end", weight="600")
s.text(20 + 3 * PW + 2 * GAP, Y0 + 36, "표본평균 (ha)", "f-mu", 12, anchor="end")
s.save(os.path.join(OUT, "02-clt.svg"))
print("ok")
