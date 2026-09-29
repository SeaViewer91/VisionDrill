"""6강 그림: 최소제곱 회귀선과 잔차, SST = SSR + SSE 분해, 의미 없는 변수를 넣을 때의 R²와 수정 R²

자료: content/lessons/data/part2_lst.csv (2부 공통 예제, 가상 도시 80개 행정동)
"""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "data", "part2_lst.csv"), encoding="utf-8")))
x = np.array([float(r["imperv_pct"]) for r in rows])
y = np.array([float(r["lst_c"]) for r in rows])
dong = [r["dong"] for r in rows]
n = len(y)

# 단순회귀 (최소제곱)
b1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
b0 = y.mean() - b1 * x.mean()
yh = b0 + b1 * x
e = y - yh
SST = np.sum((y - y.mean()) ** 2)
SSR = np.sum((yh - y.mean()) ** 2)
SSE = np.sum(e ** 2)
print(f"b0={b0:.4f} b1={b1:.4f} SST={SST:.1f} SSR={SSR:.1f} SSE={SSE:.1f} R2={SSR / SST:.4f}")


def minus(s):
    return s.replace("-", "−")


# ---------------------------------------------------------------- 1. 산점도 + 회귀선 + 잔차
s = Svg(680, 330, "행정동 80개의 불투수면 비율과 지표면온도 산점도, 최소제곱 회귀선, 점마다 회귀선까지 그은 잔차 선분")
L, R, T, B = 70, 650, 20, 280          # 그림 영역
XLO, XHI, YLO, YHI = 10, 100, 22, 42
sx = lambda v: L + (R - L) * (v - XLO) / (XHI - XLO)
sy = lambda v: B - (B - T) * (v - YLO) / (YHI - YLO)
# 축과 눈금
s.line(L, B, R, B, "s-mu", 1.2)
s.line(L, T, L, B, "s-mu", 1.2)
for v in range(10, 101, 10):
    s.line(sx(v), B, sx(v), B + 5, "s-mu", 1.2)
    s.text(sx(v), B + 19, f"{v}", "f-mu", 11)
for v in range(22, 43, 4):
    s.line(L - 5, sy(v), L, sy(v), "s-mu", 1.2)
    s.text(L - 9, sy(v) + 4, f"{v}", "f-mu", 11, anchor="end")
s.text((L + R) / 2, B + 40, "불투수면 비율 (%)", "f-fg", 12)
s.text(22, (T + B) / 2 - 16, "LST", "f-fg", 12)
s.text(22, (T + B) / 2 + 2, "(℃)", "f-fg", 12)
# 잔차 선분 → 점 → 회귀선 순서로 그려 선이 점 위를 덮지 않게 함
for xi, yi, fi in zip(x, y, yh):
    s.line(sx(xi), sy(yi), sx(xi), sy(fi), "s-bd", 1.1)
for xi, yi in zip(x, y):
    s.circle(sx(xi), sy(yi), 3.6, "f-ac")
s.line(sx(XLO), sy(b0 + b1 * XLO), sx(XHI), sy(b0 + b1 * XHI), "s-fg", 2.2)
# 라벨: 회귀식(왼쪽 위), 잔차 설명(최대 잔차 D43 옆)
s.text(L + 14, T + 18, f"ŷ = {b0:.2f} + {b1:.3f}x", "f-fg", 13, anchor="start", weight="600")
s.text(L + 14, T + 38, "주황 선분 = 잔차 (관측값 − 적합값)", "f-bd", 12, anchor="start")
k = int(np.argmax(e))
s.text(sx(x[k]) - 10, sy(y[k]) + 4, f"{dong[k]}  잔차 +{e[k]:.1f}℃", "f-bd", 12, anchor="end")
s.save(os.path.join(OUT, "06-ols-resid.svg"))

# ---------------------------------------------------------------- 2. SST = SSR + SSE 분해
s = Svg(680, 290, "한 관측값의 편차가 회귀로 설명된 부분과 잔차로 나뉘는 도식, 그리고 80개 동 전체의 SST가 SSR과 SSE로 나뉘는 막대")
# 왼쪽: 도식 (가상 좌표, 모양만 보여 줌)
L, R, T, B = 40, 330, 30, 250
s.line(L, B, R, B, "s-mu", 1.2)
s.line(L, T, L, B, "s-mu", 1.2)
ybar_y = 180                            # 평균선 높이
line = lambda px: 225 - 0.52 * (px - L)  # 회귀선 (화면 좌표)
s.line(L, ybar_y, R, ybar_y, "s-mu", 1.5, "5 4")
s.text(R, ybar_y + 18, "평균 ȳ", "f-mu", 12, anchor="end")
s.line(L, line(L), R, line(R), "s-fg", 2.2)
s.text(R, line(R) + 22, "회귀선 ŷ", "f-fg", 12, anchor="end")
px, py = 230, 52                        # 관측점
fy = line(px)
s.line(px, py, px, fy, "s-bd", 3)                  # 잔차
s.line(px, fy, px, ybar_y, "s-ac", 3)              # 회귀로 설명된 부분
# 전체 편차 괄호 (왼쪽)
bx = px - 22
s.line(bx, py, bx, ybar_y, "s-fg", 1.2)
s.line(bx, py, bx + 6, py, "s-fg", 1.2)
s.line(bx, ybar_y, bx + 6, ybar_y, "s-fg", 1.2)
s.text(bx - 6, (py + ybar_y) / 2 - 2, "yᵢ − ȳ", "f-fg", 12, anchor="end", weight="600")
s.text(bx - 6, (py + ybar_y) / 2 + 14, "(전체)", "f-fg", 11, anchor="end")
s.circle(px, py, 5, "f-ac")
s.text(px + 12, py + 12, "yᵢ − ŷᵢ (잔차)", "f-bd", 12, anchor="start", weight="600")
s.text(px + 12, (fy + ybar_y) / 2 + 4, "ŷᵢ − ȳ (설명됨)", "f-ac", 12, anchor="start", weight="600")
s.text(px, py - 10, "yᵢ", "f-fg", 12)
# 오른쪽: 실제 자료의 제곱합 막대
BX, BW, BT, BB = 440, 70, 40, 250
H = BB - BT
hSSR = H * SSR / SST
s.text(BX + BW / 2, BT - 14, f"SST = {SST:.1f}", "f-fg", 13, weight="600")
s.rect(BX, BB - hSSR, BW, hSSR, "s-ac f-acs", 1.5)
s.rect(BX, BT, BW, H - hSSR, "s-bd f-bds", 1.5)
s.text(BX + BW + 14, BB - hSSR / 2 - 4, f"SSR = {SSR:.1f}", "f-ac", 13, anchor="start", weight="600")
s.text(BX + BW + 14, BB - hSSR / 2 + 14, f"({SSR / SST * 100:.1f}%, 설명됨)", "f-ac", 12, anchor="start")
s.text(BX + BW + 14, BT + (H - hSSR) / 2 - 4, f"SSE = {SSE:.1f}", "f-bd", 13, anchor="start", weight="600")
s.text(BX + BW + 14, BT + (H - hSSR) / 2 + 14, f"({SSE / SST * 100:.1f}%, 잔차)", "f-bd", 12, anchor="start")
s.text(BX + BW / 2, BB + 22, f"R² = {SSR / SST:.3f}", "f-fg", 13, weight="600")
s.save(os.path.join(OUT, "06-ss-split.svg"))

# ---------------------------------------------------------------- 3. 의미 없는 변수를 넣을 때
imp = x
ndvi = np.array([float(r["ndvi"]) for r in rows])
elev = np.array([float(r["elev_m"]) for r in rows])
coast = np.array([float(r["coast_km"]) for r in rows])
X0 = np.column_stack([np.ones(n), imp, ndvi, elev, coast])
K, RUNS = 20, 2000


def r2(Xm):
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    res = y - Xm @ beta
    rsq = 1 - res @ res / SST
    p = Xm.shape[1] - 1
    return rsq, 1 - (1 - rsq) * (n - 1) / (n - p - 1)


R2 = np.zeros((RUNS, K + 1))
AD = np.zeros((RUNS, K + 1))
for sd in range(RUNS):
    Z = np.random.default_rng(1000 + sd).normal(size=(n, K))
    for k in range(K + 1):
        R2[sd, k], AD[sd, k] = r2(np.column_stack([X0, Z[:, :k]]))
mR, mA = R2.mean(0), AD.mean(0)
loA, hiA = np.percentile(AD, [10, 90], axis=0)
loR, hiR = np.percentile(R2, [10, 90], axis=0)
print("R2 mean", mR[[0, 1, 5, 10, 20]].round(3), "adj mean", mA[[0, 1, 5, 10, 20]].round(3))
print("adj 10-90% at 20", loA[20].round(3), hiA[20].round(3), "R2 10-90% at 20", loR[20].round(3), hiR[20].round(3))

s = Svg(680, 290, "기본 모형에 난수로 만든 의미 없는 변수를 0개에서 20개까지 넣을 때 R²와 수정 R²의 평균과 10~90% 범위")
L, R, T, B = 70, 520, 20, 240
YLO, YHI = 0.66, 0.86
sx = lambda k: L + (R - L) * k / K
sy = lambda v: B - (B - T) * (v - YLO) / (YHI - YLO)
s.line(L, B, R, B, "s-mu", 1.2)
s.line(L, T, L, B, "s-mu", 1.2)
for k in range(0, K + 1, 5):
    s.line(sx(k), B, sx(k), B + 5, "s-mu", 1.2)
    s.text(sx(k), B + 19, f"{k}", "f-mu", 11)
for v in np.arange(0.68, 0.861, 0.04):
    s.line(L - 5, sy(v), L, sy(v), "s-mu", 1.2)
    s.text(L - 9, sy(v) + 4, f"{v:.2f}", "f-mu", 11, anchor="end")
s.text((L + R) / 2, B + 40, "추가한 의미 없는 변수 수", "f-fg", 12)
for lo, hi, cls in ((loA, hiA, "s-mu"), (loR, hiR, "s-ac")):
    for band in (lo, hi):
        s.add('<path d="M ' + " L ".join(f"{sx(k):.1f} {sy(band[k]):.1f}" for k in range(K + 1))
              + f'" class="{cls}" stroke-width="1" stroke-dasharray="4 3" fill="none"/>')
s.path("M " + " L ".join(f"{sx(k):.1f} {sy(mR[k]):.1f}" for k in range(K + 1)), "s-ac", 2.4)
s.path("M " + " L ".join(f"{sx(k):.1f} {sy(mA[k]):.1f}" for k in range(K + 1)), "s-fg", 2.4)
s.text(R + 10, sy(mR[K]) + 4, f"R² 평균 {mR[K]:.3f}", "f-ac", 12, anchor="start", weight="600")
s.text(R + 10, sy(mA[K]) + 4, f"수정 R² 평균 {mA[K]:.3f}", "f-fg", 12, anchor="start", weight="600")
s.text(R + 10, sy(mA[K]) + 22, "(점선: 10~90% 범위)", "f-mu", 11, anchor="start")
s.save(os.path.join(OUT, "06-adj-r2.svg"))
