"""38강 그림과 검산: 클래스별 PR 곡선(F1 최대점·최대 재현율), 점수 임계값에 따른 정밀도·재현율·F1

자료: content/lessons/data/make_part6.py (항만·연안 타일 24장, 가상 탐지기)
결과: 기본 탐지 결과(타일 그대로, 클래스별 NMS IoU 0.5), 같은 클래스·IoU 0.5 이상·1:1 짝짓기(점수 높은 탐지부터).
      곡선은 data/run_part6.py의 exp_pr과 같은 방식(max_dets 1000). 본문 표의 값은 part6_runs.json의 'pr'
아래 print는 본문에 쓴 추가 수치(중복 오탐 수, IoU 0.3에서의 최대 재현율, 타일 종류별 오탐, FPPI 상한 운영점)의 검산용
"""
import os, sys
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from svglib import Svg
from make_part6 import tiles, detect
from part6_eval import nms_dets, pr_points, operating_point, iou_matrix, _match

OUT = os.path.join(HERE, "..", "fig")
T = tiles()
RAW = [detect(t["objs"], t["clutter"], 1.0, seed=t["image_id"]) for t in T]
BASE = [(t["objs"], nms_dets(r, 0.5)) for t, r in zip(T, RAW)]
NAMES = ["선박", "소형선박", "차량"]


def curve(c, thr=0.5):
    S, tp, n = pr_points(BASE, c, thr, max_dets=1000)
    tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
    P, R = tpc / (tpc + fpc), tpc / n
    F = 2 * P * R / np.maximum(P + R, 1e-12)
    return S, tp, n, P, R, F, fpc


# ---------------------------------------------------------------- 검산
for c in range(3):
    S, tp, n, P, R, F, fpc = curve(c)
    i = int(F.argmax())
    print(f"[{NAMES[c]}] 정답 {n} 탐지 {len(S)} F1최대 {F[i]:.3f} 점수 {S[i]:.3f} P {P[i]:.3f} R {R[i]:.3f} 최대재현율 {R[-1]:.3f}"
          f" | IoU0.3 최대재현율 {curve(c, 0.3)[4][-1]:.3f}"
          f" | 점수0.7이상 탐지 {(S >= 0.7).sum()} TP점수 중앙값 {np.median(S[tp]):.3f}")
    for cap in (1.0, 2.0):
        k = np.where(fpc / len(T) <= cap)[0][-1]
        print(f"   FPPI<={cap}: 점수 {S[k]:.3f}", operating_point(BASE, c, S[k]))
    # NMS 뒤에도 남은 중복: 오탐인데 같은 클래스 정답(이미 다른 탐지와 짝지음)과 IoU 0.5 이상
    dup = nfp = 0
    for g, d in BASE:
        gc = [x for x in g if x["cls"] == c]
        dc = sorted([x for x in d if x["cls"] == c], key=lambda x: -x["score"])
        _, t, _, _ = _match(gc, dc, 0.5)
        if gc and dc:
            M = iou_matrix([x["hbb"] for x in dc], [x["hbb"] for x in gc])
            dup += int(((M.max(1) >= 0.5) & ~t).sum())
        nfp += int((~t).sum())
    print(f"   오탐 {nfp} 가운데 중복 {dup}")
    fk = {}
    for (g, d), tt in zip(BASE, T):
        _, t, _, _ = _match([x for x in g if x["cls"] == c], [x for x in d if x["cls"] == c and x["score"] >= 0.4], 0.5)
        fk[tt["kind"]] = fk.get(tt["kind"], 0) + int((~t).sum())
    print("   점수 0.4 오탐(타일 종류별)", fk)
print("공통 0.5:", {NAMES[c]: operating_point(BASE, c, 0.5) for c in range(3)})
print("타일 넓이 km2", 256 * 256 / 1e6, "소형선박 0.4 오탐 km2당", round(118 / (24 * 0.065536), 1))

# ---------------------------------------------------------------- 1. 클래스별 PR 곡선
s = Svg(640, 400, "세 클래스의 정밀도-재현율 곡선. 원은 F1 최대점, 사각형은 곡선 끝(최대 재현율)")
X0, X1, YT, YB = 70, 600, 30, 340
sx = lambda v: X0 + (X1 - X0) * v
sy = lambda v: YB - (YB - YT) * v
s.line(X0, YB, X1, YB, "s-mu", 1.2)
s.line(X0, YT, X0, YB, "s-mu", 1.2)
for v in (0, 0.2, 0.4, 0.6, 0.8, 1):
    s.line(sx(v), YB, sx(v), YB + 4, "s-mu", 1.2)
    s.text(sx(v), YB + 18, f"{v:g}", "f-mu", 12)
    s.line(X0 - 4, sy(v), X0, sy(v), "s-mu", 1.2)
    s.text(X0 - 8, sy(v) + 4, f"{v:g}", "f-mu", 12, anchor="end")
s.text((X0 + X1) / 2, YB + 40, "재현율", "f-mu", 12)
s.text(X0, YT - 12, "정밀도", "f-mu", 12, anchor="start")
LC = {0: ("s-ok", "f-ok"), 1: ("s-ac", "f-ac"), 2: ("s-bd", "f-bd")}
for c in range(3):
    S, tp, n, P, R, F, _ = curve(c)
    ln, fl = LC[c]
    s.path("M " + " L ".join(f"{sx(a):.1f} {sy(b):.1f}" for a, b in zip(R, P)), ln, 2.0)
    i = int(F.argmax())
    s.circle(sx(R[i]), sy(P[i]), 5.5, fl)
    s.rect(sx(R[-1]) - 4.5, sy(P[-1]) - 4.5, 9, 9, fl, 0)
# 곡선 끝 옆 이름표(곡선과 겹치지 않는 자리)
s.text(sx(0.818) + 10, sy(0.305) + 4, "선박 0.82", "f-ok", 12, anchor="start", weight="600")
s.text(sx(0.949) - 2, sy(0.551) + 22, "소형선박 0.95", "f-ac", 12, anchor="middle", weight="600")
s.text(sx(0.704) + 2, sy(0.708) + 24, "차량 0.70", "f-bd", 12, anchor="middle", weight="600")
# 범례(왼쪽 아래 빈 곳)
lx, ly = sx(0.04), sy(0.30)
for k, (txt, ln, fl) in enumerate([("선박 (정답 22)", "s-ok", "f-ok"), ("소형선박 (311)", "s-ac", "f-ac"), ("차량 (901)", "s-bd", "f-bd")]):
    s.line(lx, ly + 20 * k, lx + 26, ly + 20 * k, ln, 2.0)
    s.text(lx + 34, ly + 20 * k + 4, txt, fl, 12, anchor="start", weight="600")
s.circle(lx + 13, ly + 66, 5.5, "f-mu")
s.text(lx + 34, ly + 70, "F1 최대점", "f-mu", 12, anchor="start")
s.rect(lx + 8.5, ly + 81.5, 9, 9, "f-mu", 0)
s.text(lx + 34, ly + 90, "곡선 끝 = 최대 재현율", "f-mu", 12, anchor="start")
s.save(os.path.join(OUT, "38-pr.svg"))

# ---------------------------------------------------------------- 2. 점수 임계값에 따른 P·R·F1
s = Svg(680, 330, "소형선박과 차량에서 점수 임계값을 바꿀 때의 정밀도, 재현율, F1")
W, YT, YB = 250, 40, 260


def panel(x0, title):
    sx = lambda v: x0 + W * v
    sy = lambda v: YB - (YB - YT) * v
    s.line(x0, YB, x0 + W, YB, "s-mu", 1.2)
    s.line(x0, YT, x0, YB, "s-mu", 1.2)
    for v in (0, 0.5, 1):
        s.line(sx(v), YB, sx(v), YB + 4, "s-mu", 1.2)
        s.text(sx(v), YB + 18, f"{v:g}", "f-mu", 12)
        s.line(x0 - 4, sy(v), x0, sy(v), "s-mu", 1.2)
        s.text(x0 - 8, sy(v) + 4, f"{v:g}", "f-mu", 12, anchor="end")
    s.text(x0 + W / 2, YB + 40, "점수 임계값", "f-mu", 12)
    s.text(x0 + W / 2, 18, title, "f-fg", 13, weight="600")
    return sx, sy


for x0, c in ((70, 1), (400, 2)):
    S, tp, n, P, R, F, _ = curve(c)
    sx, sy = panel(x0, NAMES[c])
    grid = np.arange(0.0, 1.0001, 0.005)
    pts = []
    for g in grid:
        k = int((S >= g).sum())
        if k:
            pts.append((g, P[k - 1], R[k - 1], F[k - 1]))
    pts = np.array(pts)
    for j, ln, wd in ((1, "s-ac", 2.0), (2, "s-bd", 2.0), (3, "s-fg", 2.4)):
        s.path("M " + " L ".join(f"{sx(a):.1f} {sy(b):.1f}" for a, b in zip(pts[:, 0], pts[:, j])), ln, wd)
    i = int(F.argmax())
    s.line(sx(S[i]), YT, sx(S[i]), YB, "s-mu", 1.2, "4 3")
    s.circle(sx(S[i]), sy(F[i]), 4.5, "f-fg")
    s.line(sx(0.5), YT + 30, sx(0.5), YB, "s-mu", 1.0, "2 3")
    if c == 1:
        s.text(sx(S[i]) - 6, sy(0.42), f"F1 최대 (점수 {S[i]:.2f})", "f-fg", 12, anchor="end")
    else:
        s.text(sx(S[i]) + 6, YT + 12, f"F1 최대 (점수 {S[i]:.2f})", "f-fg", 12, anchor="start")
        s.text(sx(S.max()) + 6, sy(0.02) - 4, "점수 0.7 넘는", "f-mu", 11, anchor="start")
        s.text(sx(S.max()) + 6, sy(0.02) + 10, "탐지 거의 없음", "f-mu", 11, anchor="start")
# 범례(오른쪽 패널에서 곡선이 끝난 점수 0.71 오른쪽 위)
ly = 66
for k, (txt, ln, fl) in enumerate([("정밀도", "s-ac", "f-ac"), ("재현율", "s-bd", "f-bd"), ("F1", "s-fg", "f-fg")]):
    s.line(sx(0.75), ly + 20 * k, sx(0.75) + 24, ly + 20 * k, ln, 2.2)
    s.text(sx(0.75) + 30, ly + 20 * k + 4, txt, fl, 12, anchor="start", weight="600")
s.save(os.path.join(OUT, "38-thr.svg"))
