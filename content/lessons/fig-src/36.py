"""36강 그림과 본문 수치: IoU 기반 할당 도식(SVG), 정답 상자 모양과 앵커(SVG), 앵커 프리 격자점 도식(SVG)

- 본문 수치는 data/part6_runs.json의 'anchor'(run_part6.py exp_anchor)에서 가져오고, 여기서 다시 계산해 같은지 확인함
- IoU 기반 할당 도식의 IoU, TAL·SimOTA 계산 예의 검산도 여기서 출력함
"""
import math, os, sys, json
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg
from run_part6 import T, YOLO3, shape_iou, kmeans_anchors

OUT = os.path.join(HERE, "..", "fig")
J = json.load(open(os.path.join(DATA, "part6_runs.json"), encoding="utf-8"))["anchor"]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


# ---------------------------------------------------------------- 0. JSON 값 다시 계산해 확인
wh = np.array([(o["hbb"][2] - o["hbb"][0], o["hbb"][3] - o["hbb"][1]) for t in T for o in t["objs"]])
cls = np.array([o["cls"] for t in T for o in t["objs"]])
best3 = shape_iou(wh, YOLO3).max(1)
K = kmeans_anchors(wh)
bestk = shape_iou(wh, K).max(1)
for c, name in enumerate(["선박", "소형선박", "차량"]):
    m = cls == c
    print(f"{name} n={m.sum()}  YOLOv3 앵커 최대 IoU 중앙값 {np.median(best3[m]):.4f}, <0.5 {np.mean(best3[m] < 0.5):.4f}"
          f"  | k-평균 중앙값 {np.median(bestk[m]):.4f}, <0.5 {np.mean(bestk[m] < 0.5):.4f}")
    assert abs(np.median(best3[m]) - J["yolo3_best_iou"][str(c)]["median"]) < 1e-4
    assert abs(np.median(bestk[m]) - J["kmeans_best_iou"][str(c)]["median"]) < 1e-4
print("k-평균 앵커:", np.round(K, 1).tolist())
print("YOLOv3 앵커 넓이 범위:", min(a * b for a, b in YOLO3), max(a * b for a, b in YOLO3))
print("정답 상자 폭·높이 범위(클래스별, 화소):",
      {c: (np.round(wh[cls == c].min(0), 1).tolist(), np.round(wh[cls == c].max(0), 1).tolist()) for c in range(3)})
for s in (8, 16, 32):
    print("격자점 0개 비율, 스트라이드", s, {c: J["anchor_free_points"][str(s)][str(c)] for c in range(3)})

# TAL·SimOTA 계산 예 검산
ex = J["assign_example"]
s_ = np.array([c["score"] for c in ex["cand"]])
u_ = np.array([c["iou"] for c in ex["cand"]])
tal = s_ ** 1.0 * u_ ** 6.0
cost = -np.log(s_) + 3.0 * -np.log(u_)
print("후보 수", len(s_), "TAL", np.round(tal, 4).tolist(), "상위", np.argsort(-tal)[:3].tolist())
print("SimOTA 비용", np.round(cost, 4).tolist(), "IoU 합", round(u_.sum(), 4), "동적 k", int(np.sort(u_)[::-1][:10].sum()),
      "고른 후보", np.argsort(cost)[:int(u_.sum())].tolist())
print("u^6:", np.round(u_ ** 6, 3).tolist())
g = ex["gt_hbb"]
print("정답 상자 크기(화소)", round(g[2] - g[0], 1), round(g[3] - g[1], 1))

# ---------------------------------------------------------------- 1. IoU 기반 할당 도식
ST, NX, NY = 16, 8, 6
AN = [(32 * 2 ** 0.5, 32 / 2 ** 0.5), (32.0, 32.0), (32 / 2 ** 0.5, 32 * 2 ** 0.5)]   # 넓이 32², 2:1·1:1·1:2
GT = (38.0, 33.0, 82.0, 55.0)                                                      # 44×22화소
POS, NEG = 0.5, 0.4


def anchor_box(i, j, k):
    px, py = ST / 2 + ST * i, ST / 2 + ST * j
    a, b = AN[k]
    return (px - a / 2, py - b / 2, px + a / 2, py + b / 2)


allv = np.array([[[iou(anchor_box(i, j, k), GT) for k in range(3)] for i in range(NX)] for j in range(NY)])
print("도식: 앵커", allv.size, "개, 양성", int((allv >= POS).sum()), "무시", int(((allv >= NEG) & (allv < POS)).sum()))
print("칸(3,2):", np.round(allv[2, 3], 3).tolist(), " 칸(4,2):", np.round(allv[2, 4], 3).tolist())

s = Svg(760, 330, "가로 8칸·세로 6칸(스트라이드 16) 격자 위의 정답 상자와, 한 칸에 놓인 앵커 3개. 칸 중심 점은 그 칸 앵커의 "
                  "최대 IoU에 따라 양성(0.5 이상)·무시(0.4~0.5)·음성(0.4 미만)으로 칠함. 앵커 144개 가운데 양성 2개, 무시 2개")
SC, OX, OY = 3.2, 24, 12
X = lambda x: OX + x * SC
Y = lambda y: OY + y * SC
s.rect(X(0), Y(0), NX * ST * SC, NY * ST * SC, "s-mu f-bg", 1.0)
for i in range(1, NX):
    s.line(X(i * ST), Y(0), X(i * ST), Y(NY * ST), "s-mu", 0.5)
for j in range(1, NY):
    s.line(X(0), Y(j * ST), X(NX * ST), Y(j * ST), "s-mu", 0.5)
# 앵커(칸 (3,2))
sty = [("s-ac", None, 2.0), ("s-ac", "6 4", 1.6), ("s-mu", "3 3", 1.6)]
for k in range(3):
    b = anchor_box(3, 2, k)
    cl, da, wd = sty[k]
    s.add(f'<rect x="{X(b[0]):.1f}" y="{Y(b[1]):.1f}" width="{(b[2] - b[0]) * SC:.1f}" height="{(b[3] - b[1]) * SC:.1f}" '
          f'class="{cl}" fill="none" stroke-width="{wd}"' + (f' stroke-dasharray="{da}"' if da else "") + "/>")
# 정답 상자
s.add(f'<rect x="{X(GT[0]):.1f}" y="{Y(GT[1]):.1f}" width="{(GT[2] - GT[0]) * SC:.1f}" height="{(GT[3] - GT[1]) * SC:.1f}" '
      f'class="s-bd" fill="none" stroke-width="2.6"/>')
# 칸 중심 점
for j in range(NY):
    for i in range(NX):
        v = allv[j, i].max()
        cx, cy = X(ST / 2 + ST * i), Y(ST / 2 + ST * j)
        if v >= POS:
            s.circle(cx, cy, 5, "f-ac")
        elif v >= NEG:
            s.circle(cx, cy, 5, "f-bd")
        else:
            s.circle(cx, cy, 2.2, "f-mu")
# 오른쪽: 기준선과 범례
LX = 470
s.text(LX, 30, "IoU 기준 (RetinaNet식)", "f-fg", 13, anchor="start", weight="600")
NL0, NL1, NLY = LX + 6, LX + 266, 70
xv = lambda v: NL0 + (NL1 - NL0) * v
s.rect(xv(0), NLY - 6, xv(NEG) - xv(0), 12, "s-mu f-sf", 1.0)
s.rect(xv(NEG), NLY - 6, xv(POS) - xv(NEG), 12, "s-bd f-bds", 1.0)
s.rect(xv(POS), NLY - 6, xv(1) - xv(POS), 12, "s-ac f-acs", 1.0)
for v, lab in [(0, "0"), (NEG, "0.4"), (POS, "0.5"), (1, "1")]:
    s.text(xv(v), NLY + 22, lab, "f-mu", 11)
s.text((xv(0) + xv(NEG)) / 2, NLY - 12, "음성", "f-mu", 11)
s.text((xv(NEG) + xv(POS)) / 2, NLY - 12, "무시", "f-bd", 11)
s.text((xv(POS) + xv(1)) / 2, NLY - 12, "양성", "f-ac", 11)
rows = [("line", sty[0], f"가로형 앵커  {allv[2, 3, 0]:.2f} → 양성"),
        ("line", sty[1], f"정사각 앵커  {allv[2, 3, 1]:.2f} → 양성"),
        ("line", sty[2], f"세로형 앵커  {allv[2, 3, 2]:.2f} → 음성"),
        ("dot", "f-bd", f"오른쪽 칸 최대  {allv[2, 4].max():.2f} → 무시"),
        ("gt", None, "정답 상자 44×22화소")]
for n, (kind, st, lab) in enumerate(rows):
    yy = 122 + n * 30
    if kind == "line":
        cl, da, wd = st
        s.line(LX + 4, yy - 4, LX + 34, yy - 4, cl, wd + 0.4, da)
    elif kind == "dot":
        s.circle(LX + 19, yy - 4, 5, st)
    else:
        s.line(LX + 4, yy - 4, LX + 34, yy - 4, "s-bd", 2.6)
    s.text(LX + 44, yy, lab, "f-fg", 12, anchor="start")
s.text(LX, 290, f"앵커 {allv.size}개 중 양성 {int((allv >= POS).sum())} · 무시 "
                f"{int(((allv >= NEG) & (allv < POS)).sum())} · 나머지 음성", "f-mu", 12, anchor="start")
s.save(os.path.join(OUT, "36-iou-assign.svg"))

# ---------------------------------------------------------------- 2. 정답 상자 모양과 앵커(로그 축)
s = Svg(760, 380, "가로축은 정답 수평 박스의 폭, 세로축은 높이(화소, 로그 눈금). 차량은 10화소 이하, 소형선박은 대개 40화소 이하, "
                  "선박은 긴 변이 80~260화소. YOLOv3 기본 앵커 9개는 10~373화소에 퍼져 있고, 우리 자료로 "
                  "구한 k-평균 앵커 9개는 차량·소형선박 쪽(35화소 이하)에 몰림")
PX0, PY0, PW = 70, 20, 320
LO, HI = math.log10(3), math.log10(450)
fx = lambda v: PX0 + (math.log10(v) - LO) / (HI - LO) * PW
fy = lambda v: PY0 + PW - (math.log10(v) - LO) / (HI - LO) * PW
s.rect(PX0, PY0, PW, PW, "s-mu f-bg", 1.0)
for v in (5, 10, 30, 100, 300):
    s.line(fx(v), PY0, fx(v), PY0 + PW, "s-mu", 0.4, "2 3")
    s.line(PX0, fy(v), PX0 + PW, fy(v), "s-mu", 0.4, "2 3")
    s.text(fx(v), PY0 + PW + 16, str(v), "f-mu", 11)
    s.text(PX0 - 6, fy(v) + 4, str(v), "f-mu", 11, anchor="end")
s.text(PX0 + PW / 2, PY0 + PW + 36, "폭(화소)", "f-fg", 12)
s.text(PX0 - 40, PY0 + PW / 2 - 6, "높이", "f-fg", 12)
s.text(PX0 - 40, PY0 + PW / 2 + 9, "(화소)", "f-fg", 12)
CC = {2: "f-mu", 1: "f-ac", 0: "f-bd"}
for c in (2, 1, 0):
    for w, h in wh[cls == c]:
        s.circle(fx(w), fy(h), 1.7 if c else 2.6, CC[c])
for a, b in YOLO3:
    s.add(f'<rect x="{fx(a) - 5:.1f}" y="{fy(b) - 5:.1f}" width="10" height="10" class="s-fg" fill="none" stroke-width="1.6"/>')
for a, b in K:
    xx, yy = fx(a), fy(b)
    s.path(f"M {xx:.1f} {yy - 6:.1f} L {xx + 6:.1f} {yy:.1f} L {xx:.1f} {yy + 6:.1f} L {xx - 6:.1f} {yy:.1f} Z", "s-ok", 1.8)
LX = 440
items = [("dot", "f-mu", 1.9, f"차량 {int((cls == 2).sum())}대"), ("dot", "f-ac", 1.9, f"소형선박 {int((cls == 1).sum())}척"),
         ("dot", "f-bd", 2.8, f"선박 {int((cls == 0).sum())}척"), ("sq", None, None, "YOLOv3 기본 앵커 9개"),
         ("dia", None, None, "k-평균 앵커 9개(우리 자료)")]
for n, (kind, cl, r, lab) in enumerate(items):
    yy = 60 + n * 30
    if kind == "dot":
        s.circle(LX + 8, yy - 4, r + 1.2, cl)
    elif kind == "sq":
        s.add(f'<rect x="{LX + 3:.1f}" y="{yy - 9:.1f}" width="10" height="10" class="s-fg" fill="none" stroke-width="1.6"/>')
    else:
        s.path(f"M {LX + 8} {yy - 10} L {LX + 14} {yy - 4} L {LX + 8} {yy + 2} L {LX + 2} {yy - 4} Z", "s-ok", 1.8)
    s.text(LX + 24, yy, lab, "f-fg", 12, anchor="start")
s.text(LX, 240, "앵커와 모양 IoU 0.5 미만", "f-fg", 12, anchor="start", weight="600")
s.text(LX, 262, "YOLOv3 앵커: 선박 77% · 차량 59%", "f-mu", 12, anchor="start")
s.text(LX, 282, "k-평균 앵커: 선박 100% · 차량 0%", "f-mu", 12, anchor="start")
s.save(os.path.join(OUT, "36-shapes.svg"))

# ---------------------------------------------------------------- 3. 앵커 프리 격자점 도식
# 설명용 배치(48×32화소 구역): A 가로 차량, B 가로 차량(격자와 어긋남), C 45° 차량의 HBB
BOX = {"A": (2.0, 10.0, 11.0, 14.0), "B": (14.0, 21.0, 23.0, 25.0), "C": (30.0, 3.0, 39.6, 12.6)}
WW, HH = 48, 32


def pts_in(b, st):
    g = np.arange(st / 2, max(WW, HH), st)
    return [(x, y) for x in g if b[0] < x < b[2] for y in g if b[1] < y < b[3]]


s = Svg(720, 262, "0.5 m 영상의 승용차 세 대 상자(A·B는 가로로 선 9×4화소, C는 45°로 놓인 차의 수평 상자)를 스트라이드 8과 16의 격자 위에 놓고 상자 안에 들어오는 "
                  "격자점(칸 중심)을 표시함. 스트라이드 8에서는 1·0·2개, 스트라이드 16에서는 모두 0개")
SC2 = 6.4
for p, st in enumerate((8, 16)):
    ox, oy = 30 + p * 360, 44
    PXf = lambda x: ox + x * SC2
    PYf = lambda y: oy + y * SC2
    s.text(ox + WW * SC2 / 2, 24, f"스트라이드 {st}", "f-fg", 13, weight="600")
    s.rect(PXf(0), PYf(0), WW * SC2, HH * SC2, "s-mu f-bg", 1.0)
    for x in range(st, WW, st):
        s.line(PXf(x), PYf(0), PXf(x), PYf(HH), "s-mu", 0.5, "3 3")
    for y in range(st, HH, st):
        s.line(PXf(0), PYf(y), PXf(WW), PYf(y), "s-mu", 0.5, "3 3")
    # C의 실제 차량(45°) 윤곽
    cx, cy = (BOX["C"][0] + BOX["C"][2]) / 2, (BOX["C"][1] + BOX["C"][3]) / 2
    L, Wd = 9.0, 4.6
    u = np.array([1, 1]) / 2 ** 0.5 * L / 2
    v = np.array([-1, 1]) / 2 ** 0.5 * Wd / 2
    c0 = np.array([cx, cy])
    poly = [c0 - u - v, c0 + u - v, c0 + u + v, c0 - u + v]
    s.path("M " + " L ".join(f"{PXf(q[0]):.1f} {PYf(q[1]):.1f}" for q in poly) + " Z", "s-mu f-sf", 1.0)
    inside = set()
    for name, b in BOX.items():
        s.add(f'<rect x="{PXf(b[0]):.1f}" y="{PYf(b[1]):.1f}" width="{(b[2] - b[0]) * SC2:.1f}" '
              f'height="{(b[3] - b[1]) * SC2:.1f}" class="s-bd" fill="none" stroke-width="2"/>')
        q = pts_in(b, st)
        inside |= set(q)
        if name == "B":
            s.text(PXf(b[2]) + 22, PYf((b[1] + b[3]) / 2) + 4, f"{name}: {len(q)}개", "f-bd", 12, anchor="start", weight="600")
        else:
            s.text(PXf(b[0]) + (b[2] - b[0]) * SC2 / 2, PYf(b[3]) + 16, f"{name}: {len(q)}개", "f-bd", 12, weight="600")
        print(f"스트라이드 {st} 상자 {name}: 격자점 {len(q)}개")
    g = np.arange(st / 2, WW, st)
    gy = np.arange(st / 2, HH, st)
    for x in g:
        for y in gy:
            if (x, y) in inside:
                s.circle(PXf(x), PYf(y), 4.5, "f-ac")
            else:
                s.circle(PXf(x), PYf(y), 2.2, "f-mu")
s.save(os.path.join(OUT, "36-gridpoints.svg"))
