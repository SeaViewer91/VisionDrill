"""42강 그림과 본문 수치: 칼만 예측–갱신 도식(SVG), 추적 결과 궤적(PNG), 추적기별 MOTA·IDF1·HOTA 막대(SVG)

- 추적기·HOTA는 data/run_part6.py의 KF, run_tracker, hota를 그대로 씀. 지표 값은 part6_runs.json의 mot를 읽음
- JSON에 없는 본문 수치(글린트 구역 정답 상자 수, 낮은 점수 탐지의 구성, 엇갈리는 쌍의 IoU, 정상 상태 칼만 이득,
  비용 행렬 예, 정답 배마다 바뀐 ID)도 여기서 계산해 출력함
"""
import json, math, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg
from make_part6 import sequence, obb_to_hbb, TILE, render
from run_part6 import run_tracker, iou_matrix
from scipy.optimize import linear_sum_assignment

OUT = os.path.join(HERE, "..", "fig")
MOT = json.load(open(os.path.join(DATA, "part6_runs.json"), encoding="utf-8"))["mot"]


def arrow(s, x1, y1, x2, y2, cls="s-fg", fcls="f-fg", width=1.4, head=6, dash=None):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    bx, by = x2 - ux * head, y2 - uy * head
    s.line(x1, y1, bx, by, cls, width, dash)
    px, py = -uy * head * 0.55, ux * head * 0.55
    s.path(f"M {x2:.1f} {y2:.1f} L {bx + px:.1f} {by + py:.1f} L {bx - px:.1f} {by - py:.1f} Z", fcls, 0)


# ------------------------------------------------------------------ 본문 수치
Q = sequence()
n, gt, det = Q["n_frames"], Q["gt"], Q["det"]
gtb = [(f, t, *np.clip(obb_to_hbb((cx, cy, w, h, th)), 0, TILE).tolist()) for f, t, c, cx, cy, w, h, th in gt]
print("정답 상자", len(gtb), "탐지", len(det), "점수 0.5 이상", sum(d[2] >= 0.5 for d in det))
print("글린트 구역(프레임 50~65, x 200~330) 정답 상자", sum(1 for g in gt if 50 <= g[0] <= 65 and 200 <= g[3] <= 330),
      "지나는 배", sorted(set(g[1] for g in gt if 50 <= g[0] <= 65 and 200 <= g[3] <= 330)))
cnt = defaultdict(int)
for d in det:
    g = [x for x in gtb if x[0] == d[0]]
    m = iou_matrix([d[3:7]], [x[2:6] for x in g]).max() if g else 0
    band = "high" if d[2] >= 0.5 else "low" if d[2] >= 0.1 else "vlow"
    cnt[(band, "boat" if m >= 0.5 else "other")] += 1
print("탐지 구성(점수 구간, 배/그 밖)", dict(cnt))
for pair in ((3, 4), (5, 6)):
    best = max((float(iou_matrix([a[2:6]], [b[2:6]])[0, 0]), a[0]) for a in gtb for b in gtb
               if a[0] == b[0] and a[1] == pair[0] and b[1] == pair[1])
    print(f"엇갈리는 배 {pair}: 정답 상자끼리 가장 큰 IoU {best[0]:.3f} (프레임 {best[1]})")
# 정상 상태 칼만 이득(위치·속도 1차원, run_part6.KF와 같은 Q·R)
F = np.array([[1, 1], [0, 1.]]); H = np.array([[1, 0.]]); Qm = np.diag([1, 2.]); P = np.diag([4, 100.])
for _ in range(200):
    P = F @ P @ F.T + Qm
    K = P @ H.T / (H @ P @ H.T + 4.0)
    P = (np.eye(2) - K @ H) @ P
print("정상 상태 칼만 이득(위치)", round(float(K[0, 0]), 3), "예: 예측 100, 탐지 104 → 갱신", round(100 + K[0, 0] * 4, 1))
Pp, sd = P.copy(), []
for _ in range(16):
    Pp = F @ Pp @ F.T + Qm; sd.append(math.sqrt(Pp[0, 0]))
print("탐지 없이 예측만: 위치 표준편차 1·16프레임 뒤", round(sd[0], 1), round(sd[-1], 1))
# 비용 행렬 예(1 − IoU)
TRK = {"A": (100, 100, 124, 110), "B": (112, 106, 136, 116)}
DET = {"1": (102, 101, 126, 111), "2": (113, 108, 137, 118), "3": (300, 200, 316, 206)}
C = 1 - iou_matrix(list(TRK.values()), list(DET.values()))
print("비용 행렬 1-IoU\n", np.round(C, 2))
r, c = linear_sum_assignment(C)
print("헝가리안 짝", [(list(TRK)[i], list(DET)[j], round(float(C[i, j]), 2)) for i, j in zip(r, c)])

# 정답 배마다 바뀐 추적 ID
CFG = {"iou_only": dict(mode="iou"), "sort": dict(mode="sort"), "sort_maxage20": dict(mode="sort", max_age=20),
       "byte": dict(mode="byte")}
TR = {k: run_tracker(det, n, **kw) for k, kw in CFG.items()}
for k, tr in TR.items():
    seq = defaultdict(list)
    for f in range(n):
        g = [x for x in gtb if x[0] == f]; t = [x for x in tr if x[0] == f]
        if not g or not t:
            continue
        M = iou_matrix([x[2:6] for x in g], [x[2:6] for x in t])
        for i, j in zip(*linear_sum_assignment(-M)):
            if M[i, j] >= 0.5:
                seq[g[i][1]].append((f, t[j][1]))
    ch = {}
    for gid, s in seq.items():
        prev, lst = None, []
        for f, tid in s:
            if tid != prev:
                lst.append(f); prev = tid
        if len(lst) > 1:
            ch[gid] = lst[1:]
    print(k, "ID 수", len(set(x[1] for x in tr)), "| JSON", MOT[k]["n_track_ids"], "| ID 바뀐 프레임(정답 배별)", ch)


# 공식 HOTA(TrackEval metrics/hota.py)의 짝짓기: 프레임마다 (전역 정렬 점수 × IoU)로 헝가리안 → α마다 IoU ≥ α만 남김.
# run_part6.hota는 IoU만으로 짝지음. 본문 "이 시퀀스에서는 같았음"의 근거(검증 때 trackeval 1.3.0으로도 같은 값 확인)
def hota_trackeval(gt_, tr_):
    gi = {g: i for i, g in enumerate(sorted({x[1] for x in gt_}))}
    ti = {t: i for i, t in enumerate(sorted({x[1] for x in tr_}))}
    G, T = defaultdict(list), defaultdict(list)
    for x in gt_: G[x[0]].append(x)
    for x in tr_: T[x[0]].append(x)
    pm, gc, tc, fr = np.zeros((len(gi), len(ti))), np.zeros((len(gi), 1)), np.zeros((1, len(ti))), []
    for f in range(n):
        a = np.array([gi[x[1]] for x in G[f]], int); b = np.array([ti[x[1]] for x in T[f]], int)
        sim = iou_matrix([x[2:6] for x in G[f]], [x[2:6] for x in T[f]]) if len(a) and len(b) else np.zeros((len(a), len(b)))
        if len(a) and len(b):
            den = sim.sum(0)[None] + sim.sum(1)[:, None] - sim
            pm[a[:, None], b[None, :]] += np.where(den > 1e-9, sim / np.maximum(den, 1e-9), 0)
        gc[a] += 1; tc[0, b] += 1; fr.append((a, b, sim))
    gas = pm / (gc + tc - pm)
    alphas = np.arange(0.05, 0.99, 0.05)
    TP, FN, FP = np.zeros(19), np.zeros(19), np.zeros(19)
    mc = [np.zeros_like(pm) for _ in alphas]
    for a, b, sim in fr:
        if not len(a) or not len(b):
            FN += len(a); FP += len(b); continue
        r, c = linear_sum_assignment(-(gas[a[:, None], b[None, :]] * sim))
        for k, al in enumerate(alphas):
            m = sim[r, c] >= al - np.finfo(float).eps
            TP[k] += m.sum(); FN[k] += len(a) - m.sum(); FP[k] += len(b) - m.sum()
            mc[k][a[r[m]], b[c[m]]] += 1
    ass = np.array([(m * m / np.maximum(1, gc + tc - m)).sum() / max(1, tp) for m, tp in zip(mc, TP)])
    det_a = TP / np.maximum(1, TP + FN + FP)
    return round(float(np.sqrt(det_a * ass).mean()), 4), round(float(det_a.mean()), 4), round(float(ass.mean()), 4)


for k, tr in TR.items():
    print(k, "HOTA·DetA·AssA 공식 방식", hota_trackeval(gtb, tr), "| JSON", (MOT[k]["HOTA"], MOT[k]["DetA"], MOT[k]["AssA"]))

# ------------------------------------------------------------------ 그림 1: 칼만 예측–갱신
s = Svg(680, 236, "칼만 필터로 트랙을 이어 가는 과정: 예측, 탐지, 갱신, 놓친 프레임에서 예측만으로 버팀")
BW, BH = 50, 20
Y = 128


def box(cx, cy, cls, dash=None, w=BW, h=BH, width=1.8):
    s.add(f'<rect x="{cx - w / 2:.1f}" y="{cy - h / 2:.1f}" width="{w}" height="{h}" class="{cls}" fill="none" '
          f'stroke-width="{width}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")


Kg = float(K[0, 0])
xs = [80, 230, 380, 530]
pred1, det1 = (xs[1], Y), (xs[1] + 10, Y - 9)
upd1 = (pred1[0] + Kg * 10, Y - Kg * 9)
pred2 = (upd1[0] + 150, upd1[1])
pred3 = (pred2[0] + 150, upd1[1])
det3 = (pred3[0] - 8, Y + 12)
upd3 = (pred3[0] - Kg * 6, pred3[1] + Kg * (det3[1] - pred3[1]))
box(xs[0], Y, "s-ac", width=2.2)
box(*pred1, "s-fg", "5 3"); box(*det1, "s-bd", width=2); box(*upd1, "s-ac", width=2.2)
s.add(f'<rect x="{pred2[0] - 44:.1f}" y="{pred2[1] - 24:.1f}" width="88" height="48" rx="10" class="s-mu" fill="none" stroke-width="1.2" stroke-dasharray="2 3"/>')
box(*pred2, "s-fg", "5 3")
s.add(f'<rect x="{pred3[0] - 54:.1f}" y="{pred3[1] - 32:.1f}" width="108" height="64" rx="12" class="s-mu" fill="none" stroke-width="1.2" stroke-dasharray="2 3"/>')
box(*pred3, "s-fg", "5 3"); box(*det3, "s-bd", width=2); box(*upd3, "s-ac", width=2.2)
for a, b in (((xs[0] + 29, Y), (pred1[0] - 31, Y)), ((upd1[0] + 29, upd1[1]), (pred2[0] - 48, pred2[1])),
             ((pred2[0] + 29, pred2[1]), (pred3[0] - 58, pred3[1]))):
    arrow(s, a[0], a[1], b[0], b[1], "s-mu", "f-mu", 1.4)
# 축과 설명
s.line(30, 190, 650, 190, "s-mu", 1)
for x, t1, t2 in ((xs[0], "t−1", "갱신된 위치"), (xs[1], "t", "예측과 탐지 사이로 갱신"),
                  (xs[2] + 3, "t+1 (탐지 놓침)", "예측만으로 버팀"), (xs[3], "t+2", "다시 짝지어 갱신")):
    s.line(x, 186, x, 194, "s-mu", 1)
    s.text(x, 210, t1, "f-fg", 12, weight="600")
    s.text(x, 228, t2, "f-mu", 11)
# 범례
lx = 120
for cls, dash, name, w in (("s-fg", "5 3", "예측", 70), ("s-bd", None, "탐지", 70), ("s-ac", None, "갱신(트랙 상자)", 130),
                            ("s-mu", "2 3", "예측 불확실성", 120)):
    box(lx, 28, cls, dash, 26, 12, 1.6 if cls != "s-mu" else 1.2)
    s.text(lx + 20, 33, name, "f-mu", 12, anchor="start")
    lx += w + 30
s.save(os.path.join(OUT, "42-kalman.svg"))

# ------------------------------------------------------------------ 그림 2: 궤적 PNG
import cv2
from PIL import Image

P = 380
sc = P / TILE
base = render([], [], "sea", seed=7).astype(np.float32)
base[:, 200:331] = base[:, 200:331] * 0.55 + np.array([210, 220, 225]) * 0.45 * 0.55   # 글린트 구역(프레임 50~65에만 생김)
base = cv2.resize(np.clip(base, 0, 255).astype(np.uint8), (P, P), interpolation=cv2.INTER_AREA)


def color(k):
    h = (k * 0.618034) % 1.0
    hsv = np.uint8([[[int(h * 180), 200, 255]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


def panel(points):
    """points: list of (frame, id, x1, y1, x2, y2)"""
    img = base.copy()
    first = {}
    for f, tid, x1, y1, x2, y2 in sorted(points):
        cx, cy = (x1 + x2) / 2 * sc, (y1 + y2) / 2 * sc
        cv2.circle(img, (int(round(cx * 4)), int(round(cy * 4))), 4 * 2, color(tid), -1, cv2.LINE_AA, shift=2)
        first.setdefault(tid, (cx, cy))
    for tid, (cx, cy) in first.items():
        cv2.circle(img, (int(round(cx * 4)), int(round(cy * 4))), 4 * 7, (255, 255, 255), 2, cv2.LINE_AA, shift=2)
    return img


gap = np.full((P, 12, 3), 255, np.uint8)
panels = [panel(gtb), panel(TR["iou_only"]), panel(TR["byte"])]
canvas = np.concatenate([panels[0], gap, panels[1], gap, panels[2]], 1)
im = Image.fromarray(canvas).quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
im.save(os.path.join(OUT, "42-tracks.png"), optimize=True)
print("PNG", im.size, os.path.getsize(os.path.join(OUT, "42-tracks.png")) // 1024, "KB")

# ------------------------------------------------------------------ 그림 3: 지표 막대
s = Svg(680, 286, "추적기 네 가지의 MOTA, IDF1, HOTA 비교")
X0, X1, Y0, Y1 = 64, 664, 214, 44
sy = lambda v: Y0 - (Y0 - Y1) * v
for v in (0, 0.25, 0.5, 0.75, 1.0):
    s.line(X0, sy(v), X1, sy(v), "s-mu" if v else "s-fg", 0.6 if v else 1.2, None if v in (0, 1.0) else "3 3")
    s.text(X0 - 8, sy(v) + 4, f"{v:.2f}", "f-mu", 11, anchor="end")
names = [("iou_only", "IoU만"), ("sort", "SORT식"), ("sort_maxage20", "SORT식 max_age 20"), ("byte", "ByteTrack식")]
mets = [("mota", "MOTA", "f-mu"), ("idf1", "IDF1", "f-ac"), ("HOTA", "HOTA", "f-ok")]
gw = (X1 - X0) / len(names)
bw, bg = 40, 6
for gi, (k, lab) in enumerate(names):
    cx = X0 + gw * (gi + 0.5)
    for mi, (mk, _, cls) in enumerate(mets):
        v = MOT[k][mk]
        x = cx + (mi - 1) * (bw + bg) - bw / 2
        s.rect(x, sy(v), bw, Y0 - sy(v), cls, 0)
        s.text(x + bw / 2, sy(v) - 5, f"{v:.2f}", "f-fg", 11)
    s.text(cx, Y0 + 22, lab, "f-fg", 12, weight="600")
    s.text(cx, Y0 + 40, f"ID 스위치 {int(MOT[k]['num_switches'])} · ID {MOT[k]['n_track_ids']}개", "f-mu", 11)
lx = 380
for _, name, cls in mets:
    s.rect(lx, 14, 14, 14, cls, 0)
    s.text(lx + 20, 26, name, "f-fg", 12, anchor="start")
    lx += 86
s.save(os.path.join(OUT, "42-metrics.svg"))
print("ok")
