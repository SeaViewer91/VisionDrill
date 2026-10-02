"""53강 그림과 본문 수치: NMS 과잉 억제(kill)·누수(leak) 도식, 밀집 계류 비교 막대, 경계 띠(BER) 도식

- 본문 수치 대부분은 data/part8_runs.json의 nmsdiag·segerr(계산: data/run_part8.py의 exp_nmsdiag·exp_segerr)에서 가져옴
- JSON에 없는 것은 여기서 계산해 출력함
  · 놓친 정답의 내역(원시 출력에 후보 없음 / 자기 객체의 다른 상자에 지워짐 / kill), 원문 의사코드식 판정(조건 3 없음)과의 차이
  · NMS 누수(leak) 상자 수: 본문 값은 맨 아래 TIDE식 중복(Dupe) 판정으로 센 것(miss_breakdown의 leak는 정답마다 센 값이라
    밀집 장면에서 한 상자를 이웃 두 정답에 겹쳐 셀 수 있어 참고만 함)
  · 밀집 계류에서 회전 NMS·Soft-NMS 뒤 놓친 정답과 중복 상자 수
  · OFI 손 계산 예, 로짓·확률 상한
  · 분할: 경계 띠가 차지하는 비율(기준선), 띠 폭 2화소일 때 BER, 소형 영역 넓이, 허위 섬·구멍 덩어리 수
"""
import math, os, sys
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
os.environ.setdefault("OMP_NUM_THREADS", "1")
from svglib import Svg
import json

J = json.load(open(os.path.join(DATA, "part8_runs.json"), encoding="utf-8"))
ND, SE = J["nmsdiag"], J["segerr"]
OUT = os.path.join(HERE, "..", "fig")

import run_part8 as R8                     # 6부 항만 타일·원시 출력(T6, RAW6)과 nms_track·classify_dets를 그대로 씀
from run_part8 import T6, RAW6, nms_track, classify_dets, map50
from part6_eval import iou_matrix
from make_part6 import detect, obb_to_hbb


# ---------------------------------------------------------------- 1. 놓친 정답 내역과 누수
def miss_breakdown(pairs, thr, score_min=0.05):
    """pairs = [(gts, raw)]. 반환: 놓친 수, 후보 없음, 후보 살아남음, 자기 상자에 지워짐, kill, 의사코드식 kill(점수 ≥ 0.25, 조건 3 없음), 누수 상자 수"""
    c = dict(miss=0, none=0, alive=0, self=0, kill=0, pseudo=0, leak=0)
    for gts, raw in pairs:
        G = np.array([g["hbb"] for g in gts]); gc = np.array([g["cls"] for g in gts])
        keep, by = nms_track(raw, thr)
        post = [raw[i] for i in keep]
        _, types, tgt, gstate = classify_dets(gts, post)
        M = iou_matrix([d["hbb"] for d in raw], G)
        ptp = {i: int(M[i].argmax()) for i in range(len(raw)) if M[i].max() >= 0.5 and gc[M[i].argmax()] == raw[i]["cls"] and raw[i]["score"] >= score_min}
        tp_g = {tgt[k] for k, ty in enumerate(types) if ty == "TP"}
        Mp = iou_matrix([d["hbb"] for d in post], G) if post else np.zeros((0, len(G)))
        for j in range(len(gts)):
            n_same = int(sum(1 for k, d in enumerate(post) if d["cls"] == gc[j] and Mp[k, j] >= 0.5))
            c["leak"] += max(0, n_same - 1)
            if j in tp_g:
                continue
            c["miss"] += 1
            cands = [i for i, jj in ptp.items() if jj == j]
            if any(raw[i]["score"] >= 0.25 for i in cands):
                c["pseudo"] += 1
            if not cands:
                c["none"] += 1
            elif not all(i in by for i in cands):
                c["alive"] += 1
            elif any(M[by[i]].max() >= 0.5 and int(M[by[i]].argmax()) != j for i in cands):
                c["kill"] += 1
            else:
                c["self"] += 1
    return c


harbor = [(t["objs"], r) for t, r in zip(T6, RAW6)]
for thr in (0.3, 0.5, 0.7):
    c = miss_breakdown(harbor, thr)
    js = ND[f"classwise_{thr}"]
    assert c["miss"] == js["misses"] and c["kill"] == js["kills"], (c, js)
    print(f"항만 NMS {thr}: {c}  (JSON kill_ratio {js['kill_ratio']}, 의사코드식 비율 {c['pseudo'] / c['miss']:.3f})")

# 밀집 계류(run_part8.exp_nmsdiag과 같은 생성 절차·시드)
rng = np.random.default_rng(81)
dense = []
for k in range(6):
    objs = []
    for row in range(3):
        for i in range(8):
            L = rng.uniform(12, 16) / 0.5; W = L * 0.32
            th = 45 + rng.normal(0, 2)
            step = W + 2.0
            cx = 120 + row * 130 + i * step / math.sqrt(2) + rng.normal(0, 0.5)
            cy = 110 + i * step / math.sqrt(2) + row * 60 + rng.normal(0, 0.5)
            o = dict(cls=1, cx=cx, cy=cy, w=L, h=W, theta=th)
            o["obb"] = [cx, cy, L, W, th]; o["hbb"] = obb_to_hbb((cx, cy, L, W, th)).tolist()
            hb = o["hbb"]; o["area"] = (hb[2] - hb[0]) * (hb[3] - hb[1])
            objs.append(o)
    dense.append((objs, detect(objs, [], 1.0, seed=500 + k)))
for thr in (0.3, 0.5, 0.7):
    c = miss_breakdown(dense, thr)
    js = ND["dense_mooring"][f"nms_{thr}"]
    assert c["miss"] == js["misses"] and c["kill"] == js["kills"], (c, js)
    print(f"밀집 NMS {thr}: {c}")
Ls = [o["w"] * 0.5 for objs, _ in dense for o in objs]
print("밀집 계류 선박 길이(m)", round(min(Ls), 1), "~", round(max(Ls), 1), "폭 = 길이 × 0.32, 원시 상자", sum(len(r) for _, r in dense))

# ---------------------------------------------------------------- 2. OFI
def ofi_one(scores, logit):
    s = np.clip(np.asarray(scores, float), 1e-4, 1 - 1e-4)
    z = np.log(s / (1 - s)) if logit else s
    p = np.exp(z - z.max()); p /= p.sum()
    return 1 - (-(p * np.log(p)).sum()) / np.log(len(s))

ex = [0.95, 0.60, 0.30, 0.10]
print("OFI 예", ex, "로짓", [round(math.log(s / (1 - s)), 2) for s in ex], "→ 로짓 OFI", round(ofi_one(ex, True), 3), "/ 확률 OFI", round(ofi_one(ex, False), 3))
ex2 = [0.99, 0.02, 0.02, 0.02]
print("OFI 예2", ex2, "→ 로짓", round(ofi_one(ex2, True), 3), "/ 확률", round(ofi_one(ex2, False), 3))
p = np.exp([1, 0, 0, 0.0]); p /= p.sum()
print("확률 상한(1·0·0·0) 소프트맥스", np.round(p, 3), "OFI", round(1 - (-(p * np.log(p)).sum()) / np.log(4), 4), "JSON", ND["ofi"]["cap_prob_K4"])
for K in (2, 3, 4):
    p = np.exp([1.0] + [0.0] * (K - 1)); p /= p.sum()
    print(f"확률 척도 OFI 상한 K={K}:", round(1 - (-(p * np.log(p)).sum()) / np.log(K), 4))
gaps, tops, seconds = [], [], []
for t, raw in zip(T6, RAW6):
    G = np.array([g["hbb"] for g in t["objs"]])
    M = iou_matrix([d["hbb"] for d in raw], G)
    for j in range(len(G)):
        idx = np.where(M[:, j] >= 0.5)[0]
        if len(idx) < 2:
            continue
        s = np.sort([raw[i]["score"] for i in idx])[::-1]
        tops.append(s[0]); seconds.append(s[1])
        lg = np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4)))
        gaps.append(lg[0] - lg[1])
print("OFI 대상 정답", len(tops), "1위 점수 중앙값", round(float(np.median(tops)), 3), "2위", round(float(np.median(seconds)), 3),
      "1·2위 로짓 차 중앙값", round(float(np.median(gaps)), 3), "JSON", ND["ofi"])

# ---------------------------------------------------------------- 3. 분할: 경계 띠 기준선, 띠 폭, 소형 영역, 섬·구멍
import torch
from scipy import ndimage as ndi
from make_part7 import scenes
from part5_common import TinyUNet
from part7_common import to_tensor
torch.set_num_threads(1)
S = scenes()
m = TinyUNet(skip=True); m.load_state_dict(torch.load(os.path.join(DATA, ".part7_unet.pt"))); m.eval()


def infer(x):
    H = x.shape[1]; out = torch.zeros(6, H, H); cnt = torch.zeros(1, H, H)
    with torch.no_grad():
        for r in range(0, H - 31, 16):
            for c in range(0, H - 31, 16):
                out[:, r:r + 32, c:c + 32] += m(x[None, :, r:r + 32, c:c + 32])[0].softmax(0); cnt[:, r:r + 32, c:c + 32] += 1
    return (out / cnt).argmax(0).numpy()


CL = R8.CLS5
acc = {c: dict(err=0, b1=0, b2=0, band1=0, union=0) for c in range(6)}
small_areas, lost_areas, mid = [], [], dict(n=0, lost=0)
islands = hole_comps = 0
small8 = dict(n=0, lost=0, areas=[])
conf = np.zeros((6, 6), int)
for k in (8, 9):
    y = S[k]["m"]; p = infer(to_tensor(S[k]["x"][None])[0])
    conf += np.bincount(y.ravel() * 6 + p.ravel(), minlength=36).reshape(6, 6)
    for c in range(6):
        Yc, Pc = y == c, p == c
        E = Yc != Pc
        b1 = ndi.binary_dilation(Yc, iterations=1) & ~ndi.binary_erosion(Yc, iterations=1)
        b2 = ndi.binary_dilation(Yc, iterations=2) & ~ndi.binary_erosion(Yc, iterations=2)
        a = acc[c]
        a["err"] += int(E.sum()); a["b1"] += int((E & b1).sum()); a["b2"] += int((E & b2).sum())
        a["band1"] += int((b1 & (Yc | Pc)).sum()); a["union"] += int((Yc | Pc).sum())
        lg, ng = ndi.label(Yc); lp, npd = ndi.label(Pc)
        for i in range(1, ng + 1):
            comp = lg == i; ar = int(comp.sum())
            if ar < 100:
                small_areas.append(ar)
                if not (comp & Pc).any():
                    lost_areas.append(ar)
            elif ar < 400:
                mid["n"] += 1; mid["lost"] += int(not (comp & Pc).any())
        for q in range(1, npd + 1):
            if not ((lp == q) & Yc).any():
                islands += 1
        hl, nh = ndi.label(ndi.binary_fill_holes(Pc) & ~Pc); hole_comps += nh
        l8, n8 = ndi.label(Yc, structure=np.ones((3, 3)))      # 8-이웃으로 묶으면
        for i in range(1, n8 + 1):
            comp = l8 == i
            if comp.sum() < 100:
                small8["n"] += 1; small8["lost"] += int(not (comp & Pc).any()); small8["areas"].append(int(comp.sum()))
tot = {k: sum(acc[c][k] for c in range(6)) for k in acc[0]}
print("BER(띠 1) 전체", round(tot["b1"] / tot["err"], 4), "JSON", SE["ber_all"], "/ 띠 2화소", round(tot["b2"] / tot["err"], 4))
print("경계 띠(1)가 (정답∪예측)에서 차지하는 비율 전체", round(tot["band1"] / tot["union"], 4))
for c in range(6):
    a = acc[c]
    print(f"  {CL[c]}: BER {a['b1'] / a['err']:.3f} (JSON {SE['ber_per_class'][CL[c]]}), 띠 2 {a['b2'] / a['err']:.3f}, 띠 비율 {a['band1'] / a['union']:.3f}, 틀린 화소 {a['err']}")
sa, la = np.array(small_areas), np.array(lost_areas)
print("소형 영역 넓이 < 16:", int((sa < 16).sum()), "소실", int((la < 16).sum()), "/ 16~99:", int((sa >= 16).sum()), "소실", int((la >= 16).sum()), "/ 1화소:", int((sa == 1).sum()))
print("소형 영역", len(small_areas), "넓이 중앙값", np.median(small_areas), "소실", len(lost_areas), "소실 넓이 중앙값", np.median(lost_areas),
      "최대", max(small_areas), "/ 100~400화소", mid)
print("8-이웃 소형 영역", small8)
print("혼동행렬(행 = 정답, 열 = 예측)", CL); print(conf, "틀린 화소", int(conf.sum() - np.trace(conf)), "/", int(conf.sum()))
print("허위 섬(같은 클래스 정답과 한 화소도 안 겹치는 예측 덩어리)", islands, "/ 예측 덩어리", SE["n_pred_components"], "/ 구멍 덩어리", hole_comps, "구멍 화소", SE["hole_pixels_pred"])


# ---------------------------------------------------------------- 그림 1: kill vs leak 도식
def rot(cx, cy, L, W, th):
    t = math.radians(th); c, s = math.cos(t), math.sin(t)
    pts = [(-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2)]
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]


def poly(P):
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in P) + " Z"


def hbb(P):
    xs, ys = [p[0] for p in P], [p[1] for p in P]
    return min(xs), min(ys), max(xs), max(ys)


def hollow(x0, y0, x1, y1, cls, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    s.add(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" class="{cls}" stroke-width="2" fill="none"{d}/>')


s = Svg(760, 330, "왼쪽: NMS 과잉 억제(kill). 45°로 나란한 두 배의 수평 박스가 크게 겹쳐, 배 A를 노린 높은 점수 상자가 배 B를 노린 상자를 지움. "
        "오른쪽: NMS 누수(leak). 한 배에 붙은 두 상자의 IoU가 임계값보다 조금 낮아 둘 다 살아남음")
s.line(380, 30, 380, 320, "s-mu", 0.8, "3 4")
s.text(190, 24, "과잉 억제 (kill)", "f-fg", 14, weight="600")
s.text(570, 24, "누수 (leak)", "f-fg", 14, weight="600")
# 왼쪽: 두 배. 이웃 배는 폭 방향(−45° 방향)으로 폭 + 간격 4만큼 이동
L_, W_ = 150, 44
off = W_ + 4
nx, ny = math.cos(math.radians(-45)) * off, math.sin(math.radians(-45)) * off
A = rot(205, 150, L_, W_, 45)
B = rot(205 - nx, 150 - ny, L_, W_, 45)
s.path(poly(A), "s-fg f-sf", 1.4)
s.path(poly(B), "s-fg f-sf", 1.4)
ha, hb_ = hbb(A), hbb(B)
hollow(*ha, "s-ac")
hollow(*hb_, "s-bd", "6 4")
iou_hbb = iou_matrix([list(ha)], [list(hb_)])[0, 0]
print("그림 1 두 HBB IoU", round(float(iou_hbb), 3), "상자", [round(v) for v in ha], [round(v) for v in hb_])
ca = (sum(p[0] for p in A) / 4, sum(p[1] for p in A) / 4); cb = (sum(p[0] for p in B) / 4, sum(p[1] for p in B) / 4)
s.text(ca[0], ca[1] + 5, "A", "f-fg", 13, weight="600")
s.text(cb[0], cb[1] + 5, "B", "f-fg", 13, weight="600")
s.text(ha[2], ha[1] - 7, "A를 노린 상자 0.92", "f-ac", 11, anchor="end")
s.text(hb_[0], hb_[3] + 16, "B를 노린 상자 0.81 → 지워짐", "f-bd", 11, anchor="start")
s.text(190, 300, f"상자끼리 IoU {iou_hbb:.2f} > 임계값 0.3", "f-mu", 11)
s.text(190, 316, "B는 원시 출력엔 있었는데 결과엔 없음", "f-mu", 11)
# 오른쪽: 한 배에 상자 둘
C = rot(570, 160, 170, 50, 0)
s.path(poly(C), "s-fg f-sf", 1.4)
s.text(570, 165, "C", "f-fg", 13, weight="600")
hollow(480, 128, 660, 194, "s-ac")
hollow(498, 122, 680, 200, "s-bd", "6 4")
iou_r = iou_matrix([[480, 128, 660, 194]], [[498, 122, 680, 200]])[0, 0]
print("누수 상자와 C의 IoU", round(float(iou_matrix([hbb(C)], [[480, 128, 660, 194], [498, 122, 680, 200]])[0].min()), 3))
print("그림 1 누수 상자 IoU", round(float(iou_r), 3))
s.text(480, 116, "C를 노린 상자 0.90", "f-ac", 11, anchor="start")
s.text(680, 218, "C를 노린 상자 0.74 → 남음", "f-bd", 11, anchor="end")
s.text(570, 300, f"상자끼리 IoU {iou_r:.3f} ≤ 임계값 0.7", "f-mu", 11)
s.text(570, 316, "둘 다 C와 IoU ≥ 0.5 → 하나는 중복 오탐", "f-mu", 11)
s.save(os.path.join(OUT, "53-kill-leak.svg"))

# ---------------------------------------------------------------- 그림 2: 항만 vs 밀집 계류
dm = ND["dense_mooring"]
s = Svg(760, 320, "왼쪽: 놓친 정답 가운데 kill 비율. 항만 타일(클래스별 NMS)은 0.3에서도 2%이고 0.5·0.7에서는 0, 밀집 계류는 0.3에서 80%, 0.5에서 45%. "
        "오른쪽: 밀집 계류의 mAP50. NMS 0.3·0.5·0.7, Soft-NMS, 회전 NMS 0.5")
X0, Y0, H = 70, 260, 200
s.line(X0, Y0, 350, Y0, "s-fg", 1.2); s.line(X0, Y0, X0, Y0 - H - 10, "s-fg", 1.2)
for v in (0, 0.25, 0.5, 0.75, 1.0):
    yy = Y0 - v * H
    s.line(X0 - 4, yy, X0, yy, "s-fg", 1); s.text(X0 - 8, yy + 4, f"{v:.2f}".rstrip("0").rstrip(".") if v else "0", "f-mu", 11, anchor="end")
s.text(210, 24, "kill / 놓친 정답", "f-fg", 13, weight="600")
for gi, thr in enumerate((0.3, 0.5, 0.7)):
    gx = X0 + 25 + gi * 90
    hv = ND[f"classwise_{thr}"]["kill_ratio"]; dv = dm[f"nms_{thr}"]["kill_ratio"]
    s.rect(gx, Y0 - hv * H, 28, max(hv * H, 0.01), "s-ac f-acs", 1.2)
    s.rect(gx + 32, Y0 - dv * H, 28, dv * H, "s-bd f-bds", 1.2)
    s.text(gx + 14, Y0 - hv * H - 5, f"{hv:.2f}", "f-ac", 10)
    s.text(gx + 46, Y0 - dv * H - 5, f"{dv:.2f}", "f-bd", 10)
    s.text(gx + 30, Y0 + 17, f"NMS {thr}", "f-fg", 11)
s.rect(240, 44, 12, 12, "s-ac f-acs", 1.2); s.text(258, 54, "항만 타일", "f-fg", 11, anchor="start")
s.rect(240, 62, 12, 12, "s-bd f-bds", 1.2); s.text(258, 72, "밀집 계류", "f-fg", 11, anchor="start")
X1 = 440
s.line(X1, Y0, 740, Y0, "s-fg", 1.2); s.line(X1, Y0, X1, Y0 - H - 10, "s-fg", 1.2)
for v in (0, 0.25, 0.5, 0.75, 1.0):
    yy = Y0 - v * H
    s.line(X1 - 4, yy, X1, yy, "s-fg", 1); s.text(X1 - 8, yy + 4, f"{v:.2f}".rstrip("0").rstrip(".") if v else "0", "f-mu", 11, anchor="end")
s.text(590, 24, "밀집 계류 mAP50", "f-fg", 13, weight="600")
bars = [("0.3", dm["nms_0.3"]["ap50"]), ("0.5", dm["nms_0.5"]["ap50"]), ("0.7", dm["nms_0.7"]["ap50"]),
        ("Soft", dm["soft_nms_ap50"]), ("회전 0.5", dm["rotated_nms_0.5_ap50"])]
for i, (lab, v) in enumerate(bars):
    bx = X1 + 18 + i * 57
    cls = "s-ok f-sf" if lab == "Soft" else "s-bd f-bds"
    s.rect(bx, Y0 - v * H, 36, v * H, cls, 1.2)
    s.text(bx + 18, Y0 - v * H - 5, f"{v:.3f}", "f-fg", 10)
    s.text(bx + 18, Y0 + 17, lab, "f-fg", 11)
s.text(590, Y0 + 40, "NMS 임계값 / Soft-NMS / 회전 NMS", "f-mu", 11)
s.text(210, Y0 + 40, "NMS IoU 임계값(클래스별)", "f-mu", 11)
s.save(os.path.join(OUT, "53-dense.svg"))

# ---------------------------------------------------------------- 그림 3: 경계 띠와 BER
s = Svg(760, 300, "격자 한 칸이 한 화소. 왼쪽: 정답 영역(회색)과 경계 띠(1화소 팽창 − 1화소 침식, 줄무늬 테두리). "
        "가운데: 예측이 경계에서 한 화소 밀린 경우, 틀린 화소가 모두 띠 안이라 BER = 1. 오른쪽: 내부에 다른 클래스로 칠한 덩어리가 있으면 띠 밖 오차가 생겨 BER이 낮아짐")
N = 10; CELL = 20
gt = np.zeros((N, N), bool); gt[2:8, 2:8] = True
band = ndi.binary_dilation(gt) & ~ndi.binary_erosion(gt)
pred_b = np.zeros((N, N), bool); pred_b[2:8, 3:9] = True
pred_c = gt.copy(); pred_c[4:6, 4:6] = False


def grid(ox, oy, pred, title):
    s.text(ox + N * CELL / 2, oy - 12, title, "f-fg", 13, weight="600")
    for r in range(N):
        for c in range(N):
            x, y = ox + c * CELL, oy + r * CELL
            if gt[r, c]:
                s.rect(x, y, CELL, CELL, "s-mu f-sf", 0.5)
            else:
                s.rect(x, y, CELL, CELL, "s-mu f-bg", 0.5)
    for r in range(N):
        for c in range(N):
            if band[r, c]:
                x, y = ox + c * CELL, oy + r * CELL
                s.line(x + 3, y + CELL - 3, x + CELL - 3, y + 3, "s-ac", 1.0)
    if pred is not None:
        E = pred != gt
        for r in range(N):
            for c in range(N):
                if E[r, c]:
                    s.circle(ox + c * CELL + CELL / 2, oy + r * CELL + CELL / 2, 5, "f-bd")
        ys, xs = np.where(pred)
        s.add(f'<rect x="{ox + xs.min() * CELL:.1f}" y="{oy + ys.min() * CELL:.1f}" width="{(xs.max() - xs.min() + 1) * CELL:.1f}" '
              f'height="{(ys.max() - ys.min() + 1) * CELL:.1f}" class="s-fg" stroke-width="2" fill="none" stroke-dasharray="5 3"/>')
        e, eb = int(E.sum()), int((E & band).sum())
        return e, eb
    return None


OY = 50
grid(30, OY, None, "정답과 경계 띠")
e1, b1 = grid(280, OY, pred_b, "경계에서 한 칸 밀림")
e2, b2 = grid(530, OY, pred_c, "내부 구멍")
s.text(130, OY + N * CELL + 24, f"띠 화소 {int(band.sum())}개", "f-mu", 11)
s.text(380, OY + N * CELL + 24, f"오차 {e1} · 띠 안 {b1} → BER {b1 / e1:.2f}", "f-fg", 11)
s.text(630, OY + N * CELL + 24, f"오차 {e2} · 띠 안 {b2} → BER {b2 / e2:.2f}", "f-fg", 11)
s.circle(300, OY + N * CELL + 44, 5, "f-bd"); s.text(310, OY + N * CELL + 48, "틀린 화소", "f-fg", 11, anchor="start")
s.line(396, OY + N * CELL + 50, 410, OY + N * CELL + 36, "s-ac", 1.0); s.text(416, OY + N * CELL + 48, "경계 띠", "f-fg", 11, anchor="start")
s.add(f'<rect x="490" y="{OY + N * CELL + 37}" width="16" height="12" class="s-fg" stroke-width="2" fill="none" stroke-dasharray="5 3"/>')
s.text(512, OY + N * CELL + 48, "예측 영역", "f-fg", 11, anchor="start")
print("그림 3 BER", e1, b1, e2, b2)
s.save(os.path.join(OUT, "53-band.svg"))
print("그림 저장 완료")

# ---------------------------------------------------------------- 밀집 계류: 회전 NMS·Soft-NMS 뒤 놓친 정답과 중복(누수) 상자 수
from part6_eval import nms_dets, soft_nms_dets
for name, fn in (("회전 NMS 0.5", lambda r: nms_dets(r, 0.5, rotated=True)), ("HBB NMS 0.5", lambda r: nms_dets(r, 0.5)),
                 ("Soft-NMS(점수 ≥ 0.3)", lambda r: [d for d in soft_nms_dets(r, 0.5) if d["score"] >= 0.3])):
    miss = leak = 0
    for objs, raw in dense:
        post = fn(raw)
        _, types, tgt, gstate = classify_dets(objs, post)
        miss += sum(1 for g in gstate if g != "TP"); leak += sum(1 for t in types if t == "Dupe")
    print(f"밀집 {name}: 놓친 정답 {miss}, 중복(Dupe) 상자 {leak}")

# 누수 상자 수를 TIDE식 중복(Dupe) 판정으로 셈(한 상자를 두 정답에 겹쳐 세지 않음) — 본문 표에 쓰는 값
for nm, pairs in (("항만", harbor), ("밀집", dense)):
    for thr in (0.3, 0.5, 0.7):
        dup = 0
        for gts, raw in pairs:
            keep, _ = nms_track(raw, thr)
            _, types, _, _ = classify_dets(gts, [raw[i] for i in keep])
            dup += sum(1 for t in types if t == "Dupe")
        print(f"{nm} NMS {thr}: 누수(Dupe) 상자 {dup}")
