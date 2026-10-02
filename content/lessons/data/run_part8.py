"""8부(모델 진단, Diagnostics) 본문 수치를 만드는 계산 모음. 결과는 part8_runs.json. CPU 1스레드로 전체 20분 안팎.

    python3 run_part8.py              # 전체
    python3 run_part8.py tide robust  # 일부만 (다른 결과는 유지)

진단 대상(모두 앞 부의 가상 자료·모델)
- 탐지: 6부 항만 타일 24장 + 가상 탐지기의 기본 탐지 결과(클래스별 NMS 0.5) — 34강
- 분류: 5부 VD-CNN(best, 27에폭) + 5부 타일 자료(시험·연무·계절 시험) — 27·48강
- 분할: 7부 U-Net(장면 0~5로 학습) + 7부 시험 장면 8·9 — 43·50강
- 다종 센서: 4부 가상 연안 장면(광학 6밴드 + SAR) — 20·22강
지표 정의는 Diagnostics 노트를 따르되, 노트가 정하지 않은 세부(예: 경계 띠 폭, OFI의 K가 모자랄 때)는 여기서 정한 방식을 note에 적음.

실험 이름과 쓰는 강
- tide     : TIDE 계열 6대 오차 분해와 오차별 ΔAP(오라클), 중심·스케일 편차, 고신뢰 배경 오탐 비율 (52강)
- nmsdiag  : NMS 과잉 억제(kill)·교차 클래스 억제율(CCSR)·OFI(로짓 vs 확률) (53강)
- segerr   : 분할 오차 분해: 경계 오차 비율(BER), 소형 영역 소실, 병합·쪼개짐, 구멍 (53강)
- slice    : 취약 슬라이스 Z-검정과 BH 보정, 라벨 잡음 격리(교차 예측 신뢰 학습), 분포 거리(W1·MMD), 탐지 슬라이스 (54강)
- calib    : ECE·MCE·온도 스케일링, MC 드롭아웃 불확실성 분해 (55강)
- repr     : 유효 랭크·CUR·비활성 채널(DCR)·NC1, 유효 수용영역(ERF)과 RFM, ViT 어텐션 엔트로피 (56강)
- robust   : 10대 교란 × 5단계 심각도, RDI·CE·mCE, 광학 증강 모델과 비교 (57강)
- fusion   : 광학+SAR 융합, 모달리티 결측·구름·정렬 오차, 섀플리 값·DR·MBR (58강)
- causal   : 바로가기 단서 실험(반사실 개입), ACE·반사실 뒤집힘률, 단서 무작위화 처방 (59강)
- hw       : int8 가짜 양자화의 층별 SQNR·PAR, 이상치 채널과 SmoothQuant, 루프라인 산술 강도 (60강)
- ops      : 두 모델 버전의 플립 분석(N10·회귀율), 꼬리 지연시간, 5대 배포 게이트 판정 (61강)
- prx      : 처방 규칙 엔진 실행 예(탐지기·분류기), 임계값 범위에 따른 처방 변화 (51강)
"""
import io, contextlib, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import part5_common as P5C
import make_part5 as P5
from part5_common import TinyCNN, TinyUNet, TinyViT, FlatMLP, evaluate, n_params
from part7_common import fit, to_tensor, norm_stats
from make_part6 import tiles, detect, obb_to_hbb
from part6_eval import iou_matrix, nms_dets, coco_eval, pr_points, ap_from_pr

OUT = os.environ.get("PART8_OUT") or os.path.join(HERE, "part8_runs.json")
R = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
D = P5C.data()
CLS5 = ["산림", "농경지", "시가지", "수계", "나지", "갯벌"]


def r4(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 4)


def save():
    json.dump(R, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def base_model():
    m = TinyCNN()
    m.load_state_dict(torch.load(os.path.join(HERE, ".part5_base.pt")))
    m.eval()
    return m


def logits_of(m, x, bs=300):
    m.eval()
    with torch.no_grad():
        return torch.cat([m(x[i:i + bs]) for i in range(0, len(x), bs)])


T6 = tiles()
RAW6 = [detect(t["objs"], t["clutter"], 1.0, seed=t["image_id"]) for t in T6]
BASE6 = [(t["objs"], nms_dets(r, 0.5)) for t, r in zip(T6, RAW6)]


# ================================================================== 52 탐지 오차 분해
def classify_dets(gts, dets, tf=0.5, tb=0.1):
    """TIDE 계열 분류. 반환: dets 각각의 유형(TP·Cls·Loc·Both·Dupe·Bkg)과 짝 정답, 정답 각각의 상태(TP·Miss·Covered)"""
    dets = sorted(dets, key=lambda d: -d["score"])
    G = [g["hbb"] for g in gts]
    M = iou_matrix([d["hbb"] for d in dets], G) if dets and G else np.zeros((len(dets), len(G)))
    gcls = np.array([g["cls"] for g in gts]) if gts else np.zeros(0, int)
    matched = np.full(len(gts), -1)
    types, tgt = [], []
    for i, d in enumerate(dets):
        same = (gcls == d["cls"]) if len(G) else np.zeros(0, bool)
        cand = [j for j in range(len(G)) if same[j] and matched[j] < 0 and M[i, j] >= tf]
        if cand:
            j = max(cand, key=lambda j: M[i, j]); matched[j] = i
            types.append("TP"); tgt.append(j); continue
        iou_same = M[i][same].max() if same.any() else 0.0
        iou_any = M[i].max() if len(G) else 0.0
        if iou_same >= tf:
            types.append("Dupe"); tgt.append(int(np.where(same)[0][M[i][same].argmax()]))
        elif iou_any >= tf:
            types.append("Cls"); tgt.append(int(M[i].argmax()))
        elif iou_same >= tb:
            types.append("Loc"); tgt.append(int(np.where(same)[0][M[i][same].argmax()]))
        elif iou_any >= tb:
            types.append("Both"); tgt.append(int(M[i].argmax()))
        else:
            types.append("Bkg"); tgt.append(-1)
    covered = set(j for t, j in zip(types, tgt) if t in ("Cls", "Loc"))
    gstate = ["TP" if matched[j] >= 0 else ("Covered" if j in covered else "Miss") for j in range(len(gts))]
    return dets, types, tgt, gstate


def map50(images):
    return coco_eval(images, thrs=np.array([0.5]))["ap50"]


def exp_tide():
    base_ap = map50(BASE6)
    cnt = {k: 0 for k in ["TP", "Cls", "Loc", "Both", "Dupe", "Bkg"]}
    cnt_hi = dict(cnt)
    miss = covered = 0
    dc, ds, center_dom = [], [], 0
    loc_on_tp = 0
    by_size = {"small": {k: 0 for k in ["Loc", "Miss", "Bkg", "TP"]}, "medium+": {k: 0 for k in ["Loc", "Miss", "Bkg", "TP"]}}
    fixed = {k: [] for k in ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]}
    for gts, dets in BASE6:
        ds_, types, tgt, gstate = classify_dets(gts, dets)
        for d, t, j in zip(ds_, types, tgt):
            cnt[t] += 1
            if d["score"] >= 0.5:
                cnt_hi[t] += 1
            if t == "Loc":
                loc_on_tp += gstate[j] == "TP"
                g = gts[j]["hbb"]; b = d["hbb"]
                gw, gh = g[2] - g[0], g[3] - g[1]; bw, bh = b[2] - b[0], b[3] - b[1]
                c = math.hypot((b[0] + b[2]) / 2 - (g[0] + g[2]) / 2, (b[1] + b[3]) / 2 - (g[1] + g[3]) / 2) / math.sqrt(gw * gh)
                s = abs(math.log(bw / gw)) + abs(math.log(bh / gh))
                dc.append(c); ds.append(s); center_dom += c >= s
            if t in ("TP", "Loc", "Bkg"):
                if t == "Bkg":
                    a = (d["hbb"][2] - d["hbb"][0]) * (d["hbb"][3] - d["hbb"][1])
                else:
                    a = gts[j]["area"]
                by_size["small" if a < 32 ** 2 else "medium+"][t] += 1
        miss += gstate.count("Miss"); covered += gstate.count("Covered")
        for j, st in enumerate(gstate):
            if st == "Miss":
                by_size["small" if gts[j]["area"] < 32 ** 2 else "medium+"]["Miss"] += 1
        # 오라클: 한 유형씩 고친 결과
        for k in fixed:
            nd, ng = [], list(gts)
            used = {jj for jj, st in enumerate(gstate) if st == "TP"}
            owner = {}                                # TIDE: 한 정답을 가리키는 Cls·Loc 오차 가운데 점수가 가장 높은 것만 고칠 자격
            for ii, (t, j) in enumerate(zip(types, tgt)):
                if t in ("Cls", "Loc"):
                    owner.setdefault(j, ii)
            for ii, (d, t, j) in enumerate(zip(ds_, types, tgt)):
                d2 = dict(d)
                if t == k and k in ("Cls", "Loc") and (j in used or owner.get(j) != ii):
                    continue                          # 짝이 이미 있는 정답이나 더 높은 점수의 오차가 차지한 정답을 가리키면 고치는 대신 지움
                if t == k and k == "Cls":
                    d2["cls"] = gts[j]["cls"]; used.add(j)
                elif t == k and k == "Loc":
                    d2["hbb"] = list(gts[j]["hbb"]); used.add(j)
                elif t == k and k in ("Both", "Dupe", "Bkg"):
                    continue
                nd.append(d2)
            if k == "Miss":
                ng = [g for g, st in zip(gts, gstate) if st != "Miss"]
            fixed[k].append((ng, nd))
    dap = {k: r4(map50(v) - base_ap) for k, v in fixed.items()}
    fp = sum(cnt[k] for k in ["Cls", "Loc", "Both", "Dupe", "Bkg"])
    # 고신뢰 배경 오탐 비율: 원래 라벨 vs 라벨 20%를 지운 경우(44강 missing과 같은 방식)
    def hi_bkg_ratio(images, thr=0.85):
        hi = allfp = 0
        for gts, dets in images:
            ds_, types, _, _ = classify_dets(gts, dets)
            for d, t in zip(ds_, types):
                if t != "TP":
                    allfp += 1
                    hi += (t == "Bkg" and d["score"] >= thr)
        return hi, allfp
    rng = np.random.default_rng(5)
    dropped = [([o for o in g if rng.random() >= 0.2], d) for g, d in BASE6]
    h0, f0 = hi_bkg_ratio(BASE6); h1, f1 = hi_bkg_ratio(dropped)
    hs = {}
    for thr in (0.5, 0.7, 0.85):
        a, b = hi_bkg_ratio(BASE6, thr); c, e = hi_bkg_ratio(dropped, thr)
        hs[str(thr)] = dict(clean=r4(a / b), drop20=r4(c / e))
    R["tide"] = dict(base_map50=base_ap, counts=cnt, counts_score_ge_0_5=cnt_hi, n_gt=sum(len(g) for g, _ in BASE6), miss=miss,
                     covered_gt=covered, fp_total=fp, delta_ap50_oracle=dap,
                     loc=dict(n=len(dc), center_median=r4(np.median(dc)), scale_median=r4(np.median(ds)), center_dominant_frac=r4(center_dom / len(dc)), on_matched_gt=loc_on_tp),
                     by_size=by_size, high_conf_bkg_ratio=dict(clean=r4(h0 / f0), drop20=r4(h1 / f1), n_fp_clean=f0, n_fp_drop20=f1, by_threshold=hs),
                     note=("6부 기본 탐지 결과, 전경 IoU 0.5·배경 IoU 0.1, 점수 높은 탐지부터 판정. Covered = 짝은 없지만 Cls·Loc 오차가 가리키는 정답(TIDE처럼 Miss에서 뺌). "
                           "오라클 ΔAP는 mAP50 기준(TIDE 방식): Cls는 클래스를 고치고, Loc는 상자를 정답으로 바꾸되 가리키는 정답이 이미 짝(TP)이 있거나 그 정답을 가리키는 Cls·Loc 오차 중 점수 1위가 아니면 그 탐지를 지우고, Both·Dupe·Bkg는 그 탐지를 지우고, Miss는 놓친 정답을 정답 목록에서 뺀 뒤 다시 잼. "
                           "중심 우세 = Δcenter ≥ Δscale인 Loc 오차 비율(이 자습서에서 정한 판정). loc.on_matched_gt = 이미 짝이 있는 정답 둘레의 Loc 오차 수(사실상 덜 지워진 두 번째 상자). 고신뢰 배경 오탐 비율 = (점수 ≥ 0.85인 Bkg) / 전체 오탐"))


# ================================================================== 53 NMS·분할 오차
def nms_track(dets, thr, agnostic=False):
    """탐욕 NMS. 남은 것과, 지워진 상자마다 지운 상자 인덱스"""
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["score"])
    B = np.array([d["hbb"] for d in dets]) if dets else np.zeros((0, 4))
    M = iou_matrix(B, B) if len(dets) else np.zeros((0, 0))
    alive = np.ones(len(dets), bool); by = {}
    keep = []
    for i in order:
        if not alive[i]:
            continue
        keep.append(i)
        for j in order:
            if alive[j] and j != i and M[i, j] > thr and (agnostic or dets[j]["cls"] == dets[i]["cls"]):
                alive[j] = False; by[j] = i
    return keep, by


def exp_nmsdiag():
    out = {}
    for thr in (0.3, 0.5, 0.7):
        for agn in (False, True):
            kills = misses = pre_tp = cc_supp = 0
            for t, raw in zip(T6, RAW6):
                gts = t["objs"]
                G = np.array([g["hbb"] for g in gts]); gc = np.array([g["cls"] for g in gts])
                keep, by = nms_track(raw, thr, agn)
                post = [raw[i] for i in keep]
                _, types, tgt, gstate = classify_dets(gts, post)
                M = iou_matrix([d["hbb"] for d in raw], G)
                # 원시 출력에서 정답 클래스·IoU ≥ 0.5인 상자(Pre-NMS TP)
                ptp = {i: int(M[i].argmax()) for i in range(len(raw)) if M[i].max() >= 0.5 and gc[M[i].argmax()] == raw[i]["cls"] and raw[i]["score"] >= 0.05}
                pre_tp += len(ptp)
                for i, j in ptp.items():
                    if i in by and raw[by[i]]["cls"] != raw[i]["cls"]:
                        cc_supp += 1
                tp_post_g = {tgt[k] for k, ty in enumerate(types) if ty == "TP"}
                for j, st in enumerate(gstate):
                    if j in tp_post_g:
                        continue
                    misses += 1
                    cands = [i for i, jj in ptp.items() if jj == j]
                    # kill: 후보가 있었는데 모두 지워졌고, 지운 상자 중 하나 이상이 다른 정답을 맞힌(또는 노린) 상자
                    if cands and all(i in by for i in cands):
                        sup = [by[i] for i in cands]
                        if any(M[s].max() >= 0.5 and int(M[s].argmax()) != j for s in sup):
                            kills += 1
            out[f"{'agnostic' if agn else 'classwise'}_{thr}"] = dict(misses=misses, kills=kills, kill_ratio=r4(kills / max(misses, 1)),
                                                                      pre_nms_tp=pre_tp, ccsr=r4(cc_supp / max(pre_tp, 1)))
    # 밀집 계류 장면: 45°로 나란히 붙여 맨 소형선박 줄(옆 배와 간격 약 1 m) 6장 — 이웃 정답끼리 HBB가 크게 겹침
    from make_part6 import obb_corners
    rng = np.random.default_rng(81)
    dense = []
    for k in range(6):
        objs = []
        for row in range(3):
            for i in range(8):
                L = rng.uniform(12, 16) / 0.5; W = L * 0.32
                th = 45 + rng.normal(0, 2)
                step = W + 2.0                                   # 폭 + 1 m
                cx = 120 + row * 130 + i * step / math.sqrt(2) + rng.normal(0, 0.5)
                cy = 110 + i * step / math.sqrt(2) + row * 60 + rng.normal(0, 0.5)
                o = dict(cls=1, cx=cx, cy=cy, w=L, h=W, theta=th)
                o["obb"] = [cx, cy, L, W, th]; o["hbb"] = obb_to_hbb((cx, cy, L, W, th)).tolist()
                hb = o["hbb"]; o["area"] = (hb[2] - hb[0]) * (hb[3] - hb[1])
                objs.append(o)
        dense.append((objs, detect(objs, [], 1.0, seed=500 + k)))
    G0 = [np.array([o["hbb"] for o in objs]) for objs, _ in dense]
    nb = []
    for G in G0:
        M = iou_matrix(G, G); np.fill_diagonal(M, 0); nb += M.max(1).tolist()
    dd = dict(n_gt=sum(len(o) for o, _ in dense), neighbor_gt_iou_median=r4(np.median(nb)), neighbor_gt_iou_max=r4(np.max(nb)))
    from part6_eval import soft_nms_dets
    for thr in (0.3, 0.5, 0.7):
        kills = misses = 0
        for objs, raw in dense:
            keep, by = nms_track(raw, thr)
            post = [raw[i] for i in keep]
            _, types, tgt, gstate = classify_dets(objs, post)
            G = np.array([g["hbb"] for g in objs]); M = iou_matrix([d["hbb"] for d in raw], G)
            ptp = {i: int(M[i].argmax()) for i in range(len(raw)) if M[i].max() >= 0.5 and raw[i]["cls"] == 1 and raw[i]["score"] >= 0.05}
            tp_g = {tgt[q] for q, ty in enumerate(types) if ty == "TP"}
            for j in range(len(objs)):
                if j in tp_g:
                    continue
                misses += 1
                cands = [i for i, jj in ptp.items() if jj == j]
                if cands and all(i in by for i in cands) and any(M[by[i]].max() >= 0.5 and int(M[by[i]].argmax()) != j for i in cands):
                    kills += 1
        ap = map50([(o, [raw[i] for i in nms_track(raw, thr)[0]]) for o, raw in dense])
        dd[f"nms_{thr}"] = dict(misses=misses, kills=kills, kill_ratio=r4(kills / max(misses, 1)), recall=r4(1 - misses / dd["n_gt"]), ap50=ap)
    dd["soft_nms_ap50"] = map50([(o, soft_nms_dets(raw, 0.5)) for o, raw in dense])
    from part6_eval import nms_dets as _nd
    dd["rotated_nms_0.5_ap50"] = map50([(o, _nd(raw, 0.5, rotated=True)) for o, raw in dense])
    dd["note"] = "가상 밀집 계류 타일 6장(소형선박 24척씩, 45°로 나란히, 옆 배와 간격 약 1 m). 평가는 HBB mAP50"
    out["dense_mooring"] = dd
    # OFI: 정답마다 IoU ≥ 0.5인 원시 상자 중 점수 상위 K개(K = min(4, 개수), 2개 이상인 정답만)
    def ofi(use_logit, K=4, sharpen=1.0):
        Hs, n_used = [], 0
        for t, raw in zip(T6, RAW6):
            G = np.array([g["hbb"] for g in t["objs"]])
            if not raw:
                continue
            M = iou_matrix([d["hbb"] for d in raw], G)
            for j in range(len(G)):
                idx = np.where(M[:, j] >= 0.5)[0]
                if len(idx) < 2:
                    continue
                s = np.sort(np.array([raw[i]["score"] for i in idx]))[::-1][:K]
                s = np.clip(s, 1e-4, 1 - 1e-4)
                z = (np.log(s / (1 - s)) if use_logit else s) * sharpen
                p = np.exp(z - z.max()); p /= p.sum()
                Hs.append(-(p * np.log(p + 1e-12)).sum() / np.log(len(s))); n_used += 1
        return r4(1 - np.mean(Hs)), n_used
    o_logit, n1 = ofi(True); o_prob, _ = ofi(False)
    # 이론 상한 예: K=4, 한 박스 점수 1·나머지 0(확률을 그대로 소프트맥스) / 로짓 4·−4
    p = np.exp(np.array([1, 0, 0, 0.0])); p /= p.sum(); cap_prob = 1 - (-(p * np.log(p)).sum()) / np.log(4)
    z = np.array([4, -4, -4, -4.0]); p = np.exp(z - z.max()); p /= p.sum(); cap_logit = 1 - (-(p * np.log(p)).sum()) / np.log(4)
    out["ofi"] = dict(logit=o_logit, prob=o_prob, n_gt_used=n1, cap_prob_K4=r4(cap_prob), cap_logit_K4_pm4=r4(cap_logit),
                      note="정답마다 IoU ≥ 0.5인 NMS 전 상자 점수 상위 K개(2개 이상인 정답만, K = min(4, 개수), 엔트로피를 ln K로 나눔). 가상 탐지기의 점수를 로짓 ln(s/(1−s))로 바꿔 계산")
    out["note"] = ("kill = 놓친 정답 가운데, NMS 전에는 정답 클래스·IoU ≥ 0.5 상자(점수 ≥ 0.05)가 있었는데 모두 지워졌고 지운 상자 중 하나가 다른 정답을 노린(IoU ≥ 0.5) 경우. "
                   "kill_ratio = kill / 놓친 정답. CCSR = 다른 클래스 상자에 지워진 Pre-NMS TP / 전체 Pre-NMS TP")
    R["nmsdiag"] = out


def exp_segerr():
    from make_part7 import scenes
    S = scenes()
    sys.path.insert(0, HERE)
    m = TinyUNet(skip=True)
    m.load_state_dict(torch.load(os.path.join(HERE, ".part7_unet.pt"))); m.eval()

    def infer(x):
        H = x.shape[1]; out = torch.zeros(6, H, H); cnt = torch.zeros(1, H, H)
        with torch.no_grad():
            for r in range(0, H - 31, 16):
                for c in range(0, H - 31, 16):
                    out[:, r:r + 32, c:c + 32] += m(x[None, :, r:r + 32, c:c + 32])[0].softmax(0); cnt[:, r:r + 32, c:c + 32] += 1
        return (out / cnt).argmax(0).numpy()
    per = {c: dict(err=0, bnd=0) for c in range(6)}
    small = dict(n=0, lost=0); merges = splits = 0; n_gt_comp = n_pred_comp = 0; holes_pred = holes_gt = 0
    eps = 1
    for k in (8, 9):
        y = S[k]["m"]; p = infer(to_tensor(S[k]["x"][None])[0])
        for c in range(6):
            Yc, Pc = y == c, p == c
            E = Yc != Pc
            band = ndi.binary_dilation(Yc, iterations=eps) & ~ndi.binary_erosion(Yc, iterations=eps)
            per[c]["err"] += int(E.sum()); per[c]["bnd"] += int((E & band).sum())
            lg, ng = ndi.label(Yc); lp, npd = ndi.label(Pc)
            n_gt_comp += ng; n_pred_comp += npd
            for i in range(1, ng + 1):
                comp = lg == i
                a = comp.sum()
                if a < 100:                               # 소형 영역: 100화소(1 ha) 미만
                    small["n"] += 1; small["lost"] += int(not (comp & Pc).any())
                ids = np.unique(lp[comp & Pc]); ids = ids[ids > 0]
                big = [q for q in ids if ((lp == q) & comp).sum() >= 20]
                splits += len(big) >= 2
            for q in range(1, npd + 1):
                comp = lp == q
                ids = np.unique(lg[comp & Yc]); ids = ids[ids > 0]
                big = [i for i in ids if ((lg == i) & comp).sum() >= 20]
                merges += len(big) >= 2
            holes_pred += int((ndi.binary_fill_holes(Pc) & ~Pc).sum()); holes_gt += int((ndi.binary_fill_holes(Yc) & ~Yc).sum())
    ber = {CLS5[c]: r4(per[c]["bnd"] / max(per[c]["err"], 1)) for c in range(6)}
    tot_e = sum(v["err"] for v in per.values()); tot_b = sum(v["bnd"] for v in per.values())
    R["segerr"] = dict(ber_per_class=ber, ber_all=r4(tot_b / tot_e), err_pixels=tot_e, small_regions=small,
                       small_loss_rate=r4(small["lost"] / max(small["n"], 1)), merges=merges, splits=splits,
                       n_gt_components=n_gt_comp, n_pred_components=n_pred_comp, hole_pixels_pred=holes_pred, hole_pixels_gt=holes_gt,
                       note=("7부 U-Net(겹쳐 평균 추론), 시험 장면 8·9. BER_c = 경계 띠(정답 영역을 1화소 팽창 − 1화소 침식) 안의 틀린 화소 / 클래스 c의 틀린 화소. "
                             "소형 영역 = 정답 연결 요소 100화소(1 ha) 미만, 소실 = 예측과 한 화소도 안 겹침. 병합 = 예측 덩어리 하나가 같은 클래스 정답 덩어리 둘 이상과 20화소 이상씩 겹침, "
                             "쪼개짐 = 정답 덩어리 하나가 예측 덩어리 둘 이상으로. 구멍 = 덩어리 안 메워진 빈 곳의 화소 수"))


# ================================================================== 54 슬라이스·라벨·분포
def gap_features(m, x, bs=300):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(m.features(x[i:i + bs]).mean(dim=(2, 3)))
    return torch.cat(out)


def bh(pvals, q=0.1):
    """벤저미니-호크버그: 기각되는 가설 인덱스"""
    p = np.asarray(pvals); M = len(p); o = np.argsort(p)
    ok = np.where(p[o] <= (np.arange(1, M + 1) / M) * q)[0]
    return set(o[: ok.max() + 1].tolist()) if len(ok) else set()


def exp_slice():
    m = base_model()
    raw = P5.load()
    out = {}
    for split in ("test", "shift"):
        x, y = D[f"x_{split}"], D[f"y_{split}"]
        pred = logits_of(m, x).argmax(1)
        ok = (pred == y).numpy().astype(float)
        p_glob = ok.mean()
        xr = raw[f"x_{split}"]; mm = raw[f"m_{split}"]; reg = raw[f"r_{split}"]
        bright = xr[:, :3].mean(axis=(1, 2, 3)); ndvi = ((xr[:, 3] - xr[:, 2]) / (xr[:, 3] + xr[:, 2])).mean(axis=(1, 2))
        mixed = np.array([len(np.unique(a)) > 1 for a in mm])
        cloud = xr[:, :3].max(axis=(1, 2, 3)) > np.percentile(xr[:, :3].max(axis=(1, 2, 3)), 90)
        sl = {}
        for c in range(6):
            sl[f"클래스={CLS5[c]}"] = (y.numpy() == c)
        for r in np.unique(reg):
            sl[f"지역={int(r)}"] = reg == r
        sl["경계 타일"] = mixed; sl["단일 클래스 타일"] = ~mixed
        t1, t2 = np.percentile(bright, [33.3, 66.7])
        sl["밝기 하위 1/3"] = bright < t1; sl["밝기 상위 1/3"] = bright >= t2
        sl["밝은 점 상위 10%(구름 의심)"] = cloud
        for c in range(6):                                   # 두 조건 조합 예: 클래스 × 경계 타일
            sl[f"클래스={CLS5[c]} & 경계 타일"] = (y.numpy() == c) & mixed
        rows, pv = [], []
        for name, mask in sl.items():
            n = int(mask.sum())
            if n < 10:
                continue
            ps = ok[mask].mean()
            z = (ps - p_glob) / math.sqrt(p_glob * (1 - p_glob) / n)
            p = stats.norm.cdf(z)                             # 한쪽: 슬라이스가 더 나쁨
            rows.append(dict(slice=name, n=n, acc=r4(ps), sdr=r4((p_glob - ps) / p_glob), z=r4(z), p=float(f"{p:.3g}")))
            pv.append(p)
        rej = bh(pv, 0.1)
        for i, r in enumerate(rows):
            r["bh_q0.1"] = i in rej
        rows.sort(key=lambda r: r["z"])
        out[split] = dict(global_acc=r4(p_glob), n_slices=len(rows), n_reject_bh=len(rej), n_p_lt_0_05=int(sum(p < 0.05 for p in pv)),
                          min_slice=rows[0]["slice"], min_slice_acc=rows[0]["acc"], worst=rows[:8])
    # 라벨 잡음 격리: 학습 라벨 10% 대칭 잡음 → 2겹 교차 예측 확률로 신뢰 학습(confident learning)
    rng = np.random.default_rng(1)
    y_true = D["y_train"].clone()
    y_noisy = y_true.clone()
    flip = torch.tensor(rng.random(len(y_true)) < 0.10)
    new = torch.tensor(rng.integers(0, 5, len(y_true))); new = new + (new >= y_true).long()
    y_noisy[flip] = new[flip]
    probs = torch.zeros(len(y_true), 6)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(y_true), generator=g); half = len(perm) // 2
    for a, b in ((perm[:half], perm[half:]), (perm[half:], perm[:half])):
        torch.manual_seed(0)
        mm_ = TinyCNN()
        fit(mm_, D["x_train"][a], y_noisy[a], D["x_val"], D["y_val"], epochs=20)
        probs[b] = logits_of(mm_, D["x_train"][b]).softmax(1)
    yn = y_noisy.numpy(); P = probs.numpy()
    t = np.array([P[yn == j, j].mean() for j in range(6)])
    flag = np.zeros(len(yn), bool)
    for i in range(len(yn)):
        above = [j for j in range(6) if P[i, j] >= t[j]]
        if above:
            jbest = max(above, key=lambda j: P[i, j])
            flag[i] = jbest != yn[i]
    fl = flip.numpy()
    out["label_noise"] = dict(n=len(yn), n_flipped=int(fl.sum()), n_flagged=int(flag.sum()), precision=r4((flag & fl).sum() / max(flag.sum(), 1)),
                              recall=r4((flag & fl).sum() / fl.sum()), thresholds=[r4(v) for v in t],
                              simple_disagree=dict(n=int((P.argmax(1) != yn).sum()), precision=r4(((P.argmax(1) != yn) & fl).sum() / max((P.argmax(1) != yn).sum(), 1)),
                                                   recall=r4(((P.argmax(1) != yn) & fl).sum() / fl.sum())),
                              note="5부 학습 라벨에 10% 대칭 잡음. 2겹 교차 예측(각 반을 다른 반으로 학습한 VD-CNN, 20에폭)의 확률로 신뢰 학습의 클래스별 문턱 t_j(주어진 라벨이 j인 표본의 평균 자기 확신도)를 적용")
    # 분포 거리: 밴드 평균의 W1, VD-CNN 특징의 MMD²(RBF, 중앙값 대역폭), 순열 검정
    def tile_means(split):
        return raw[f"x_{split}"].mean(axis=(2, 3))
    w1 = {}
    for split in ("val", "test", "shift"):
        w1[split] = [r4(stats.wasserstein_distance(tile_means("train")[:, b], tile_means(split)[:, b])) for b in range(4)]
    F_tr = gap_features(m, D["x_train"][:400]).numpy(); rs = np.random.default_rng(0)
    def mmd2(A, B, sig):
        def k(X, Y):
            d = ((X[:, None] - Y[None]) ** 2).sum(-1); return np.exp(-d / (2 * sig ** 2))
        return k(A, A).mean() - 2 * k(A, B).mean() + k(B, B).mean()
    allf = np.concatenate([F_tr, gap_features(m, D["x_test"][:400]).numpy()])
    sig = float(np.median(np.sqrt(((allf[:200, None] - allf[None, :200]) ** 2).sum(-1))))
    mm = {}
    for split in ("val", "test", "shift"):
        Fb = gap_features(m, D[f"x_{split}"][:400]).numpy()
        v = mmd2(F_tr, Fb, sig)
        Z = np.concatenate([F_tr, Fb]); cnt = 0
        for _ in range(100):
            p = rs.permutation(len(Z)); cnt += mmd2(Z[p[:400]], Z[p[400:]], sig) >= v
        mm[split] = dict(mmd2=r4(v), perm_p=r4((cnt + 1) / 101))
    out["drift"] = dict(w1_band_means=w1, w1_bands=["B2", "B3", "B4", "B8"], mmd=mm, sigma=r4(sig),
                        note="W1 = 타일별 밴드 평균 반사율 분포의 1차 와서스타인 거리(학습 4,200장 대비). MMD² = VD-CNN 전역 평균 풀링 특징(64차원) 400장씩, RBF 커널(대역폭 = 거리 중앙값), 순열 100회")
    # 탐지 슬라이스: 타일 종류별 mAP50
    det = {}
    for kind in ("harbor", "sea", "marina"):
        imgs = [b for t, b in zip(T6, BASE6) if t["kind"] == kind]
        det[kind] = dict(n_tiles=len(imgs), n_gt=sum(len(g) for g, _ in imgs), map50=map50(imgs))
    out["detection_slices"] = det
    R["slice"] = out


# ================================================================== 55 보정·불확실성
def ece(probs, y, bins=15):
    conf, pred = probs.max(1); acc = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1); e = 0.0; mce = 0.0; rows = []
    for i in range(bins):
        msk = (conf > edges[i]) & (conf <= edges[i + 1])
        if msk.any():
            gap = abs(acc[msk].mean().item() - conf[msk].mean().item())
            e += msk.float().mean().item() * gap; mce = max(mce, gap)
            rows.append([r4(edges[i]), r4(edges[i + 1]), int(msk.sum()), r4(conf[msk].mean()), r4(acc[msk].mean())])
    return e, mce, rows


def fit_temperature(logits, y):
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(logits / t.exp(), y); loss.backward(); return loss
    opt.step(closure)
    return float(t.exp().detach())


def auroc(score, label):
    from sklearn.metrics import roc_auc_score
    return r4(roc_auc_score(label, score))


def exp_calib():
    m = base_model()
    out = {}
    Lv = logits_of(m, D["x_val"])
    T = fit_temperature(Lv, D["y_val"])
    for split in ("test", "shift"):
        L = logits_of(m, D[f"x_{split}"]); y = D[f"y_{split}"]
        e0, m0, rows0 = ece(L.softmax(1), y); e1, m1, rows1 = ece((L / T).softmax(1), y)
        out[split] = dict(acc=r4((L.argmax(1) == y).float().mean()), ece=r4(e0), mce=r4(m0), nll=r4(F.cross_entropy(L, y)),
                          ece_T=r4(e1), mce_T=r4(m1), nll_T=r4(F.cross_entropy(L / T, y)), mean_conf=r4(L.softmax(1).max(1).values.mean()),
                          bins=rows0, bins_T=rows1)
    out["temperature"] = r4(T)
    # MC 드롭아웃: 드롭아웃 0.3 VD-CNN을 새로 학습(30에폭), 추론 때 드롭아웃만 켜고 30번
    torch.manual_seed(0)
    md = TinyCNN(dropout=0.3)
    res = fit(md, D["x_train"], D["y_train"], D["x_val"], D["y_val"], {"test": (D["x_test"], D["y_test"]), "shift": (D["x_shift"], D["y_shift"])}, epochs=30)
    out["dropout_model"] = dict(test_best=res["best"]["test"], shift_best=res["best"]["shift"], best_epoch=res["best_epoch"])
    torch.save(md.state_dict(), os.path.join(HERE, ".part8_dropout.pt"))
    def mc(x, Tn=30):
        md.eval()
        for mod in md.modules():
            if isinstance(mod, nn.Dropout):
                mod.train()
        torch.manual_seed(1)
        with torch.no_grad():
            P = torch.stack([md(x).softmax(1) for _ in range(Tn)])
        md.eval()
        pbar = P.mean(0)
        tot = -(pbar * (pbar + 1e-12).log()).sum(1)
        ale = (-(P * (P + 1e-12).log()).sum(2)).mean(0)
        return pbar, tot, ale, tot - ale
    U = {}
    for split in ("test", "shift"):
        pbar, tot, ale, epi = mc(D[f"x_{split}"]); y = D[f"y_{split}"]
        wrong = (pbar.argmax(1) != y).numpy()
        U[split] = dict(acc=r4((pbar.argmax(1) == y).float().mean()), total=r4(tot.mean()), aleatoric=r4(ale.mean()), epistemic=r4(epi.mean()),
                        epistemic_correct=r4(epi[~torch.tensor(wrong)].mean()), epistemic_wrong=r4(epi[torch.tensor(wrong)].mean()) if wrong.any() else None,
                        auroc_error_by_epistemic=auroc(epi.numpy(), wrong), auroc_error_by_total=auroc(tot.numpy(), wrong),
                        auroc_error_by_maxprob=auroc(-pbar.max(1).values.numpy(), wrong), ece_mc=r4(ece(pbar, y)[0]))
        U[split]["_epi"] = epi.numpy()
    lab = np.r_[np.zeros(len(U["test"]["_epi"])), np.ones(len(U["shift"]["_epi"]))]
    U["auroc_shift_vs_test_by_epistemic"] = auroc(np.r_[U["test"]["_epi"], U["shift"]["_epi"]], lab)
    for s in ("test", "shift"):
        U[s].pop("_epi")
    out["mc_dropout"] = U
    out["note"] = "VD-CNN(best). ECE는 15칸 등간격, 온도 T는 검증 900장의 NLL 최소화(LBFGS). MC 드롭아웃은 드롭아웃 0.3 VD-CNN(새로 학습) 30회 샘플, 엔트로피는 자연로그(최대 ln 6 = 1.79)"
    R["calib"] = out


# ================================================================== 56 표현
def eff_rank(A):
    s = np.linalg.svd(A - A.mean(0), compute_uv=False)
    p = s / s.sum(); H = -(p * np.log(p + 1e-12)).sum()
    return float(np.exp(H))


def nc1(Fe, y):
    mu = Fe.mean(0); K = int(y.max()) + 1
    Sw = np.zeros((Fe.shape[1],) * 2); Sb = np.zeros_like(Sw)
    for c in range(K):
        Fc = Fe[y == c]; mc = Fc.mean(0)
        Sw += (Fc - mc).T @ (Fc - mc) / len(Fe)
        Sb += np.outer(mc - mu, mc - mu) / K
    return float(np.trace(Sw @ np.linalg.pinv(Sb, rcond=1e-6, hermitian=True)) / K)   # Σ_B의 계수는 K−1


def dcr(m, x, eps=1e-4):
    """합성곱 블록마다 ReLU 뒤 채널 평균 |활성|이 eps 미만인 비율"""
    acts = []
    hooks = [mod.register_forward_hook(lambda mm, i, o: acts.append(o.detach())) for mod in m.features if isinstance(mod, nn.ReLU)]
    m.eval()
    with torch.no_grad():
        m(x)
    for h in hooks:
        h.remove()
    return [r4((a.abs().mean(dim=(0, 2, 3)) < eps).float().mean()) for a in acts]


def exp_repr():
    out = {}
    m = base_model()
    Ftr = gap_features(m, D["x_train"]).numpy(); Fte = gap_features(m, D["x_test"]).numpy()
    er = eff_rank(Fte)
    out["base"] = dict(eff_rank=r4(er), cur=r4(er / 64), dcr=dcr(m, D["x_test"][:500]), nc1_train=r4(nc1(Ftr, D["y_train"].numpy())),
                       nc1_test=r4(nc1(Fte, D["y_test"].numpy())))
    # 대조: 정규화 없이 큰 학습률(Adam 0.02)로 10에폭 — 죽은 ReLU 채널이 생기는지
    torch.manual_seed(0)
    md = TinyCNN(norm="none")
    res = fit(md, D["x_train"], D["y_train"], D["x_val"], D["y_val"], {"test": (D["x_test"], D["y_test"])}, epochs=10, lr=0.02)
    Fd = gap_features(md, D["x_test"]).numpy(); erd = eff_rank(Fd)
    out["no_bn_lr0.02"] = dict(test_best=res["best"]["test"], eff_rank=r4(erd), cur=r4(erd / 64), dcr=dcr(md, D["x_test"][:500]),
                               nc1_test=r4(nc1(Fd, D["y_test"].numpy())))
    # 유효 수용영역: 마지막 합성곱 출력(8×8)의 가운데 칸 (4,4) 값 합을 입력으로 미분, 시험 300장 평균 |기울기|
    m.eval()
    x = D["x_test"][:300].clone().requires_grad_(True)
    f = m.features(x)
    f[:, :, 4, 4].sum().backward()
    g = x.grad.abs().sum(dim=(0, 1)).numpy()
    cy, cx = np.unravel_index(g.argmax(), g.shape)
    yy, xx = np.mgrid[0:32, 0:32]
    cyc, cxc = (g * yy).sum() / g.sum(), (g * xx).sum() / g.sum()
    rr = np.sqrt((yy - cyc) ** 2 + (xx - cxc) ** 2)
    order = np.argsort(rr.ravel()); cum = np.cumsum(g.ravel()[order]) / g.sum()
    r95 = float(rr.ravel()[order][np.searchsorted(cum, 0.95)])
    r50 = float(rr.ravel()[order][np.searchsorted(cum, 0.50)])
    out["erf"] = dict(center=[r4(cyc), r4(cxc)], r50=r4(r50), r95=r4(r95), theoretical_rf=18, nonzero_extent=[int((g.sum(1) > 0).sum()), int((g.sum(0) > 0).sum())],
                      rfm_examples={name: r4(math.log(2 * r95 / S)) for name, S in [("차량 9×4(대각 9.8)", 9.85), ("소형선박 24×8(대각 25.3)", 25.3), ("타일 32×32(대각 45.3)", 45.25), ("선박 200×34(대각 202.9)", 202.9)]},
                      grad_map=[[r4(v) for v in row] for row in (g / g.max())[::2, ::2]])
    # ViT 어텐션 엔트로피: 5부 작은 ViT를 15에폭 학습해 시험 200장
    torch.manual_seed(0)
    vit = TinyViT()
    resv = fit(vit, D["x_train"], D["y_train"], D["x_val"], D["y_val"], {"test": (D["x_test"], D["y_test"])}, epochs=15, opt="adamw", lr=1e-3, wd=0.05)
    vit.eval()
    ents = []
    with torch.no_grad():
        t = vit.embed(D["x_test"][:200]).flatten(2).transpose(1, 2)
        t = torch.cat([vit.cls.expand(len(t), -1, -1), t], 1) + vit.pos
        for layer in vit.enc.layers:
            h = layer.norm1(t)
            _, A = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=False)
            Hh = -(A * (A + 1e-12).log()).sum(-1).mean(dim=(0, 2)) / math.log(A.shape[-1])
            ents.append([r4(v) for v in Hh])
            t = layer(t)
    out["vit_attention_entropy"] = dict(test_acc_best=resv["best"]["test"], normalized_entropy_by_layer_head=ents, n_tokens=65,
                                        note="엔트로피를 ln 65로 나눈 값(1 = 모든 토큰에 고르게, 0 = 한 토큰에만). 작은 ViT 15에폭(5부 32강 구조)")
    out["note"] = ("유효 랭크 = exp(특잇값 비율의 엔트로피), CUR = 유효 랭크 / 채널 수(64). DCR = 채널 평균 |활성| < 1e-4 비율(합성곱 블록 3개의 ReLU 뒤). "
                   "NC1 = tr(Σ_W Σ_B⁺)/K. ERF = 마지막 합성곱 가운데 칸의 입력 기울기 크기, R95 = 기울기 질량 95%를 담는 반지름(화소). RFM = ln(2·R95 / 객체 대각선)")
    R["repr"] = out


# ================================================================== 57 환경 강건성
NUIS = ["BC", "IV", "CM", "TC", "OCC", "LR", "DEF", "OV", "MB", "ATM"]
NUIS_KO = {"BC": "배경 클러터", "IV": "조도 급변", "CM": "위장(대비 저하)", "TC": "열적 크로스오버·경면 반사", "OCC": "가림·중첩",
           "LR": "극저해상도", "DEF": "사각 왜곡", "OV": "경계 절단", "MB": "모션 블러", "ATM": "대기 산란"}


def corrupt(x, k, s, seed=0):
    """x: (N,4,32,32) 반사율 numpy. 교란 k, 심각도 s(1~5). 같은 seed면 같은 교란"""
    rng = np.random.default_rng(seed * 100 + s)
    x = x.copy(); N = len(x); i = s - 1
    if k == "BC":
        a = [0.05, 0.1, 0.15, 0.22, 0.3][i]
        n = ndi.gaussian_filter(rng.normal(0, 1, (N, 1, 32, 32)), (0, 0, 0.7, 0.7)); n /= n.std()
        x = x * (1 + a * n)
    elif k == "IV":
        g = [0.1, 0.2, 0.3, 0.45, 0.6][i]
        th = rng.uniform(0, 2 * np.pi, N); yy, xx = np.mgrid[0:32, 0:32] / 31 - 0.5
        ramp = np.cos(th)[:, None, None] * xx + np.sin(th)[:, None, None] * yy
        x = x * (1 + 2 * g * ramp)[:, None]
    elif k == "CM":
        a = [0.1, 0.2, 0.3, 0.4, 0.5][i]
        other = P5.MEAN[rng.integers(0, 6, N)][:, :, None, None]
        x = (1 - a) * x + a * other
    elif k == "TC":
        n = [2, 4, 8, 12, 20][i]
        for j in range(N):
            for _ in range(n):
                r, c = rng.integers(1, 31, 2); x[j, :, r - 1:r + 1, c - 1:c + 1] += 0.3
    elif k == "OCC":
        f = [0.1, 0.2, 0.3, 0.4, 0.5][i]; side = int(round(32 * math.sqrt(f)))
        for j in range(N):
            r, c = rng.integers(0, 32 - side + 1, 2); x[j, :, r:r + side, c:c + side] = 0.4 + rng.normal(0, 0.01)
    elif k == "LR":
        f = [1.5, 2, 3, 4, 6][i]
        small = ndi.zoom(x, (1, 1, 1 / f, 1 / f), order=1, mode="nearest")      # mode 기본값(constant)은 배율 2에서 끝 행·열을 0으로 채움
        x = ndi.zoom(small, (1, 1, 32 / small.shape[2], 32 / small.shape[3]), order=1, mode="nearest")[:, :, :32, :32]
    elif k == "DEF":
        sh = [0.1, 0.2, 0.3, 0.4, 0.5][i]
        for j in range(N):
            M = np.array([[1, sh * rng.choice([-1, 1])], [0, 1 + sh / 2]])
            off = np.array([16, 16]) - M @ np.array([16, 16])
            for b in range(4):
                x[j, b] = ndi.affine_transform(x[j, b], M, offset=off, order=1, mode="reflect")
    elif k == "OV":
        d = [3, 6, 9, 12, 16][i]
        donor = x[rng.permutation(N)].copy()
        for j in range(N):
            side = rng.integers(0, 4)
            if side == 0: x[j, :, :d] = donor[j, :, :d]
            elif side == 1: x[j, :, -d:] = donor[j, :, -d:]
            elif side == 2: x[j, :, :, :d] = donor[j, :, :, :d]
            else: x[j, :, :, -d:] = donor[j, :, :, -d:]
    elif k == "MB":
        L = [3, 5, 7, 9, 13][i]
        for j in range(N):
            ker = np.zeros((L, L)); th = rng.uniform(0, np.pi)
            for t in np.linspace(-(L - 1) / 2, (L - 1) / 2, 4 * L):
                ker[int(round((L - 1) / 2 + t * np.sin(th))), int(round((L - 1) / 2 + t * np.cos(th)))] = 1
            ker /= ker.sum()
            for b in range(4):
                x[j, b] = ndi.convolve(x[j, b], ker, mode="reflect")
    elif k == "ATM":
        t = [0.9, 0.8, 0.7, 0.6, 0.5][i]
        A = np.array([0.12, 0.10, 0.08, 0.05], np.float32)[None, :, None, None]
        x = x * t + A * (1 - t)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def robust_eval(models, xs, ys):
    res = {name: {"clean": r4(evaluate(m, to_tensor(xs), ys)[1])} for name, m in models.items()}
    for k in NUIS:
        for name in models:
            res[name][k] = []
        for s in range(1, 6):
            xc = to_tensor(corrupt(xs, k, s))
            for name, m in models.items():
                res[name][k].append(r4(evaluate(m, xc, ys)[1]))
    return res


def rdi_mce(res, name, base="flat_mlp"):
    clean = res[name]["clean"]
    rdi = {k: r4((clean - np.mean(res[name][k])) / clean) for k in NUIS}
    ce = {k: r4(sum(1 - a for a in res[name][k]) / sum(1 - a for a in res[base][k])) for k in NUIS}
    return rdi, ce, r4(100 * np.mean(list(ce.values())))


def exp_robust():
    raw = P5.load()
    xs, ys = raw["x_test"], D["y_test"]
    m = base_model()
    torch.manual_seed(0)
    flat = FlatMLP()
    fit(flat, D["x_train"], D["y_train"], D["x_val"], D["y_val"], epochs=30)
    torch.manual_seed(0)
    aug = TinyCNN()
    resa = fit(aug, D["x_train"], D["y_train"], D["x_val"], D["y_val"], epochs=30, aug="geo_photo")
    torch.save(aug.state_dict(), os.path.join(HERE, ".part8_aug.pt"))
    models = {"vdcnn": m, "vdcnn_aug": aug, "flat_mlp": flat}
    res = robust_eval(models, xs, ys)
    out = {"acc": res, "nuisances": NUIS_KO}
    for name in ("vdcnn", "vdcnn_aug"):
        rdi, ce, mce = rdi_mce(res, name)
        out[name] = dict(rdi=rdi, ce=ce, mce=mce, max_rdi=max(rdi.values()), worst=max(rdi, key=rdi.get),
                         n_robust=sum(v < 0.15 for v in rdi.values()), n_critical=sum(v >= 0.35 for v in rdi.values()))
    out["note"] = ("5부 시험 타일 900장에 10대 교란을 5단계로 넣음(교란 구현은 corrupt). 지표는 mAP 대신 분류 정확도. "
                   "RDI_k = (클린 − 5단계 평균)/클린. CE_k = Σ(1 − 정확도) / 기준 모델의 Σ(1 − 정확도), 기준 모델 = 화소 펼침 MLP(5부, 새로 30에폭). mCE = 100 × CE 평균. "
                   "vdcnn_aug = 기하+광학 증강(45강)으로 새로 학습한 VD-CNN 30에폭")
    R["robust"] = out


# ================================================================== 58 다종 센서 융합
class PatchNet(nn.Module):
    def __init__(self, cin, k=9):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, k))

    def forward(self, x):
        return self.net(x)


def exp_fusion():
    Z = np.load(os.path.join(HERE, "part4_scene.npz"))
    refl = Z["refl"].astype(np.float32); sar = np.log10(Z["sar"].astype(np.float32) + 1e-4)[None]; cls = Z["cls"].astype(np.int64)
    eo_mu, eo_sd = refl.mean((1, 2), keepdims=True), refl.std((1, 2), keepdims=True)
    sa_mu, sa_sd = sar.mean(), sar.std()
    EO = (refl - eo_mu) / eo_sd; SA = (sar - sa_mu) / sa_sd
    H = 300; r = 4
    rng = np.random.default_rng(0)
    def sample(rows, n):
        ys = rng.integers(rows[0] + r, rows[1] - r, n); xs = rng.integers(r, H - r, n)
        return ys, xs
    def patches(arr, ys, xs, dy=0, dx=0):
        out = np.stack([arr[:, y - r + dy:y + r + 1 + dy, x - r + dx:x + r + 1 + dx] for y, x in zip(ys, xs)])
        return torch.tensor(out)
    tr_y, tr_x = sample((0, 180), 12000)
    te_y, te_x = sample((196, 300), 5000)
    te_y = np.clip(te_y, 196 + 8, 300 - 9 - r); te_x = np.clip(te_x, 8 + r, H - 9 - r)
    y_tr = torch.tensor(cls[tr_y, tr_x]); y_te = torch.tensor(cls[te_y, te_x])
    # 구름: 시험 영역에 덩어리 구름(광학만 가림)
    cm = ndi.gaussian_filter(rng.normal(0, 1, (H, H)), 12); cloud = cm > np.quantile(cm[196:], 0.6)
    EOc = EO.copy(); cval = (0.45 - eo_mu) / eo_sd
    EOc[:, cloud] = np.broadcast_to(cval, (6, H, H))[:, cloud]
    def X(eo, sa, ys, xs, dy=0, dx=0, drop=None):
        e = patches(eo, ys, xs); s_ = patches(sa, ys, xs, dy, dx)
        if drop == "EO": e = torch.zeros_like(e)
        if drop == "SAR": s_ = torch.zeros_like(s_)
        return torch.cat([e, s_], 1)
    Xtr = X(EO, SA, tr_y, tr_x)
    def train(cin_sel, moddrop=False):
        torch.manual_seed(0)
        net = PatchNet(len(cin_sel))
        opt = torch.optim.Adam(net.parameters(), 1e-3); g = torch.Generator().manual_seed(0)
        for ep in range(12):
            perm = torch.randperm(len(Xtr), generator=g)
            for i in range(0, len(perm), 128):
                b = perm[i:i + 128]; xb = Xtr[b][:, cin_sel].clone()
                if moddrop:
                    u = torch.rand(len(b), generator=g)
                    xb[u < 0.25, :6] = 0
                    xb[(u >= 0.25) & (u < 0.5), 6:] = 0
                loss = F.cross_entropy(net(xb), y_tr[b]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); return net
    def acc(net, x, sel):
        with torch.no_grad():
            return r4((net(x[:, sel]).argmax(1) == y_te).float().mean())
    EOs, ALL = list(range(6)), list(range(7))
    nets = {"eo": (train(EOs), EOs), "sar": (train([6]), [6]), "fusion": (train(ALL), ALL), "fusion_moddrop": (train(ALL, True), ALL)}
    tests = {"clear": X(EO, SA, te_y, te_x), "cloudy": X(EOc, SA, te_y, te_x), "eo_missing": X(EO, SA, te_y, te_x, drop="EO"),
             "sar_missing": X(EO, SA, te_y, te_x, drop="SAR"), "cloudy_sar_missing": X(EOc, SA, te_y, te_x, drop="SAR")}
    xm = X(EOc, SA, te_y, te_x).clone(); cc = torch.tensor(cloud[te_y, te_x])
    xm[cc, :6] = 0                                       # 구름 마스크: 중심 화소가 구름이면 광학 입력을 결측(0)으로 처리
    tests["cloudy_masked"] = xm
    res = {n: {t: acc(net, x, sel) for t, x in tests.items()} for n, (net, sel) in nets.items()}
    shifts = {}
    for n in ("fusion", "fusion_moddrop"):
        net, sel = nets[n]
        shifts[n] = {d: acc(net, X(EO, SA, te_y, te_x, dy=d, dx=d), sel) for d in (0, 1, 2, 4, 6, 8)}
    chance = r4(max(np.bincount(cls[196:].ravel(), minlength=9)) / cls[196:].size)
    # 섀플리(2명): v(S) = 그 센서만 넣고 나머지는 0으로 비운 융합 모델 정확도
    shap = {}
    for n in ("fusion", "fusion_moddrop"):
        net, sel = nets[n]
        for cond, base_x in (("clear", EO), ("cloudy", EOc)):
            vN = acc(net, X(base_x, SA, te_y, te_x), sel); vE = acc(net, X(base_x, SA, te_y, te_x, drop="SAR"), sel)
            vS = acc(net, X(base_x, SA, te_y, te_x, drop="EO"), sel); v0 = acc(net, torch.zeros_like(tests["clear"]), sel)
            phiE = 0.5 * (vE - v0) + 0.5 * (vN - vS); phiS = 0.5 * (vS - v0) + 0.5 * (vN - vE)
            drE = (vN - vS) / vN; drS = (vN - vE) / vN
            shap[f"{n}_{cond}"] = dict(v_all=vN, v_eo=vE, v_sar=vS, v_none=v0, phi_eo=r4(phiE), phi_sar=r4(phiS),
                                       mcr_eo=r4(100 * max(phiE, 0) / (max(phiE, 0) + max(phiS, 0))), dr_eo=r4(drE), dr_sar=r4(drS),
                                       mbr=r4(drE / drS) if abs(drS) > 1e-6 else None)
    R["fusion"] = dict(acc=res, misalignment=shifts, shapley=shap, chance_majority=chance, cloud_frac_test=r4(cloud[196:].mean()),
                       n_train=len(y_tr), n_test=len(y_te),
                       note=("4부 가상 연안 장면(광학 6밴드 + SAR 1채널 로그 강도), 화소 중심 9×9 패치로 9클래스 분류하는 작은 합성곱망 12에폭. "
                             "학습 행 0~179, 시험 행 196~299. 결측 = 그 센서 입력을 0(표준화 평균)으로. 구름 = 시험 영역 약 40%의 광학 밴드를 밝은 구름 값으로 덮음(SAR는 그대로). cloudy_masked = 구름 낀 화소(패치 중심 기준)는 광학을 결측으로 바꿔 넣음. "
                             "정렬 오차 = SAR 패치를 대각선으로 d화소 밀어 넣음. 모달리티 드롭아웃 = 학습 때 25%는 광학, 25%는 SAR를 0으로. "
                             "섀플리 v(S) = 그 센서 조합만 넣은 융합 모델 정확도. DR_m = (v_all − v_all∖m)/v_all, MBR = DR_EO / DR_SAR"))


# ================================================================== 59 인과·바로가기
CUE_POS = [(2, 3 + 5 * c) for c in range(6)]


def add_cue(x, pos_cls):
    x = x.copy()
    for j, c in enumerate(pos_cls):
        r, cc = CUE_POS[int(c)]
        x[j, :, r:r + 3, cc:cc + 3] = 0.35
    return x


def exp_causal():
    raw = P5.load(); rng = np.random.default_rng(7)
    ytr = raw["y_train"]; yte = raw["y_test"]
    pos_tr = np.where(rng.random(len(ytr)) < 0.9, ytr, rng.integers(0, 6, len(ytr)))
    xtr_cue = to_tensor(add_cue(raw["x_train"], pos_tr))
    pos_val = np.where(rng.random(len(raw["y_val"])) < 0.9, raw["y_val"], rng.integers(0, 6, len(raw["y_val"])))
    xval_cue = to_tensor(add_cue(raw["x_val"], pos_val))
    wrong = (yte + rng.integers(1, 6, len(yte))) % 6
    tests = {"clean": to_tensor(raw["x_test"]), "cue_aligned": to_tensor(add_cue(raw["x_test"], yte)), "cue_counter": to_tensor(add_cue(raw["x_test"], wrong))}
    out = {}
    for name in ("shortcut", "cue_randomized"):
        torch.manual_seed(0)
        m = TinyCNN()
        if name == "shortcut":
            fit(m, xtr_cue, torch.tensor(ytr), xval_cue, D["y_val"], epochs=30)
        else:                                           # 처방: 단서 위치를 라벨과 무관하게(개입 do(단서) 증강)
            pos_r = rng.integers(0, 6, len(ytr))
            fit(m, to_tensor(add_cue(raw["x_train"], pos_r)), torch.tensor(ytr), to_tensor(add_cue(raw["x_val"], rng.integers(0, 6, len(raw["y_val"])))), D["y_val"], epochs=30)
        r = {}
        P = {}
        for t, x in tests.items():
            L = logits_of(m, x); P[t] = L.softmax(1)
            r[f"acc_{t}"] = r4((L.argmax(1) == D["y_test"]).float().mean())
        pc = P["cue_counter"].argmax(1).numpy()
        r["flip_to_cue_rate"] = r4((pc == wrong).mean())
        yt = torch.tensor(yte)
        r["ace_true_class_conf"] = r4((P["cue_aligned"][torch.arange(len(yt)), yt] - P["cue_counter"][torch.arange(len(yt)), yt]).mean())
        r["changed_pred_rate"] = r4((P["cue_aligned"].argmax(1) != P["cue_counter"].argmax(1)).float().mean())
        out[name] = r
    out["note"] = ("바로가기 단서 = 타일 위쪽 가장자리의 밝은 3×3 표식(반사율 0.35). 학습 타일의 90%는 표식 위치가 클래스마다 정해진 자리(6곳), 10%는 무작위. "
                   "시험: clean = 표식 없음, cue_aligned = 맞는 자리, cue_counter = 틀린 클래스 자리(반사실 개입 do(단서 = 다른 값)). "
                   "ACE = 정답 클래스 확신도의 (맞는 단서 − 틀린 단서) 평균, flip_to_cue_rate = 틀린 단서 클래스로 예측한 비율. "
                   "처방 모델 cue_randomized = 학습 때 표식 위치를 라벨과 무관하게 무작위로 둠")
    R["causal"] = out


# ================================================================== 60 하드웨어·양자화
def qparams_asym(x):
    lo, hi = min(float(x.min()), 0.0), max(float(x.max()), 0.0)
    S = (hi - lo) / 255 if hi > lo else 1.0
    Z = round(-128 - lo / S)
    return S, Z


def fq_asym(x, S, Z):
    return (torch.clamp(torch.round(x / S) + Z, -128, 127) - Z) * S


def fq_w_perchannel(w):
    m = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp_min(1e-12)
    S = m / 127
    return torch.clamp(torch.round(w / S), -127, 127) * S


def sqnr(a, b):
    return float(10 * torch.log10((a ** 2).mean() / ((a - b) ** 2).mean().clamp_min(1e-20)))


def layers_of(m):
    return [mod for mod in list(m.features) + list(m.head) if isinstance(mod, (nn.Conv2d, nn.Linear))]


def quant_run(m, xcal, xtest):
    """합성곱·선형 층마다 입력 활성(층별 비대칭 per-tensor, 보정 자료의 최솟값~최댓값)과 가중치(채널별 대칭)를 int8로 가짜 양자화"""
    Ls = layers_of(m); ins = {}
    hooks = [L.register_forward_pre_hook(lambda mod, inp, i=i: ins.setdefault(i, []).append(inp[0].detach())) for i, L in enumerate(Ls)]
    with torch.no_grad():
        m(xcal)
    for h in hooks:
        h.remove()
    qp = {i: qparams_asym(torch.cat(v)) for i, v in ins.items()}
    import copy
    mq = copy.deepcopy(m); mq.eval()
    for i, L in enumerate(layers_of(mq)):
        L.weight.data = fq_w_perchannel(L.weight.data)
        S, Z = qp[i]
        L.register_forward_pre_hook(lambda mod, inp, S=S, Z=Z: (fq_asym(inp[0], S, Z),))
    outs32, outsq = {}, {}
    for model, store in ((m, outs32), (mq, outsq)):
        hs = [L.register_forward_hook(lambda mod, inp, o, i=i, store=store: store.__setitem__(i, o.detach())) for i, L in enumerate(layers_of(model))]
        with torch.no_grad():
            model(xtest)
        for h in hs:
            h.remove()
    par = {}
    with torch.no_grad():
        acts = {}
        hs = [L.register_forward_pre_hook(lambda mod, inp, i=i: acts.__setitem__(i, inp[0].detach())) for i, L in enumerate(Ls)]
        m(xtest)
        for h in hs:
            h.remove()
    crr = {}
    for i, a in acts.items():
        dims = (0, 2, 3) if a.dim() == 4 else (0,)
        pc = a.abs().amax(dim=dims) / a.abs().mean(dim=dims).clamp_min(1e-12)
        par[i] = r4(pc.max())
        cm = a.abs().amax(dim=dims)
        crr[i] = r4(cm.max() / cm.median().clamp_min(1e-12))     # 채널 간 범위 비: 가장 넓은 채널 / 채널 중앙값
    quant_run.crr = crr
    return mq, {i: r4(sqnr(outs32[i], outsq[i])) for i in outs32}, par, qp


def exp_hw():
    out = {}
    m = base_model()
    xcal, xte = D["x_train"][:500], D["x_test"]
    mq, sq, par, qp = quant_run(m, xcal, xte)
    names = ["conv1", "conv2", "conv3", "fc"]
    out["base"] = dict(sqnr_db={names[i]: v for i, v in sq.items()}, par_max={names[i]: v for i, v in par.items()},
                       channel_range_ratio={names[i]: v for i, v in quant_run.crr.items()},
                       acc_fp32=r4(evaluate(m, xte, D["y_test"])[1]), acc_int8=r4(evaluate(mq, xte, D["y_test"])[1]),
                       act_scale={names[i]: r4(qp[i][0]) for i in qp})
    # 이상치 채널: conv2 입력의 한 채널(범위가 중간쯤인 채널)을 ×30, 그 채널 가중치 ÷30(fp32 출력은 같음) → per-tensor 양자화에 불리
    import copy
    mo = copy.deepcopy(m); conv1, bn1, conv2 = mo.features[0], mo.features[1], mo.features[4]
    acts0 = []
    h0 = m.features[4].register_forward_pre_hook(lambda mod, inp: acts0.append(inp[0].detach()))
    with torch.no_grad():
        m(xcal)
    h0.remove()
    cm0 = torch.cat(acts0).abs().amax(dim=(0, 2, 3))
    jj = int(torch.argsort(cm0)[len(cm0) // 2])                # 범위가 중간쯤인 채널 하나를 이상치로 만듦
    alpha = 30.0
    with torch.no_grad():
        bn1.weight[jj] *= alpha; bn1.bias[jj] *= alpha          # ReLU 앞 배치 정규화 출력 채널 jj를 30배
        conv2.weight[:, jj] /= alpha
    same = float((logits_of(mo, xte) - logits_of(m, xte)).abs().max())
    mqo, sqo, paro, _ = quant_run(mo, xcal, xte)
    crr_o = dict(quant_run.crr)
    # SmoothQuant식 이주: 채널별 s_j = max|X_j|^a / max|W_j|^(1-a), a = 0.5 → 활성 ÷ s, 가중치 × s
    ms = copy.deepcopy(mo)
    acts = []
    h = ms.features[4].register_forward_pre_hook(lambda mod, inp: acts.append(inp[0].detach()))
    with torch.no_grad():
        ms(xcal)
    h.remove()
    Xmax = torch.cat(acts).abs().amax(dim=(0, 2, 3)).clamp_min(1e-5)
    Wmax = ms.features[4].weight.abs().amax(dim=(0, 2, 3)).clamp_min(1e-5)
    sj = (Xmax ** 0.5) / (Wmax ** 0.5)
    with torch.no_grad():
        ms.features[1].weight /= sj; ms.features[1].bias /= sj   # ReLU는 양의 배율과 바뀌므로 배치 정규화 출력에서 나눔
        ms.features[4].weight *= sj[None, :, None, None]
    same2 = float((logits_of(ms, xte) - logits_of(m, xte)).abs().max())
    mqs, sqs, pars, _ = quant_run(ms, xcal, xte)
    out["outlier_channel"] = jj
    out["outlier"] = dict(fp32_max_logit_diff=float(f"{same:.2e}"), sqnr_db={names[i]: v for i, v in sqo.items()}, par_max={names[i]: v for i, v in paro.items()},
                          channel_range_ratio={names[i]: v for i, v in crr_o.items()},
                          acc_int8=r4(evaluate(mqo, xte, D["y_test"])[1]))
    out["smoothquant"] = dict(fp32_max_logit_diff=float(f"{same2:.2e}"), sqnr_db={names[i]: v for i, v in sqs.items()}, par_max={names[i]: v for i, v in pars.items()},
                              channel_range_ratio={names[i]: v for i, v in quant_run.crr.items()},
                              acc_int8=r4(evaluate(mqs, xte, D["y_test"])[1]), alpha=0.5)
    # 루프라인: 층별 산술 강도(연산/바이트), int8 기준, 가상 엣지 장치
    roof = {}
    shapes = [(4, 16, 32), (16, 32, 16), (32, 64, 8)]
    for bsz in (1, 64):
        rows = []
        for (ci, co, hw), nm in zip(shapes, names[:3]):
            macs = bsz * co * hw * hw * ci * 9
            byts = bsz * (ci * hw * hw + co * hw * hw) + co * ci * 9          # int8: 입력 + 출력 + 가중치, 1바이트씩
            rows.append(dict(layer=nm, gops=r4(2 * macs / 1e9), bytes=byts, ai=r4(2 * macs / byts)))
        roof[f"batch{bsz}"] = rows
    peak, bw = 4e12, 25.6e9
    out["roofline"] = dict(layers=roof, device=dict(peak_int8_ops=peak, bandwidth_Bps=bw, ridge_ops_per_byte=r4(peak / bw)),
                           note="가상 엣지 장치(정수 연산 최대 4 TOPS, 메모리 대역폭 25.6 GB/s) — 실제 제품 수치가 아님. 산술 강도 = 2·MAC / (입력+출력+가중치 바이트, int8)")
    out["note"] = ("VD-CNN(best). 가짜 양자화: 가중치 채널별 대칭 int8, 층 입력 활성 층별 비대칭 int8(학습 타일 500장의 최솟값~최댓값). "
                   "SQNR_l = 10·log10(E[X²]/E[(X − X̂)²]), 층 출력을 fp32와 비교(앞 층 오차가 쌓인 값). PAR = 채널별 max|x| / mean|x|의 최댓값(층 입력, 채널 배율에 무관). 채널 간 범위 비 = 채널별 max|x|의 최댓값 / 중앙값(이상치 채널이 생기면 커짐)")
    R["hw"] = out


# ================================================================== 61 배포 게이트
def exp_ops():
    import onnxruntime as ort
    A = base_model()
    B = TinyCNN(dropout=0.3); B.load_state_dict(torch.load(os.path.join(HERE, ".part8_dropout.pt"))); B.eval()
    out = {}
    for split in ("test", "shift"):
        y = D[f"y_{split}"]
        ca = (logits_of(A, D[f"x_{split}"]).argmax(1) == y).numpy(); cb = (logits_of(B, D[f"x_{split}"]).argmax(1) == y).numpy()
        n11, n10, n01, n00 = int((ca & cb).sum()), int((ca & ~cb).sum()), int((~ca & cb).sum()), int((~ca & ~cb).sum())
        per = {CLS5[c]: int(((y.numpy() == c) & ca & ~cb).sum()) for c in range(6)}
        out[split] = dict(acc_A=r4(ca.mean()), acc_B=r4(cb.mean()), N11=n11, N10=n10, N01=n01, N00=n00, net=n01 - n10,
                          RR_pct=r4(100 * n10 / (n11 + n10)), N10_per_class=per)
    # 게이트 판정에 쓸 B의 지표
    raw = P5.load()
    rob = robust_eval({"A": A, "B": B, "flat_mlp": None} if False else {"A": A, "B": B}, raw["x_test"], D["y_test"])
    base_rob = R.get("robust", {}).get("acc", {}).get("flat_mlp")
    gate = {}
    for nm, mm in (("A", A), ("B", B)):
        rdi = {k: r4((rob[nm]["clean"] - np.mean(rob[nm][k])) / rob[nm]["clean"]) for k in NUIS}
        mce = r4(100 * np.mean([sum(1 - a for a in rob[nm][k]) / sum(1 - a for a in base_rob[k]) for k in NUIS])) if base_rob else None
        L = logits_of(mm, D["x_test"]); e = ece(L.softmax(1), D["y_test"])[0]
        Ls = logits_of(mm, D["x_shift"])
        sl_acc = {}
        pred = L.argmax(1)
        for c in range(6):
            msk = D["y_test"] == c; sl_acc[f"클래스={CLS5[c]}"] = r4((pred[msk] == c).float().mean())
        sl_acc["연무·계절 시험"] = r4((Ls.argmax(1) == D["y_shift"]).float().mean())
        gate[nm] = dict(acc=r4((pred == D["y_test"]).float().mean()), max_rdi=max(rdi.values()), worst_rdi=max(rdi, key=rdi.get), mce=mce,
                        ece=r4(e), slice_acc=sl_acc, min_slice=min(sl_acc.values()))
    # 지연시간: ONNX Runtime CPU 1스레드, 배치 1을 2,000번
    p = os.path.join(HERE, ".part8_B.onnx")
    torch.onnx.export(B, torch.zeros(1, 4, 32, 32), p, input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
    x1 = D["x_test"][:1].numpy(); ts = []
    for _ in range(50):
        sess.run(None, {"x": x1})
    for i in range(2000):
        t0 = time.perf_counter(); sess.run(None, {"x": D["x_test"][i % 900:i % 900 + 1].numpy()}); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    lat = dict(p50=r4(np.percentile(ts, 50)), p95=r4(np.percentile(ts, 95)), p99=r4(np.percentile(ts, 99)), max=r4(ts.max()), mean=r4(ts.mean()))
    tau = dict(slice_min=0.85, slice_drop=0.03, max_rdi=0.40, mce=100.0, ece=0.08, p99_ms=1.0)
    gA, gB = gate["A"], gate["B"]
    drops = {k: r4(gA["slice_acc"][k] - gB["slice_acc"][k]) for k in gA["slice_acc"]}
    G = dict(G_macro=gB["acc"] >= gA["acc"], G_slice=(gB["min_slice"] >= tau["slice_min"]) and (max(drops.values()) <= tau["slice_drop"]),
             G_robust=(gB["mce"] is not None and gB["mce"] <= tau["mce"]) and gB["max_rdi"] <= tau["max_rdi"],
             G_calib=gB["ece"] <= tau["ece"], G_latency=lat["p99"] <= tau["p99_ms"])
    out["gate"] = dict(metrics=gate, slice_drop_A_to_B=drops, latency_ms_B=lat, thresholds=tau, gates=G, decision="Go" if all(G.values()) else "No-Go",
                       note=("A = 5부 VD-CNN(best), B = 드롭아웃 0.3으로 다시 학습한 VD-CNN(55강, best). 게이트는 Diagnostics 노트의 5대 논리곱 구조를 따르되, "
                             "G_robust는 'max RDI ≤ 0.40'(= 유지율 1 − RDI ≥ 0.60)으로 고쳐 씀. 지표는 mAP 대신 정확도, 슬라이스는 클래스별 + 연무·계절 시험. "
                             "기준값(τ)은 이 예제에서 정한 잠정치. 지연시간은 이 실행 환경 CPU 기준"))
    R["ops"] = out


# ================================================================== 51 처방 엔진
RULES = [
    # (규칙 id, 등급, 범주, 지표 이름, 비교, (낮은 기준, 높은 기준), 재학습 필요, 처방)
    ("RX-DATA-01", "P0", "데이터·라벨", "missing_gt_ratio", ">=", (0.10, 0.15), "라벨 재검수(재학습은 그 뒤)",
     "고신뢰 배경 오탐(점수 ≥ 0.85) 표본을 격리해 라벨 누락 여부를 전수 검수"),
    ("RX-DATA-02", "P0", "데이터·라벨", "critical_slice_miss", ">=", (0.50, 0.70), "자료 수집 후 재학습",
     "미탐률이 높은 슬라이스의 자료를 집중 수집하고 그 조건을 겨냥한 증강 구성"),
    ("RX-MODEL-01", "P1", "학습·구조", "nms_kill_ratio", ">=", (0.20, 0.30), "재학습(손실·헤드) 또는 후처리 교체",
     "밀집 객체의 이웃 정답 삭제: Soft-NMS·Repulsion Loss 검토, 근본적으로는 NMS 없는 구조"),
    ("RX-MODEL-02", "P1", "학습·구조", "loc_center_dominant", ">=", (0.50, 0.60), "재학습",
     "위치 오차가 중심 어긋남 위주: CIoU·EIoU 계열 손실, 이동 증강"),
    ("RX-MODEL-03", "P1", "학습·구조", "cur", "<", (0.20, 0.25), "재학습",
     "특징 공간 일부만 쓰임(유효 랭크 낮음): 직교 규제·채널 수 재검토. 클래스 수가 적으면 원래 낮을 수 있음"),
    ("RX-MODEL-04", "P1", "학습·구조", "max_rdi", ">=", (0.35, 0.40), "재학습(증강)",
     "가장 취약한 교란을 겨냥한 증강·전처리(57강의 3계층 처방)"),
    ("RX-INF-01", "P2", "추론·보정", "ece", ">=", (0.10, 0.12), "재학습 없음",
     "온도 스케일링 등 사후 보정(같은 분포 검증 자료로 맞춤)"),
]


def run_engine(metrics, level):
    out = []
    for rid, pr, cat, key, op, (lo, hi), retrain, action in RULES:
        if key not in metrics or metrics[key] is None:
            continue
        thr = lo if level == "low" else hi
        v = metrics[key]
        hit = v >= thr if op == ">=" else v < thr
        if hit:
            out.append(dict(id=rid, priority=pr, category=cat, metric=key, value=v, threshold=thr, threshold_range=[lo, hi],
                            retrain=retrain, action=action))
    order = {"P0": 0, "P1": 1, "P2": 2}
    return sorted(out, key=lambda r: order[r["priority"]])


def exp_prx():
    t = R["tide"]; nd = R["nmsdiag"]; c = R["calib"]; rp = R["repr"]; rb = R["robust"]; sl = R["slice"]
    det_normal = dict(missing_gt_ratio=t["high_conf_bkg_ratio"]["clean"], nms_kill_ratio=nd["classwise_0.5"]["kill_ratio"],
                      loc_center_dominant=t["loc"]["center_dominant_frac"])
    det_dense = dict(det_normal, nms_kill_ratio=nd["dense_mooring"]["nms_0.5"]["kill_ratio"])
    det_missing = dict(det_normal, missing_gt_ratio=t["high_conf_bkg_ratio"]["by_threshold"]["0.5"]["drop20"])
    worst_shift = sl["shift"]["worst"][0]
    clf = dict(ece=c["test"]["ece"], cur=rp["base"]["cur"], max_rdi=rb["vdcnn"]["max_rdi"],
               critical_slice_miss=r4(1 - min(r["acc"] for r in sl["test"]["worst"])))
    clf_shift = dict(clf, ece=c["shift"]["ece"], critical_slice_miss=r4(1 - worst_shift["acc"]))
    cases = {"detector_6부": det_normal, "detector_밀집계류": det_dense, "detector_라벨20%누락(점수기준0.5)": det_missing,
             "classifier_시험": clf, "classifier_연무·계절": clf_shift}
    out = {"rules": [dict(id=r[0], priority=r[1], category=r[2], metric=r[3], op=r[4], range=list(r[5]), retrain=r[6], action=r[7]) for r in RULES]}
    for name, mtr in cases.items():
        out[name] = dict(metrics=mtr, low=run_engine(mtr, "low"), high=run_engine(mtr, "high"))
    out["schema_example"] = dict(model="VD-CNN v1 (5부, best)", dataset="5부 연무·계절 시험(900장)", diagnosed_at="2026-10-02",
                                 metrics=clf_shift, prescriptions=run_engine(clf_shift, "high"),
                                 provenance=dict(code="run_part8.py", thresholds="Diagnostics 노트 잠정치를 범위로"))
    out["note"] = ("규칙의 기준값은 Diagnostics 노트(본문·참조 코드)의 값들을 포함하는 범위로 두고, 낮은 기준(low)과 높은 기준(high)에서 각각 엔진을 돌린 결과. "
                   "missing_gt_ratio = 고신뢰 배경 오탐 / 전체 오탐(52강), nms_kill_ratio(53강), loc_center_dominant(52강), cur(56강), max_rdi(57강), ece(55강), "
                   "critical_slice_miss = 1 − 가장 나쁜 슬라이스 정확도(54강). 등급은 조치의 성격: P0 = 자료·라벨 결함(학습 중단·정제), P1 = 재학습이 필요한 손실·구조·증강, P2 = 재학습 없는 후처리·보정")
    R["prx"] = out


EXPS = {k[4:]: v for k, v in list(globals().items()) if k.startswith("exp_")}

if __name__ == "__main__":
    names = sys.argv[1:] or list(EXPS)
    for nme in names:
        t0 = time.time()
        EXPS[nme]()
        R.setdefault("_meta", {})[nme] = round(time.time() - t0, 1)
        save()
        print(nme, "done", round(time.time() - t0, 1), "s", flush=True)
