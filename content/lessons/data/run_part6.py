"""6부 본문 수치를 만드는 계산 모음. 결과는 part6_runs.json (몇 분, 학습 없음).

    python3 run_part6.py            # 전체
    python3 run_part6.py nms map    # 일부만 (다른 결과는 유지)

기본 설정(본문에서 '기본 탐지 결과'라 부름): 타일 24장을 가상 탐지기에 그대로 넣고(scale 1, 시드 = 영상 번호)
클래스별 NMS(IoU 0.5)를 거친 결과. HBB 평가는 pycocotools와 같은 값이 나오는지 'check'에서 확인함

실험 이름과 쓰는 강
- data   : 자료 요약(클래스별 개수, 크기 구간, 타일별 개수) (34·39강)
- check  : 직접 구현한 COCO AP와 pycocotools 비교 (39강)
- iou    : 크기별 1~2화소 어긋남의 IoU, GIoU·DIoU·CIoU 예, 기본 결과에서 짝지은 상자의 IoU 분포 (35강)
- anchor : YOLOv3 기본 앵커와 우리 자료의 모양 맞음, k-평균 앵커, 앵커 프리 격자점 수, TAL·SimOTA 계산 예 (36강)
- nms    : NMS 임계값·클래스 무관·Soft-NMS·WBF 비교, 45° 주차장 재현율 (37강)
- slice  : 2,048 장면 전체 축소 추론 vs 슬라이스 추론(겹침 0·64·128) vs 둘을 합친 것 (37강)
- pr     : 클래스별 PR 곡선 점, 운영점(점수 임계값별 P·R·F1·FPPI), F1 최대점 (38강)
- map    : AP 계산 방식(11점·전체 점·101점), mAP50·75·50-95, maxDets, 크기별 AP, 매크로·마이크로 평균 (39강)
- seg    : 5부 U-Net 예측으로 픽셀 정확도·mIoU·Dice·카파·경계 F1, 선박 마스크 AP (40강)
- obb    : HBB IoU vs 회전 IoU 평가, 채움 비율, 각도 경계 불연속 예 (41강)
- mot    : 추적기 세 가지(IoU만, SORT식 칼만, ByteTrack식 2단계)의 MOTA·IDF1·HOTA·ID 스위치 (42강)
"""
import io, contextlib, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_part6 import tiles, detect, big_scene, sequence, obb_corners, obb_to_hbb, norm_angle, TILE, GSD
from part6_eval import (iou_matrix, iou_family, riou, riou_matrix, nms, nms_dets, soft_nms_dets, wbf_dets,
                        coco_eval, pr_points, ap_from_pr, operating_point, AREA, IOU_THRS)

OUT = os.environ.get("PART6_OUT") or os.path.join(os.path.dirname(__file__), "part6_runs.json")
R = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
T = tiles()
RAW = [detect(t["objs"], t["clutter"], 1.0, seed=t["image_id"]) for t in T]


def base_images(thr=0.5, **kw):
    return [(t["objs"], nms_dets(r, thr, **kw)) for t, r in zip(T, RAW)]


BASE = base_images()


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def save():
    json.dump(R, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ 자료
def exp_data():
    per = np.zeros(3, int)
    bins = {c: [0, 0, 0] for c in range(3)}
    for t in T:
        for o in t["objs"]:
            per[o["cls"]] += 1
            a = o["area"]
            bins[o["cls"]][0 if a < 32 ** 2 else 1 if a < 96 ** 2 else 2] += 1
    lens = {c: [round(float(np.percentile([o["w"] * GSD for t in T for o in t["objs"] if o["cls"] == c], q)), 1) for q in (0, 50, 100)] for c in range(3)}
    R["data"] = dict(n_tiles=len(T), kinds={k: sum(t["kind"] == k for t in T) for k in ["harbor", "sea", "marina"]},
                     per_class=per.tolist(), size_bins_small_medium_large=bins, length_m_min_med_max=lens,
                     objs_per_tile=[len(t["objs"]) for t in T], vehicles_per_tile=[sum(o["cls"] == 2 for o in t["objs"]) for t in T],
                     raw_dets=sum(len(r) for r in RAW), base_dets=sum(len(d) for _, d in BASE))


def exp_check():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = {"images": [{"id": t["image_id"], "width": TILE, "height": TILE} for t in T],
          "categories": [{"id": c} for c in range(3)], "annotations": []}
    aid = 1
    for t in T:
        for o in t["objs"]:
            x1, y1, x2, y2 = o["hbb"]
            gt["annotations"].append({"id": aid, "image_id": t["image_id"], "category_id": o["cls"],
                                      "bbox": [x1, y1, x2 - x1, y2 - y1], "area": o["area"], "iscrowd": 0})
            aid += 1
    dt = [{"image_id": t["image_id"], "category_id": x["cls"], "score": x["score"],
           "bbox": [x["hbb"][0], x["hbb"][1], x["hbb"][2] - x["hbb"][0], x["hbb"][3] - x["hbb"][1]]}
          for t, (_, d) in zip(T, BASE) for x in d]
    with contextlib.redirect_stdout(io.StringIO()):
        C = COCO(); C.dataset = gt; C.createIndex()
        E = COCOeval(C, C.loadRes(dt), "bbox"); E.evaluate(); E.accumulate(); E.summarize()
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    mine = coco_eval(BASE)
    R["check"] = dict(pycocotools={n: r4(v) for n, v in zip(names, E.stats)},
                      mine=dict(AP=mine["ap"], AP50=mine["ap50"], AP75=mine["ap75"],
                                AP_small=coco_eval(BASE, area="small")["ap"], AP_medium=coco_eval(BASE, area="medium")["ap"],
                                AP_large=coco_eval(BASE, area="large")["ap"]))


# ------------------------------------------------------------------ 35 IoU
def exp_iou():
    out = {}
    shapes = {"차량 9×4": (9, 4), "소형선박 24×8": (24, 8), "선박 200×34": (200, 34)}
    tab = {}
    for name, (w, h) in shapes.items():
        g = [0, 0, w, h]
        tab[name] = {f"x{d}px": r4(iou_matrix([g], [[d, 0, w + d, h]])[0, 0]) for d in (1, 2, 3)}
        tab[name]["xy1px"] = r4(iou_matrix([g], [[1, 1, w + 1, h + 1]])[0, 0])
        tab[name]["scale1.1"] = r4(iou_matrix([g], [[-(0.05 * w), -(0.05 * h), w * 1.05, h * 1.05]])[0, 0])
    out["shift"] = tab
    g = [100, 100, 124, 108]                          # 소형선박 24×8
    ex = {"겹침 없음 가까움(4화소 떨어짐)": [128, 100, 152, 108], "겹침 없음 멂(40화소)": [164, 100, 188, 108],
          "안에 포함(중심 같음, 절반 크기)": [106, 102, 118, 106], "같은 넓이·중심 어긋남 6화소": [106, 100, 130, 108],
          "중심 같고 모양 다름(16×12)": [104, 98, 120, 110]}
    out["family"] = {k: {kk: r4(vv) for kk, vv in iou_family(g, b).items()} for k, b in ex.items()}
    out["family_gt"] = g
    # 기본 결과에서 정답마다 가장 잘 맞는 탐지의 IoU (같은 클래스, 0.1 이상)
    dist = {}
    for c in range(3):
        v = []
        for gts, dets in BASE:
            G = [x["hbb"] for x in gts if x["cls"] == c]
            D = [x["hbb"] for x in dets if x["cls"] == c]
            if G and D:
                M = iou_matrix(G, D).max(1)
                v += M[M > 0.1].tolist()
        v = np.array(v)
        dist[c] = dict(n=len(v), median=r4(np.median(v)), frac_ge_0_5=r4((v >= 0.5).mean()), frac_ge_0_75=r4((v >= 0.75).mean()),
                       frac_ge_0_9=r4((v >= 0.9).mean()))
    out["best_iou_dist"] = dist
    R["iou"] = out


# ------------------------------------------------------------------ 36 앵커·배정
YOLO3 = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]


def shape_iou(wh, anchors):
    wh, A = np.asarray(wh, float), np.asarray(anchors, float)
    inter = np.minimum(wh[:, None, 0], A[None, :, 0]) * np.minimum(wh[:, None, 1], A[None, :, 1])
    return inter / (wh[:, None, 0] * wh[:, None, 1] + A[None, :, 0] * A[None, :, 1] - inter)


def kmeans_anchors(wh, k=9, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    C = wh[rng.choice(len(wh), k, replace=False)]
    for _ in range(iters):
        a = shape_iou(wh, C).argmax(1)
        C2 = np.array([np.median(wh[a == j], 0) if (a == j).any() else C[j] for j in range(k)])
        if np.allclose(C2, C):
            break
        C = C2
    return C[np.argsort(C.prod(1))]


def exp_anchor():
    wh, cls = [], []
    for t in T:
        for o in t["objs"]:
            hb = o["hbb"]; wh.append((hb[2] - hb[0], hb[3] - hb[1])); cls.append(o["cls"])
    wh, cls = np.array(wh), np.array(cls)
    out = {}
    best = shape_iou(wh, YOLO3).max(1)
    out["yolo3_anchors"] = YOLO3
    out["yolo3_best_iou"] = {c: dict(median=r4(np.median(best[cls == c])), frac_lt_0_5=r4((best[cls == c] < 0.5).mean()),
                                    frac_lt_0_3=r4((best[cls == c] < 0.3).mean())) for c in range(3)}
    K = kmeans_anchors(wh)
    bk = shape_iou(wh, K).max(1)
    out["kmeans_anchors"] = [[round(float(a), 1), round(float(b), 1)] for a, b in K]
    out["kmeans_best_iou"] = {c: dict(median=r4(np.median(bk[cls == c])), frac_lt_0_5=r4((bk[cls == c] < 0.5).mean())) for c in range(3)}
    # 앵커 프리: 정답 HBB 안에 들어가는 스트라이드 8·16·32 격자점(칸 중심) 수
    pts = {}
    for s in (8, 16, 32):
        cnt = []
        for t in T:
            for o in t["objs"]:
                x1, y1, x2, y2 = o["hbb"]
                gx = np.arange(s / 2, TILE, s)
                nx = ((gx > x1) & (gx < x2)).sum(); ny = ((gx > y1) & (gx < y2)).sum()
                cnt.append(nx * ny)
        cnt = np.array(cnt)
        pts[s] = {c: dict(zero=r4((cnt[cls == c] == 0).mean()), median=float(np.median(cnt[cls == c]))) for c in range(3)}
    out["anchor_free_points"] = pts
    # TAL·SimOTA 계산 예: 외해 타일에서 소형선박 하나(정답)를 둘러싼 후보(가상 탐지기 원시 출력, 같은 정답에서 나온 것)
    ti = next(i for i, t in enumerate(T) if t["kind"] == "sea" and sum(1 for d in RAW[i] if d["src"] != "clutter" and d["src"] != "bg") >= 6)
    t = T[ti]
    counts = {}
    for d in RAW[ti]:
        if isinstance(d["src"], int):
            counts[d["src"]] = counts.get(d["src"], 0) + 1
    gi = max(counts, key=counts.get)
    g = t["objs"][gi]
    cand = [d for d in RAW[ti] if d["src"] == gi][:8]
    ious = iou_matrix([d["hbb"] for d in cand], [g["hbb"]])[:, 0]
    sc = np.array([d["score"] if d["cls"] == g["cls"] else d["score"] * 0.2 for d in cand])   # 정답 클래스 점수(다른 클래스로 낸 것은 낮게 둠)
    tal = sc ** 1.0 * ious ** 6.0
    cost = -np.log(np.clip(sc, 1e-8, 1)) + 3.0 * -np.log(np.clip(ious, 1e-8, 1))
    dyn_k = max(1, int(np.sort(ious)[::-1][:10].sum()))
    out["assign_example"] = dict(tile=t["image_id"], gt_cls=g["cls"], gt_hbb=[round(v, 1) for v in g["hbb"]],
                                 cand=[dict(hbb=[round(v, 1) for v in d["hbb"]], score=r4(s), iou=r4(u), tal=r4(a), simota_cost=r4(c))
                                       for d, s, u, a, c in zip(cand, sc, ious, tal, cost)],
                                 tal_top=[int(i) for i in np.argsort(-tal)[:3]], simota_dynamic_k=dyn_k,
                                 simota_pick=[int(i) for i in np.argsort(cost)[:dyn_k]],
                                 note="TAL: t = s^1 · u^6 (α=1, β=6), 상위 k개. SimOTA: 비용 = 분류 비용(-log s) + 3·IoU 비용(-log IoU), 동적 k = 상위 10개 IoU 합의 정수부(최소 1)")
    R["anchor"] = out


# ------------------------------------------------------------------ 37 NMS
def dense_lot_recall(images, thr=0.5):
    """45° 주차장(차량 각도가 45°±20° 안) 차량의 재현율(IoU 0.5, 점수 무관)"""
    hit = n = 0
    for gts, dets in images:
        G = [x for x in gts if x["cls"] == 2 and abs((x["theta"] % 90) - 45) < 20]
        D = [x["hbb"] for x in dets if x["cls"] == 2]
        n += len(G)
        if G and D:
            M = iou_matrix([x["hbb"] for x in G], D)
            # 한 탐지는 한 정답에만: 탐욕 매칭
            used = set()
            for gi in np.argsort(-M.max(1)):
                j = [k for k in np.argsort(-M[gi]) if k not in used and M[gi, k] >= thr]
                if j:
                    used.add(j[0]); hit += 1
    return r4(hit / n) if n else None


def exp_nms():
    out = {}
    raw_images = [(t["objs"], r) for t, r in zip(T, RAW)]
    out["raw"] = dict(n=sum(len(r) for r in RAW), **{k: v for k, v in coco_eval(raw_images).items() if k != "per_class"})
    gt_nb = []
    for t in T:      # 이웃 정답끼리의 최대 HBB IoU (45° 주차장)
        G = [x for x in t["objs"] if x["cls"] == 2]
        if len(G) > 1:
            M = iou_matrix([x["hbb"] for x in G], [x["hbb"] for x in G]); np.fill_diagonal(M, 0)
            ang = np.array([abs((x["theta"] % 90) - 45) < 20 for x in G])
            if ang.any():
                gt_nb += M.max(1)[ang].tolist()
    gt_nb = np.array(gt_nb)
    out["lot45_neighbor_gt_iou"] = dict(n=len(gt_nb), median=r4(np.median(gt_nb)), frac_gt_0_5=r4((gt_nb > 0.5).mean()), frac_gt_0_3=r4((gt_nb > 0.3).mean()))
    for thr in (0.3, 0.5, 0.7):
        im = base_images(thr)
        e = coco_eval(im)
        out[f"nms_{thr}"] = dict(n=sum(len(d) for _, d in im), ap50=e["ap50"], ap=e["ap"], per_class_ap50={c: e["per_class"][c][0] for c in range(3)},
                                lot45_recall=dense_lot_recall(im))
    im = base_images(0.5, agnostic=True)
    e = coco_eval(im)
    out["agnostic_0.5"] = dict(n=sum(len(d) for _, d in im), ap50=e["ap50"], ap=e["ap"], per_class_ap50={c: e["per_class"][c][0] for c in range(3)})
    im = [(t["objs"], soft_nms_dets(r, 0.5)) for t, r in zip(T, RAW)]
    e = coco_eval(im)
    out["soft_nms_gauss0.5"] = dict(n=sum(len(d) for _, d in im), ap50=e["ap50"], ap=e["ap"], per_class_ap50={c: e["per_class"][c][0] for c in range(3)},
                                    lot45_recall=dense_lot_recall(im))
    im = [(t["objs"], wbf_dets(r, 0.55)) for t, r in zip(T, RAW)]
    for g, d in im:
        for x in d:
            hb = x["hbb"]
    e = coco_eval(im)
    out["wbf_0.55"] = dict(n=sum(len(d) for _, d in im), ap50=e["ap50"], ap75=e["ap75"], ap=e["ap"], per_class_ap50={c: e["per_class"][c][0] for c in range(3)})
    e = coco_eval(BASE)
    out["nms_0.5"]["ap75"] = e["ap75"]
    # 앙상블: 같은 타일을 두 '모델'(시드가 다른 가상 탐지기)로 탐지해 합치기 — 합집합 NMS vs WBF
    from ensemble_boxes import weighted_boxes_fusion
    A = BASE
    Bm = [(t["objs"], nms_dets(detect(t["objs"], t["clutter"], 1.0, seed=1000 + t["image_id"]), 0.5)) for t in T]
    eb = coco_eval(Bm)
    out["model_b"] = dict(ap50=eb["ap50"], ap75=eb["ap75"], ap=eb["ap"])
    un = [(g, nms_dets(da + db, 0.5)) for (g, da), (_, db) in zip(A, Bm)]
    eu = coco_eval(un)
    out["ensemble_union_nms"] = dict(ap50=eu["ap50"], ap75=eu["ap75"], ap=eu["ap"])
    wf = []
    for (g, da), (_, db) in zip(A, Bm):
        lists = [(np.clip(np.array([d["hbb"] for d in x]) / TILE, 0, 1).tolist(), [d["score"] for d in x], [d["cls"] for d in x]) for x in (da, db)]
        b, s_, l = weighted_boxes_fusion([x[0] for x in lists], [x[1] for x in lists], [x[2] for x in lists], iou_thr=0.55, skip_box_thr=0.0)
        wf.append((g, [dict(cls=int(c), score=float(sc), hbb=(bb * TILE).tolist()) for bb, sc, c in zip(b, s_, l)]))
    ew = coco_eval(wf)
    out["ensemble_wbf"] = dict(ap50=ew["ap50"], ap75=ew["ap75"], ap=ew["ap"])
    out["wbf_note"] = "wbf_0.55는 한 모델의 NMS 전 원시 출력에 바로 WBF를 건 것. ensemble_*는 두 모델의 NMS 후 결과를 합친 것(ensemble_boxes 1.0.9, conf_type avg)"
    R["nms"] = out


def exp_slice():
    B = big_scene()
    S = B["size"]
    gts = B["objs"]
    out = dict(n_gt=len(gts), per_class=np.bincount([o["cls"] for o in gts], minlength=3).tolist(),
               max_ship_len_px=round(max(o["w"] for o in gts), 1))

    def ev(dets, name):
        e = coco_eval([(gts, dets)], max_dets=1000)
        rec = {}
        for c in range(3):
            _, tp, n = pr_points([(gts, dets)], c, 0.5, max_dets=1000)
            rec[c] = r4(tp.sum() / n) if n else None
        out[name] = dict(n_dets=len(dets), ap50=e["ap50"], ap=e["ap"], recall50=rec,
                         per_class_ap50={c: e["per_class"][c][0] for c in range(3)})

    # 1) 전체를 640으로 줄여 한 번
    full = detect(gts, B["clutter"], scale=640 / S, seed=7)
    full_n = nms_dets(full, 0.5)
    ev(full_n, "full_640")
    # 2) 512 슬라이스, 겹침 0·64·128
    for ov in (0, 64, 128):
        step = 512 - ov
        starts = list(range(0, S - 512 + 1, step))
        if starts[-1] != S - 512:
            starts.append(S - 512)
        dets = []
        k = 0
        for y0 in starts:
            for x0 in starts:
                k += 1
                dets += detect(gts, B["clutter"], 1.0, seed=100 + k, view=(x0, y0, x0 + 512, y0 + 512))
        merged = nms_dets(dets, 0.5)
        ev(merged, f"slice_ov{ov}")
        out[f"slice_ov{ov}"]["n_tiles"] = len(starts) ** 2
        if ov == 128:
            ev(nms_dets(dets + full, 0.5), "slice_ov128_plus_full")
            # 잘린 조각 상자: 어느 정답과도 IoU 0.5 미만인데 정답 하나에 절반 넘게 들어가는 탐지
            frag = 0
            G = np.array([o["hbb"] for o in gts])
            for d in merged:
                M = iou_matrix([d["hbb"]], G)[0]
                if M.max() < 0.5:
                    hb = np.array(d["hbb"])
                    inter = (np.clip(np.minimum(hb[2], G[:, 2]) - np.maximum(hb[0], G[:, 0]), 0, None) *
                             np.clip(np.minimum(hb[3], G[:, 3]) - np.maximum(hb[1], G[:, 1]), 0, None))
                    if (inter / ((hb[2] - hb[0]) * (hb[3] - hb[1]))).max() > 0.5 and d["score"] >= 0.3:
                        frag += 1
            out["slice_ov128"]["fragment_boxes_score_ge_0_3"] = frag
    R["slice"] = out


# ------------------------------------------------------------------ 38 PR
def exp_pr():
    out = {}
    for c in range(3):
        S, tp, n = pr_points(BASE, c, 0.5, max_dets=1000)
        tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
        P, Rr = tpc / np.maximum(tpc + fpc, 1), tpc / n
        F = 2 * P * Rr / np.maximum(P + Rr, 1e-12)
        i = int(F.argmax())
        # 곡선 점(그림용): 점수 임계값 0.05 간격
        curve = []
        for s in np.arange(0.95, 0.0, -0.05):
            k = (S >= s).sum()
            if k:
                curve.append([round(float(s), 2), r4(P[k - 1]), r4(Rr[k - 1])])
        out[c] = dict(n_gt=n, n_det=len(S), f1_max=r4(F[i]), f1_max_score=r4(S[i]), p_at_f1max=r4(P[i]), r_at_f1max=r4(Rr[i]),
                      max_recall=r4(Rr[-1]), curve=curve,
                      ops={str(s): operating_point(BASE, c, s) for s in (0.1, 0.25, 0.4, 0.5, 0.7)})
        for target in (0.8, 0.9):
            ok = np.where(Rr >= target)[0]
            out[c][f"score_for_recall_{target}"] = r4(S[ok[0]]) if len(ok) else None
            out[c][f"precision_at_recall_{target}"] = r4(P[ok[0]]) if len(ok) else None
    R["pr"] = out


# ------------------------------------------------------------------ 39 mAP
def exp_map():
    out = {}
    for c in range(3):
        _, tp, n = pr_points(BASE, c, 0.5)
        out[f"ap50_methods_{c}"] = {m: r4(ap_from_pr(tp, n, m)) for m in ("voc11", "voc", "coco")}
    e = coco_eval(BASE)
    out["coco"] = e
    out["per_class_ap50_95"] = {c: r4(np.mean(e["per_class"][c])) for c in range(3)}
    md = {}
    for m in (1, 10, 100, 300):
        em = coco_eval(BASE, max_dets=m)
        _, tp, n = pr_points(BASE, 2, 0.5, max_dets=m)
        md[m] = dict(ap=em["ap"], ap50=em["ap50"], vehicle_ap50=em["per_class"][2][0], vehicle_recall50=r4(tp.sum() / n))
    out["max_dets"] = md
    out["tiles_vehicles_gt_100"] = int(sum(1 for t in T if sum(o["cls"] == 2 for o in t["objs"]) > 100))
    out["by_size"] = {a: coco_eval(BASE, area=a) for a in ("small", "medium", "large")}
    # 원격탐사식 크기 구간(실제 길이 기준): 10 m 미만, 10~40 m, 40 m 이상 — HBB 넓이 대신 객체 길이로 나눔
    rs = {}
    for name, (lo, hi) in {"lt10m": (0, 10), "10to40m": (10, 40), "ge40m": (40, 1e9)}.items():
        imgs = [([dict(x, size=x["w"] * GSD) for x in g], [dict(x, size=x["obb"][2] * GSD) for x in d]) for g, d in BASE]
        rs[name] = dict(n_gt=sum(1 for g, _ in imgs for x in g if lo <= x["size"] < hi),
                        ap50=coco_eval(imgs, area=(lo, hi))["ap50"], ap=coco_eval(imgs, area=(lo, hi))["ap"])
    out["rs_length_bins"] = rs
    out["rs_length_bins_note"] = "COCO 크기 규칙을 넓이 대신 실제 길이(m)로 적용: 구간 밖 정답은 무시, 짝 없는 탐지는 예측 회전 박스 길이로 구간 판정"
    # 매크로 vs 마이크로(정답 수 가중)
    n = np.array([sum(1 for g, _ in BASE for x in g if x["cls"] == c) for c in range(3)])
    ap50 = np.array([e["per_class"][c][0] for c in range(3)])
    out["macro_ap50"] = r4(ap50.mean())
    out["weighted_ap50"] = r4((ap50 * n).sum() / n.sum())
    out["n_per_class"] = n.tolist()
    R["map"] = out


# ------------------------------------------------------------------ 40 분할 지표
def boundary_f1(pred, gt, tol=1):
    from scipy import ndimage as ndi
    def edges(m):
        return (m != ndi.maximum_filter(m, 3)) | (m != ndi.minimum_filter(m, 3))
    ep, eg = edges(pred), edges(gt)
    if ep.sum() == 0 and eg.sum() == 0:
        return None
    dp = ndi.binary_dilation(ep, iterations=tol) if tol else ep
    dg = ndi.binary_dilation(eg, iterations=tol) if tol else eg
    P = (ep & dg).sum() / max(ep.sum(), 1)
    Rr = (eg & dp).sum() / max(eg.sum(), 1)
    return 2 * P * Rr / (P + Rr) if P + Rr else 0.0


def exp_seg():
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from part5_common import data, TinyUNet
    D = data()
    y = D["m_test"].numpy()
    out = {}
    for name, skip in (("skip", True), ("noskip", False)):
        p = os.path.join(os.path.dirname(__file__), f".part5_unet_{name}.pt")
        m = TinyUNet(skip=skip); m.load_state_dict(torch.load(p)); m.eval()
        with torch.no_grad():
            pred = torch.cat([m(D["x_test"][i:i + 300]).argmax(1) for i in range(0, 900, 300)]).numpy()
        cm = np.bincount(y.ravel() * 6 + pred.ravel(), minlength=36).reshape(6, 6).astype(float)
        tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
        iou = tp / (tp + fp + fn); dice = 2 * tp / (2 * tp + fp + fn)
        pa = tp.sum() / cm.sum()
        pe = (cm.sum(0) * cm.sum(1)).sum() / cm.sum() ** 2
        kappa = (pa - pe) / (1 - pe)
        mixed = [i for i in range(len(y)) if len(np.unique(y[i])) > 1]
        bf = {tol: r4(np.mean([v for v in (boundary_f1(pred[i], y[i], tol) for i in mixed) if v is not None])) for tol in (0, 1, 2)}
        out[name] = dict(pixel_acc=r4(pa), miou=r4(iou.mean()), iou=[r4(v) for v in iou], dice=[r4(v) for v in dice],
                         mean_dice=r4(dice.mean()), kappa=r4(kappa), class_pixel_frac=[r4(v) for v in cm.sum(1) / cm.sum()],
                         boundary_f1_mixed_tiles=bf, n_mixed=len(mixed))
        if name == "skip":
            # 한 타일 예: 수계 타일을 통째로 갯벌로 칠한 경우가 있는지 → 타일별 정확도 최솟값
            acc_t = (pred == y).reshape(len(y), -1).mean(1)
            out["tile_acc_min"] = r4(acc_t.min()); out["tiles_acc_lt_0_5"] = int((acc_t < 0.5).sum())
    # 다이스와 IoU 관계 확인: Dice = 2IoU/(1+IoU)
    out["dice_from_iou_check"] = r4(2 * out["skip"]["iou"][3] / (1 + out["skip"]["iou"][3]))
    # 마스크 AP: 6부 타일 선박·소형선박 회전 박스를 마스크로, 탐지의 회전 박스를 마스크로 (pycocotools segm)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    def coco_run(kind):
        gt = {"images": [{"id": t["image_id"], "width": TILE, "height": TILE} for t in T], "categories": [{"id": c} for c in range(3)], "annotations": []}
        aid = 1
        for t in T:
            for o in t["objs"]:
                pc = obb_corners(o["cx"], o["cy"], o["w"], o["h"], o["theta"])
                x1, y1, x2, y2 = o["hbb"]
                gt["annotations"].append({"id": aid, "image_id": t["image_id"], "category_id": o["cls"], "iscrowd": 0,
                                          "segmentation": [pc.ravel().tolist()], "bbox": [x1, y1, x2 - x1, y2 - y1], "area": o["area"]})
                aid += 1
        dt = []
        for t, (_, d) in zip(T, BASE):
            for x in d:
                pc = obb_corners(*x["obb"])
                x1, y1, x2, y2 = x["hbb"]
                dt.append({"image_id": t["image_id"], "category_id": x["cls"], "score": x["score"],
                           "segmentation": [pc.ravel().tolist()], "bbox": [x1, y1, x2 - x1, y2 - y1]})
        with contextlib.redirect_stdout(io.StringIO()):
            C = COCO(); C.dataset = gt; C.createIndex()
            E = COCOeval(C, C.loadRes(dt), kind); E.params.areaRng = [[0, 1e10]] * 4; E.evaluate(); E.accumulate(); E.summarize()
        return dict(AP=r4(E.stats[0]), AP50=r4(E.stats[1]), AP75=r4(E.stats[2]))
    out["mask_ap_obb_polygons"] = coco_run("segm")
    out["mask_ap_note"] = "정답·탐지의 회전 박스 다각형을 마스크로 써서 pycocotools segm 평가(크기 구간 없이). 마스크 IoU는 다각형 래스터 기준이라 41강의 회전 IoU와 거의 같음"
    R["seg"] = out


# ------------------------------------------------------------------ 41 OBB
def exp_obb():
    out = {}
    e_h = coco_eval(BASE)
    e_r = coco_eval([(g, nms_dets([x for x in r], 0.5, rotated=True)) for (g, _), r in zip(BASE, RAW)], rotated=True)
    out["hbb_eval"] = dict(ap50=e_h["ap50"], ap=e_h["ap"], per_class_ap50={c: e_h["per_class"][c][0] for c in range(3)},
                          per_class_ap75={c: e_h["per_class"][c][5] for c in range(3)})
    out["obb_eval_rotated_nms"] = dict(ap50=e_r["ap50"], ap=e_r["ap"], per_class_ap50={c: e_r["per_class"][c][0] for c in range(3)},
                                       per_class_ap75={c: e_r["per_class"][c][5] for c in range(3)})
    # 채움 비율: OBB 넓이 / HBB 넓이
    fill = {}
    for c in range(3):
        v = [o["w"] * o["h"] / o["area"] for t in T for o in t["objs"] if o["cls"] == c]
        fill[c] = dict(median=r4(np.median(v)), min=r4(np.min(v)))
    out["fill_ratio"] = fill
    # 길이 100 m(200화소)·폭 34화소 선박이 각도에 따라 HBB 채움 비율
    out["fill_by_angle_200x34"] = {a: r4(200 * 34 / np.prod(np.diff(obb_to_hbb((0, 0, 200, 34, a)).reshape(2, 2), axis=0))) for a in (0, 15, 30, 45, 60, 90)}
    # 같은 선박 두 척이 나란히(45°, 폭 사이 간격 4화소): HBB IoU vs 회전 IoU
    a = (100, 100, 200, 34, 45)
    off = 38 / np.sqrt(2)
    b = (100 - off, 100 + off, 200, 34, 45)
    out["side_by_side_45"] = dict(hbb_iou=r4(iou_matrix([obb_to_hbb(a)], [obb_to_hbb(b)])[0, 0]), rotated_iou=r4(riou(a, b)))
    # 각도 어긋남에 따른 회전 IoU(같은 중심): 길쭉한 선박 vs 정사각형에 가까운 물체
    out["riou_vs_angle"] = {f"{d}deg": dict(ship_200x34=r4(riou((0, 0, 200, 34, 0), (0, 0, 200, 34, d))),
                                            boat_24x8=r4(riou((0, 0, 24, 8, 0), (0, 0, 24, 8, d))),
                                            near_square_20x16=r4(riou((0, 0, 20, 16, 0), (0, 0, 20, 16, d)))) for d in (2, 5, 10, 15, 30)}
    # 경계 불연속: 정답 -88°, 예측 +89° (실제 차이 3°)
    g, p = (0, 0, 200, 34, -88), (0, 0, 200, 34, 89)
    out["boundary_example"] = dict(gt_theta=-88, pred_theta=89, naive_l1_deg=177, true_diff_deg=3, rotated_iou=r4(riou(g, p)))
    # 짧은 변 기준 규약에서 정사각형에 가까운 물체: w≈h면 긴 변이 바뀌며 각도가 90° 튐
    out["square_like"] = dict(obj=(0, 0, 20, 19.6, 10), alt=(0, 0, 19.6, 20, 100 - 180), riou_same=r4(riou((0, 0, 20, 19.6, 10), (0, 0, 19.6, 20, -80))))
    # 8파라미터 예
    out["eight_param_example"] = [round(float(v), 1) for v in obb_corners(100, 60, 40, 12, 30).ravel()]
    R["obb"] = out


# ------------------------------------------------------------------ 42 MOT
class KF:
    """등속 칼만 필터: 상태 (cx, cy, w, h, vx, vy)"""
    def __init__(self, box):
        x1, y1, x2, y2 = box
        self.x = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0, 0], float)
        self.P = np.diag([4, 4, 4, 4, 100, 100]).astype(float)
        self.F = np.eye(6); self.F[0, 4] = self.F[1, 5] = 1
        self.H = np.eye(4, 6)
        self.Q = np.diag([1, 1, 1, 1, 2, 2]).astype(float)
        self.Rm = np.diag([4, 4, 9, 9]).astype(float)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def update(self, box):
        x1, y1, x2, y2 = box
        z = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        S = self.H @ self.P @ self.H.T + self.Rm
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def box(self):
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def run_tracker(det, n_frames, mode="sort", high=0.5, low=0.1, iou_thr=0.3, max_age=5, min_hits=2):
    """mode: 'iou' (움직임 예측 없이 직전 상자와 IoU 매칭), 'sort' (칼만 예측 + 헝가리안, 높은 점수만),
    'byte' (ByteTrack식: 높은 점수로 1차 매칭 후 남은 트랙을 낮은 점수 탐지와 2차 매칭)"""
    from scipy.optimize import linear_sum_assignment
    tracks, out, nid = [], [], 1
    for f in range(n_frames):
        D = [d for d in det if d[0] == f]
        hi = [d for d in D if d[2] >= high]
        lo = [d for d in D if low <= d[2] < high] if mode == "byte" else []
        for tr in tracks:
            tr["pred"] = tr["kf"].predict() if mode != "iou" else tr["last"]
        def match(trs, ds):
            if not trs or not ds:
                return [], list(range(len(trs))), list(range(len(ds)))
            M = iou_matrix([t["pred"] for t in trs], [d[3:7] for d in ds])
            r, c = linear_sum_assignment(-M)
            pairs = [(i, j) for i, j in zip(r, c) if M[i, j] >= iou_thr]
            ut = [i for i in range(len(trs)) if i not in [p[0] for p in pairs]]
            ud = [j for j in range(len(ds)) if j not in [p[1] for p in pairs]]
            return pairs, ut, ud
        pairs, ut, ud = match(tracks, hi)
        for i, j in pairs:
            t = tracks[i]; box = np.array(hi[j][3:7])
            if mode != "iou":
                t["kf"].update(box)
            t["last"] = box; t["miss"] = 0; t["hits"] += 1
        if mode == "byte":
            rest = [tracks[i] for i in ut]
            pairs2, ut2, _ = match(rest, lo)
            for i, j in pairs2:
                t = rest[i]; box = np.array(lo[j][3:7]); t["kf"].update(box); t["last"] = box; t["miss"] = 0; t["hits"] += 1
            unmatched = [rest[i] for i in ut2]
        else:
            unmatched = [tracks[i] for i in ut]
        for t in unmatched:
            t["miss"] += 1
        for j in ud:
            box = np.array(hi[j][3:7])
            tracks.append(dict(id=nid, kf=KF(box), last=box, miss=0, hits=1, pred=box)); nid += 1
        tracks = [t for t in tracks if t["miss"] <= max_age]
        for t in tracks:
            if t["miss"] == 0 and t["hits"] >= min_hits:
                out.append((f, t["id"], *t["last"].tolist()))
    return out


def hota(gt, tr, n_frames):
    """HOTA(Luiten 등 2021): α = 0.05~0.95마다 DetA·AssA를 구해 sqrt(DetA·AssA)를 평균. 프레임별 헝가리안 매칭(IoU ≥ α)"""
    from scipy.optimize import linear_sum_assignment
    res = []
    G = {f: [g for g in gt if g[0] == f] for f in range(n_frames)}
    Tt = {f: [t for t in tr if t[0] == f] for f in range(n_frames)}
    for a in np.arange(0.05, 0.96, 0.05):
        matches = []
        TP = FN = FP = 0
        for f in range(n_frames):
            g, t = G[f], Tt[f]
            if g and t:
                M = iou_matrix([x[2:6] for x in g], [x[2:6] for x in t])
                r, c = linear_sum_assignment(-M)
                pm = [(i, j) for i, j in zip(r, c) if M[i, j] >= a]
            else:
                pm = []
            TP += len(pm); FN += len(g) - len(pm); FP += len(t) - len(pm)
            matches += [(g[i][1], t[j][1]) for i, j in pm]
        if TP == 0:
            res.append((0, 0, 0)); continue
        from collections import Counter
        pair = Counter(matches)
        gcount = Counter(x[1] for x in gt)
        tcount = Counter(x[1] for x in tr)
        ass = 0.0
        for (gi, ti), tpa in pair.items():
            fna = gcount[gi] - tpa
            fpa = tcount[ti] - tpa
            ass += tpa * (tpa / (tpa + fna + fpa))
        AssA = ass / TP
        DetA = TP / (TP + FN + FP)
        res.append((np.sqrt(DetA * AssA), DetA, AssA))
    r = np.array(res)
    return dict(HOTA=r4(r[:, 0].mean()), DetA=r4(r[:, 1].mean()), AssA=r4(r[:, 2].mean()))


def exp_mot():
    if not hasattr(np, "asfarray"):      # motmetrics 1.4가 NumPy 2에서 없어진 함수를 씀
        np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
    import motmetrics as mm
    Q = sequence()
    n = Q["n_frames"]
    gtb = []
    for f, tid, c, cx, cy, w, h, th in Q["gt"]:
        hb = np.clip(obb_to_hbb((cx, cy, w, h, th)), 0, TILE)
        gtb.append((f, tid, *hb.tolist()))
    out = dict(n_frames=n, dt=Q["dt"], n_gt_tracks=len(set(g[1] for g in gtb)), n_gt_boxes=len(gtb), n_det=len(Q["det"]))
    # 글린트 구간 놓침
    out["n_det_ge_0_5"] = sum(1 for d in Q["det"] if d[2] >= 0.5)
    for name, kw in {"iou_only": dict(mode="iou"), "sort": dict(mode="sort"), "sort_maxage20": dict(mode="sort", max_age=20),
                     "byte": dict(mode="byte"), "byte_maxage20": dict(mode="byte", max_age=20)}.items():
        tr = run_tracker(Q["det"], n, **kw)
        acc = mm.MOTAccumulator(auto_id=True)
        for f in range(n):
            g = [x for x in gtb if x[0] == f]
            t = [x for x in tr if x[0] == f]
            dist = mm.distances.iou_matrix([[x[2], x[3], x[4] - x[2], x[5] - x[3]] for x in g],
                                           [[x[2], x[3], x[4] - x[2], x[5] - x[3]] for x in t], max_iou=0.5)
            acc.update([x[1] for x in g], [x[1] for x in t], dist)
        mh = mm.metrics.create()
        s = mh.compute(acc, metrics=["mota", "motp", "idf1", "num_switches", "num_false_positives", "num_misses", "num_fragmentations",
                                     "mostly_tracked", "num_unique_objects"], name=name)
        row = {k: (r4(v) if isinstance(v, float) else int(v)) for k, v in s.iloc[0].to_dict().items()}
        row["n_track_ids"] = len(set(x[1] for x in tr))
        row.update(hota(gtb, tr, n))
        out[name] = row
    out["note"] = "MOTA·IDF1·ID 스위치는 motmetrics 1.4(IoU 0.5에서 짝), HOTA는 직접 구현(α 0.05~0.95 평균). 추적기 설정은 run_tracker 참고(점수 높음 0.5, 낮음 0.1, IoU 0.3, max_age 5, min_hits 2)"
    R["mot"] = out


EXPS = {k[4:]: v for k, v in globals().items() if k.startswith("exp_")}

if __name__ == "__main__":
    names = sys.argv[1:] or list(EXPS)
    for nme in names:
        t0 = time.time()
        EXPS[nme]()
        R.setdefault("_meta", {})[nme] = round(time.time() - t0, 1)
        save()
        print(nme, "done", round(time.time() - t0, 1), "s", flush=True)
