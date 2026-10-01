"""6부 공통 평가 도구: IoU 계열, NMS 계열, COCO 방식 AP, 정밀도-재현율, 회전 IoU.

교육용으로 직접 구현함. HBB의 COCO AP는 pycocotools 결과와 맞는지 run_part6.py가 확인함.
상자 HBB = (x1, y1, x2, y2), 회전 박스 OBB = (cx, cy, w, h, theta[도]) — 규약은 make_part6.py 참고.
"""
import numpy as np
from shapely.geometry import Polygon

from make_part6 import obb_corners


# ----------------------------------------------------------------- IoU 계열
def iou_matrix(a, b):
    """HBB 묶음 a (N,4), b (M,4) 사이 IoU 행렬"""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-12)


def iou_family(a, b):
    """상자 두 개의 IoU, GIoU, DIoU, CIoU (CIoU는 Zheng 등 2020 정의)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    iou = float(iou_matrix(a, b)[0, 0])
    cx1, cy1, cx2, cy2 = min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
    c_area = (cx2 - cx1) * (cy2 - cy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = aa + ab - ix * iy
    giou = iou - (c_area - union) / c_area
    rho2 = ((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) ** 2 + ((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) ** 2
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    diou = iou - rho2 / c2
    v = 4 / np.pi ** 2 * (np.arctan((b[2] - b[0]) / (b[3] - b[1])) - np.arctan((a[2] - a[0]) / (a[3] - a[1]))) ** 2
    alpha = v / (1 - iou + v) if (1 - iou + v) > 0 else 0.0
    ciou = diou - alpha * v
    return dict(iou=iou, giou=float(giou), diou=float(diou), ciou=float(ciou))


def poly(o):
    return Polygon(obb_corners(*o))


def riou(o1, o2):
    """회전 박스 두 개의 IoU (다각형 교집합)"""
    p, q = poly(o1), poly(o2)
    inter = p.intersection(q).area
    return inter / (p.area + q.area - inter)


def riou_matrix(A, B):
    P = [poly(o) for o in A]
    Q = [poly(o) for o in B]
    M = np.zeros((len(P), len(Q)))
    for i, p in enumerate(P):
        bp = p.bounds
        for j, q in enumerate(Q):
            bq = q.bounds
            if bp[0] > bq[2] or bq[0] > bp[2] or bp[1] > bq[3] or bq[1] > bp[3]:
                continue
            inter = p.intersection(q).area
            if inter > 0:
                M[i, j] = inter / (p.area + q.area - inter)
    return M


# ----------------------------------------------------------------- NMS 계열
def nms(boxes, scores, thr=0.5, iou_fn=None):
    """탐욕적 NMS. 남길 인덱스(점수 내림차순)"""
    boxes, scores = np.asarray(boxes, float), np.asarray(scores, float)
    order = np.argsort(-scores, kind="stable")
    keep = []
    if iou_fn is None:
        M = iou_matrix(boxes, boxes)
    else:
        M = iou_fn(boxes, boxes)
    alive = np.ones(len(boxes), bool)
    for i in order:
        if not alive[i]:
            continue
        keep.append(int(i))
        alive &= ~(M[i] > thr)
        alive[i] = False
    return keep


def nms_dets(dets, thr=0.5, agnostic=False, rotated=False):
    """dets(list of dict)에 클래스별(또는 클래스 무관) NMS"""
    key = "obb" if rotated else "hbb"
    fn = riou_matrix if rotated else None
    out = []
    groups = [None] if agnostic else sorted(set(d["cls"] for d in dets))
    for g in groups:
        sub = [d for d in dets if agnostic or d["cls"] == g]
        if not sub:
            continue
        k = nms([d[key] for d in sub], [d["score"] for d in sub], thr, fn)
        out += [sub[i] for i in k]
    return out


def soft_nms_dets(dets, sigma=0.5, score_thr=0.001):
    """가우시안 Soft-NMS(Bodla 등 2017): 겹친 상자를 지우지 않고 점수를 exp(-IoU²/σ)배로 낮춤. 클래스별"""
    out = []
    for g in sorted(set(d["cls"] for d in dets)):
        sub = [dict(d) for d in dets if d["cls"] == g]
        B = np.array([d["hbb"] for d in sub])
        S = np.array([d["score"] for d in sub], float)
        idx = list(range(len(sub)))
        while idx:
            m = max(idx, key=lambda i: S[i])
            idx.remove(m)
            if S[m] < score_thr:
                continue
            d = dict(sub[m]); d["score"] = float(S[m]); out.append(d)
            if idx:
                ious = iou_matrix(B[m:m + 1], B[idx])[0]
                S[idx] = S[idx] * np.exp(-(ious ** 2) / sigma)
    return out


def wbf_dets(dets, thr=0.55, size=512):
    """가중 박스 융합(Solovyev 등 2019, ensemble_boxes 구현): 겹친 상자를 점수 가중 평균으로 합침"""
    from ensemble_boxes import weighted_boxes_fusion
    if not dets:
        return []
    B = np.clip(np.array([d["hbb"] for d in dets]) / size, 0, 1)
    S = [d["score"] for d in dets]
    L = [d["cls"] for d in dets]
    b, s, l = weighted_boxes_fusion([B.tolist()], [S], [L], iou_thr=thr, skip_box_thr=0.0)
    return [dict(cls=int(c), score=float(sc), hbb=(bb * size).tolist()) for bb, sc, c in zip(b, s, l)]


# ----------------------------------------------------------------- COCO 방식 평가
AREA = {"all": (0, 1e10), "small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}
IOU_THRS = np.round(np.arange(0.5, 0.951, 0.05), 2)
REC_PTS = np.linspace(0, 1, 101)


def _match(gts, dets, thr, rotated=False, area_rng=(0, 1e10)):
    """한 영상·한 클래스. COCO 방식 탐욕 매칭(점수 순, 아직 짝 없는 정답 중 IoU 가장 큰 것).
    크기 구간 밖 정답은 '무시'로 두고(크기는 "size" 키가 있으면 그 값, 없으면 넓이), 무시 정답과 짝지어진 탐지나 크기 구간 밖의 짝 없는 탐지는 셈에서 뺌(COCO 규칙)"""
    key = "obb" if rotated else "hbb"
    g_ign = np.array([not (area_rng[0] <= g.get("size", g["area"]) < area_rng[1]) for g in gts], bool)
    order_g = np.argsort(g_ign, kind="stable")            # 무시하지 않는 정답을 먼저
    gts = [gts[i] for i in order_g]
    g_ign = g_ign[order_g]
    dets = sorted(dets, key=lambda d: -d["score"])
    if gts and dets:
        M = riou_matrix([d[key] for d in dets], [g[key] for g in gts]) if rotated else \
            iou_matrix([d[key] for d in dets], [g[key] for g in gts])
    else:
        M = np.zeros((len(dets), len(gts)))
    gm = np.full(len(gts), -1)
    tp = np.zeros(len(dets), bool)
    ign = np.zeros(len(dets), bool)
    for di in range(len(dets)):
        best, bj = min(thr, 1 - 1e-10), -1
        for gj in range(len(gts)):
            if gm[gj] >= 0:
                continue
            if bj > -1 and not g_ign[bj] and g_ign[gj]:
                break
            if M[di, gj] < best:
                continue
            best, bj = M[di, gj], gj
        if bj == -1:
            continue
        gm[bj] = di
        ign[di] = g_ign[bj]
        tp[di] = True
    # 짝 없는 탐지 중 크기 구간 밖인 것은 무시
    for di, d in enumerate(dets):
        if not tp[di]:
            hb = d["hbb"]
            a = d.get("size", (hb[2] - hb[0]) * (hb[3] - hb[1]))
            if not (area_rng[0] <= a < area_rng[1]):
                ign[di] = True
    return [d["score"] for d in dets], tp, ign, int((~g_ign).sum())


def pr_points(images, cls, thr=0.5, rotated=False, area="all", max_dets=100):
    """images: list of (gts, dets) per image. 반환: 점수 내림차순 (scores, tp, n_gt)"""
    S, T, I, n_gt = [], [], [], 0
    for gts, dets in images:
        g = [x for x in gts if x["cls"] == cls]
        d = sorted([x for x in dets if x["cls"] == cls], key=lambda x: -x["score"])[:max_dets]
        s, tp, ign, n = _match(g, d, thr, rotated, AREA[area] if isinstance(area, str) else area)
        S += s; T += tp.tolist(); I += ign.tolist(); n_gt += n
    S, T, I = np.array(S), np.array(T, bool), np.array(I, bool)
    o = np.argsort(-S, kind="mergesort")
    S, T, I = S[o], T[o], I[o]
    return S[~I], T[~I], n_gt


def ap_from_pr(tp, n_gt, method="coco"):
    """AP. coco: 재현율 0~1을 101점으로 보고 각 점 이상에서의 최대 정밀도(보간) 평균
    voc11: 11점 보간(VOC2007), voc: 재현율이 바뀌는 모든 점(VOC2010 이후)"""
    if n_gt == 0:
        return float("nan")
    tpc = np.cumsum(tp)
    fpc = np.cumsum(~tp)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-12)
    if method == "voc":
        mrec = np.concatenate([[0], rec, [1]])
        mpre = np.concatenate([[0], prec, [0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    pts = np.linspace(0, 1, 11) if method == "voc11" else REC_PTS
    env = prec.copy()
    for i in range(len(env) - 2, -1, -1):
        env[i] = max(env[i], env[i + 1])
    out = []
    for r in pts:
        k = np.searchsorted(rec, r, side="left")
        out.append(env[k] if k < len(env) else 0.0)
    return float(np.mean(out))


def coco_eval(images, classes=(0, 1, 2), rotated=False, area="all", max_dets=100, thrs=IOU_THRS):
    """클래스별·IoU 임계값별 AP 표와 요약(mAP50, mAP75, mAP50-95; 매크로 평균, 정답 없는 클래스는 뺌)"""
    tab = {}
    for c in classes:
        row = []
        for t in thrs:
            _, tp, n = pr_points(images, c, t, rotated, area, max_dets)
            row.append(ap_from_pr(tp, n))
        tab[c] = row
    A = np.array([tab[c] for c in classes], float)
    def mean(x):
        x = x[~np.isnan(x)]
        return float(x.mean()) if len(x) else float("nan")
    t50, t75 = list(np.round(thrs, 2)).index(0.5), list(np.round(thrs, 2)).index(0.75) if 0.75 in np.round(thrs, 2) else None
    return dict(per_class={CLS: [round(v, 4) for v in tab[CLS]] for CLS in classes},
                ap50=round(mean(A[:, t50]), 4), ap75=round(mean(A[:, t75]), 4) if t75 is not None else None,
                ap=round(mean(np.array([mean(A[:, k]) for k in range(A.shape[1])])), 4))


def operating_point(images, cls, score_thr, iou_thr=0.5):
    """점수 임계값에서 TP·FP·FN, 정밀도·재현율·F1, 영상당 오탐 수(FPPI)"""
    tp = fp = n_gt = 0
    for gts, dets in images:
        g = [x for x in gts if x["cls"] == cls]
        d = [x for x in dets if x["cls"] == cls and x["score"] >= score_thr]
        _, t, ign, n = _match(g, d, iou_thr)
        tp += int(t.sum()); fp += int((~t).sum()); n_gt += n
    fn = n_gt - tp
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / n_gt if n_gt else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
                fppi=round(fp / len(images), 3))
