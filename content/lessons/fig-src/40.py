"""40강 그림과 본문 수치: 분할 지표 비교(SVG 점그림), 시험 타일 하나의 화소·경계 판정(PNG)

- 대표 값은 data/part6_runs.json의 seg(run_part6.py exp_seg)에서 읽음
- JSON에 없는 값(영상별 평균 mIoU, 카파의 우연 일치율, 클래스별 면적 오차, 손 계산 예)은 여기서 계산해 출력함
- U-Net 예측은 run_part5.py unet이 저장한 best 가중치(data/.part5_unet_*.pt, git 제외)로 시험 타일을 추론해 얻음.
  가중치가 없으면 손 계산 예와 SVG만 만들고 PNG는 건너뜀(학습은 하지 않음)
"""
import json, os, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "fig")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg

S = json.load(open(os.path.join(DATA, "part6_runs.json"), encoding="utf-8"))["seg"]


def edges(m):
    """경계 화소: 3×3 이웃에 다른 라벨이 있음(경계 양쪽 한 화소씩) — run_part6.boundary_f1과 같은 정의"""
    return (m != ndi.maximum_filter(m, 3)) | (m != ndi.minimum_filter(m, 3))


def boundary_f1(pred, gt, tol=1):
    ep, eg = edges(pred), edges(gt)
    dp = ndi.binary_dilation(ep, iterations=tol) if tol else ep
    dg = ndi.binary_dilation(eg, iterations=tol) if tol else eg
    P = (ep & dg).sum() / max(ep.sum(), 1)
    R = (eg & dp).sum() / max(eg.sum(), 1)
    return 2 * P * R / (P + R) if P + R else 0.0


def scores(cm):
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    po = tp.sum() / cm.sum()
    pe = (cm.sum(0) * cm.sum(1)).sum() / cm.sum() ** 2
    iou = tp / (tp + fp + fn)
    return dict(pa=po, iou=iou, miou=iou.mean(), dice=2 * tp / (2 * tp + fp + fn), pe=pe, kappa=(po - pe) / (1 - pe))


# ---------------------------------------------------------------- 1. 손 계산 예
print("== 바다 90% 연안 영상(화소 100만: 바다 90만, 육지 8만, 양식장 2만; 행 = 참, 열 = 예측)")
A = np.array([[900000, 0, 0], [80000, 0, 0], [20000, 0, 0]], float)            # 전부 바다로 칠함
B = np.array([[895000, 3000, 2000], [4000, 76000, 0], [10000, 0, 10000]], float)  # 양식장 절반을 바다로 놓침
for k, cm in (("A 전부 바다", A), ("B 양식장 절반 놓침", B)):
    r = scores(cm)
    print(f"  {k}: PA {r['pa']:.4f} IoU {np.round(r['iou'], 4)} mIoU {r['miou']:.4f} 카파 {r['kappa']:.4f} (pe {r['pe']:.4f}) "
          f"면적비(예측/참) {np.round(cm.sum(0) / cm.sum(1), 3)}")

print("== 100×100 정사각형을 오른쪽으로 2화소 밀었을 때")
g = np.zeros((140, 140), int); g[12:112, 12:112] = 1
p = np.zeros_like(g); p[12:112, 14:114] = 1
print(f"  IoU {(g & p).sum() / (g | p).sum():.4f}  경계 F1 허용 0·1·2: {[round(boundary_f1(p, g, t), 3) for t in (0, 1, 2)]}")

print("== Dice = 2IoU/(1+IoU)")
print("  수계(스킵 있음) IoU", S["skip"]["iou"][3], "→", round(2 * S["skip"]["iou"][3] / (1 + S["skip"]["iou"][3]), 4), "/ JSON", S["dice_from_iou_check"])
for nm in ("skip", "noskip"):
    mi = S[nm]["miou"]
    print(f"  {nm}: 평균 Dice {S[nm]['mean_dice']} vs 2mIoU/(1+mIoU) {2 * mi / (1 + mi):.4f}")
print("  Dice − IoU 최대 차이", round(3 - 2 * np.sqrt(2), 4), "(IoU = √2 − 1 ≈ 0.414)")
print("== 마스크 AP(회전 박스 다각형) vs HBB AP")
mp = json.load(open(os.path.join(DATA, "part6_runs.json"), encoding="utf-8"))["map"]["coco"]
print("  마스크", S["mask_ap_obb_polygons"], " HBB", dict(AP=mp["ap"], AP50=mp["ap50"], AP75=mp["ap75"]))

# ---------------------------------------------------------------- 2. 지표 비교 점그림(SVG)
W_SKIP = os.path.join(DATA, ".part5_unet_skip.pt")
W_NOSKIP = os.path.join(DATA, ".part5_unet_noskip.pt")
HAVE = os.path.exists(W_SKIP) and os.path.exists(W_NOSKIP)
PER_IMAGE = None
if HAVE:
    import torch
    torch.set_num_threads(1)
    from part5_common import data, TinyUNet
    from make_part5 import load
    D = data()
    y = D["m_test"].numpy()
    PRED = {}
    for nm, skip, wp in (("skip", True, W_SKIP), ("noskip", False, W_NOSKIP)):
        net = TinyUNet(skip=skip); net.load_state_dict(torch.load(wp)); net.eval()
        with torch.no_grad():
            PRED[nm] = torch.cat([net(D["x_test"][i:i + 300]).argmax(1) for i in range(0, 900, 300)]).numpy()
    PER_IMAGE = {}
    print("== 5부 U-Net 시험 타일 900장")
    for nm, pred in PRED.items():
        cm = np.bincount(y.ravel() * 6 + pred.ravel(), minlength=36).reshape(6, 6).astype(float)
        r = scores(cm)
        q = 0.5 * np.abs(cm.sum(0) - cm.sum(1)).sum() / cm.sum()

        def per_tile(i, gt_only=False):
            cls = np.unique(y[i]) if gt_only else np.union1d(np.unique(y[i]), np.unique(pred[i]))
            return np.mean([((pred[i] == c) & (y[i] == c)).sum() / ((pred[i] == c) | (y[i] == c)).sum() for c in cls])
        pim = np.array([per_tile(i) for i in range(len(y))])
        pim_gt = np.mean([per_tile(i, True) for i in range(len(y))])
        pure = np.array([len(np.unique(y[i])) == 1 for i in range(len(y))])
        stray = sum(len(np.unique(pred[i])) > 1 for i in np.where(pure)[0])
        PER_IMAGE[nm] = pim.mean()
        acc_t = (pred == y).reshape(len(y), -1).mean(1)
        print(f"  {nm}: 정확도 0.5 미만 타일 {(acc_t < 0.5).sum()}장")
        print(f"  {nm}: PA {r['pa']:.4f} mIoU(합산) {r['miou']:.4f} mIoU(영상별, 정답·예측 클래스) {pim.mean():.4f} "
              f"(정답 클래스만) {pim_gt:.4f} | 한 클래스 타일 {pure.sum()}장 중 다른 클래스가 섞여 칠해진 타일 {stray}장, "
              f"한 클래스 타일 평균 {pim[pure].mean():.4f} 여러 클래스 타일 평균 {pim[~pure].mean():.4f}")
        print(f"    카파 {r['kappa']:.4f} pe {r['pe']:.4f} | 양 불일치 {q:.4f} 배치 {1 - r['pa'] - q:.4f} | "
              f"면적 오차(예측/참 − 1) {np.round(cm.sum(0) / cm.sum(1) - 1, 4)}")

rows = [("픽셀 정확도", "pixel_acc"), ("mIoU(전체 합산)", "miou"), ("mIoU(영상별 평균)", None), ("평균 Dice", "mean_dice"),
        ("카파", "kappa"), ("경계 F1 허용 0", "b0"), ("경계 F1 허용 1", "b1"), ("경계 F1 허용 2", "b2")]


def val(nm, key):
    if key is None:
        return PER_IMAGE[nm] if PER_IMAGE else None
    if key.startswith("b"):
        return S[nm]["boundary_f1_mixed_tiles"][key[1]]
    return S[nm][key]


def f3(v):  # 본문과 같게 사사오입(JSON 넷째 자리 값 0.8955 → 0.896)
    return str(Decimal(str(round(float(v), 4))).quantize(Decimal('0.001'), ROUND_HALF_UP))


if PER_IMAGE is None:
    rows = [r for r in rows if r[1] is not None]
X0, X1, V0, V1 = 190, 590, 0.5, 1.0
TOP, DY = 64, 34
s = Svg(760, TOP + DY * len(rows) + 40, "U-Net 스킵 연결 있음·없음의 분할 지표 비교. 픽셀 정확도·mIoU·Dice·카파는 0.90~0.98로 "
        "모여 있고, 경계 F1은 0.53~0.90으로 낮고 두 모델 차이도 더 큼")
xs = lambda v: X0 + (v - V0) / (V1 - V0) * (X1 - X0)
# 범례
s.circle(X0 + 4, 24, 6, "f-ac"); s.text(X0 + 16, 28.5, "스킵 있음", "f-fg", 12, anchor="start")
s.circle(X0 + 110, 24, 6, "f-bd"); s.text(X0 + 122, 28.5, "스킵 없음", "f-fg", 12, anchor="start")
s.text(655, 28.5, "있음", "f-ac", 12); s.text(715, 28.5, "없음", "f-bd", 12)
ybot = TOP + DY * (len(rows) - 1) + 16
for k in range(6):
    v = V0 + 0.1 * k
    s.line(xs(v), TOP - 16, xs(v), ybot, "s-mu", 0.6, dash="3 3")
    s.text(xs(v), ybot + 18, f"{v:.1f}", "f-mu", 11)
for j, (lab, key) in enumerate(rows):
    yy = TOP + DY * j
    a, b = val("skip", key), val("noskip", key)
    s.text(X0 - 12, yy + 4.5, lab, "f-fg", 12, anchor="end")
    s.line(xs(b), yy, xs(a), yy, "s-mu", 2)
    s.circle(xs(b), yy, 6, "f-bd")
    s.circle(xs(a), yy, 6, "f-ac")
    s.text(655, yy + 4.5, f3(a), "f-fg", 12)
    s.text(715, yy + 4.5, f3(b), "f-fg", 12)
    if j == 4:  # 면적 지표와 경계 지표 사이 구분선
        s.line(X0 - 170, yy + DY / 2, 740, yy + DY / 2, "s-mu", 0.8)
s.save(os.path.join(OUT, "40-metrics.svg"))

# ---------------------------------------------------------------- 3. 시험 타일 하나의 화소·경계 판정(PNG)
if not HAVE:
    print("U-Net 가중치가 없음: content/lessons/data에서 'python3 run_part5.py unet'을 먼저 돌리라. PNG 생성을 건너뜀")
    sys.exit(0)
from PIL import Image

RAW = load()
pred = PRED["noskip"]
cand = []
for i in range(len(y)):
    u = np.unique(y[i])
    if len(u) != 2:
        continue
    acc = (pred[i] == y[i]).mean()
    minor = u[np.argmin([(y[i] == c).sum() for c in u])]
    frac = (y[i] == minor).mean()
    if acc < 0.88 or acc > 0.97 or not 0.25 < frac < 0.5 or len(np.unique(pred[i])) > 3 or set(u.tolist()) == {3, 5}:
        continue  # 수계·갯벌 쌍은 타일 통째 혼동이 섞여 경계 예로 부적당
    cand.append((boundary_f1(pred[i], y[i], 1), i, int(minor)))
cand.sort()
_, T, C = cand[len(cand) // 4]                 # 경계가 꽤 틀린(아래 4분위 근처) 타일
gt, pr = y[T], pred[T]
print(f"그림 타일 {T}: 클래스 {[RAW['classes'][c] for c in np.unique(gt)]} 대상 클래스 {RAW['classes'][C]} "
      f"정확도 {(pr == gt).mean():.3f} IoU {((pr == C) & (gt == C)).sum() / ((pr == C) | (gt == C)).sum():.3f} "
      f"경계 F1 0·1·2 {[round(boundary_f1(pr, gt, t), 3) for t in (0, 1, 2)]} (스킵 있음 {[round(boundary_f1(PRED['skip'][T], gt, t), 3) for t in (0, 1, 2)]})")

PAL = np.array([[27, 110, 50], [176, 214, 92], [214, 60, 60], [40, 95, 210], [200, 160, 100], [150, 130, 175]], np.uint8)
fc = RAW["x_test"][T][[3, 2, 1]]
lo, hi = np.percentile(fc, 1, axis=(1, 2)), np.percentile(fc, 99, axis=(1, 2))
rgb = (np.clip((fc - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1) ** 0.8 * 255).round().astype(np.uint8).transpose(1, 2, 0)
# 화소 판정(대상 클래스 C): TP 파랑, FP 주황, FN 빨강, TN 연회색
pix = np.full(gt.shape + (3,), 222, np.uint8)
pix[(pr == C) & (gt == C)] = [40, 95, 210]
pix[(pr == C) & (gt != C)] = [245, 150, 30]
pix[(pr != C) & (gt == C)] = [205, 30, 30]
# 경계 판정(허용 1화소): 허용 띠 하늘색, 놓친 정답 경계 빨강, 띠 안 예측 경계 초록, 띠 밖 예측 경계 주황
eg, ep = edges(gt), edges(pr)
band = ndi.binary_dilation(eg, iterations=1)
reach = ndi.binary_dilation(ep, iterations=1)
bd = np.full(gt.shape + (3,), 238, np.uint8)
bd[band] = [185, 210, 240]
bd[eg & ~reach] = [205, 30, 30]
bd[ep & band] = [20, 140, 60]
bd[ep & ~band] = [245, 150, 30]
SC, G = 7, 10
N = 32 * SC
panels = [rgb, PAL[gt], PAL[pr], pix, bd]
canvas = np.zeros((N, len(panels) * N + (len(panels) - 1) * G, 4), np.uint8)
for k, pnl in enumerate(panels):
    c0 = k * (N + G)
    canvas[:, c0:c0 + N, :3] = pnl.repeat(SC, 0).repeat(SC, 1)
    canvas[:, c0:c0 + N, 3] = 255
Image.fromarray(canvas, "RGBA").quantize(colors=64, method=Image.Quantize.FASTOCTREE).save(os.path.join(OUT, "40-tile.png"), optimize=True)
print("ok")
