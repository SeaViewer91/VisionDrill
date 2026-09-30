"""33강 그림과 본문 수치: 백본·넥(FPN)·헤드 도식(SVG), 작은 U-Net 도식(SVG), 분할 결과 예시(PNG)

- 분할 결과 PNG는 run_part5.py unet이 저장한 best 가중치(data/.part5_unet_skip.pt, .part5_unet_noskip.pt, git 제외)로
  시험 타일을 추론해 만듦(학습은 하지 않음). 가중치가 없으면 PNG는 건너뜀
- 본문 수치 중 JSON(part5_runs.json)에 없는 것(검증 지역의 경계 정확도, 타일 통째 혼동 수, torchvision 부분별
  파라미터, 헝가리안 매칭 예, RoI 풀링 반올림 예)도 여기서 계산해 출력함
"""
import math, os, sys
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg

OUT = os.path.join(HERE, "..", "fig")


def arrow(s, x1, y1, x2, y2, cls="s-fg", fcls="f-fg", width=1.4, head=6, dash=None):
    """직선 화살표(머리는 채운 삼각형 경로, marker를 쓰지 않음: id 금지)"""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    bx, by = x2 - ux * head, y2 - uy * head
    s.line(x1, y1, bx, by, cls, width, dash)
    px, py = -uy * head * 0.55, ux * head * 0.55
    s.path(f"M {x2:.1f} {y2:.1f} L {bx + px:.1f} {by + py:.1f} L {bx - px:.1f} {by - py:.1f} Z", fcls, 0)


def grid_square(s, cx, cy, side, n, cls="s-fg f-acs"):
    """격자 느낌의 정사각형(칸 n×n을 흐린 선으로)"""
    x0, y0 = cx - side / 2, cy - side / 2
    s.rect(x0, y0, side, side, cls, 1.3)
    for k in range(1, n):
        s.line(x0 + side * k / n, y0, x0 + side * k / n, y0 + side, "s-mu", 0.5)
        s.line(x0, y0 + side * k / n, x0 + side, y0 + side * k / n, "s-mu", 0.5)


# ---------------------------------------------------------------- 1. 백본·넥(FPN)·헤드
s = Svg(780, 340, "640×640 입력에서 백본이 스트라이드 8·16·32의 특징 맵(80×80, 40×40, 20×20)을 내고, "
                  "넥(FPN)이 깊은 층부터 두 배씩 업샘플해 옆 연결(1×1 합성곱)과 더한 뒤, 층마다 헤드가 격자 칸마다 클래스와 박스를 냄")
ROWS = {5: 78, 4: 168, 3: 266}               # C5·C4·C3 줄의 세로 중심
SIDE = {5: 26, 4: 40, 3: 60}                 # 그림 크기(실제 비율은 1:2:4)
GRID = {5: 20, 4: 40, 3: 80}
NCELL = {5: 2, 4: 4, 3: 8}                   # 그림 안에 그리는 격자 칸 수(설명용)
XB, XN, XH, XO = 235, 440, 585, 640
for x, lab in [(XB, "백본"), (XN, "넥(FPN)"), (XH, "헤드"), (XO + 55, "출력")]:
    s.text(x, 28, lab, "f-fg", 13, weight="600")
# 입력
IX, IY, IS = 62, ROWS[3], 76
s.rect(IX - IS / 2, IY - IS / 2, IS, IS, "s-fg f-sf", 1.3)
s.text(IX, IY + IS / 2 + 18, "입력 640×640", "f-mu", 11)
arrow(s, IX + IS / 2 + 4, IY, XB - SIDE[3] / 2 - 4, IY, "s-mu", "f-mu", 1.3)
for lv in (3, 4, 5):
    y, sd = ROWS[lv], SIDE[lv]
    grid_square(s, XB, y, sd, NCELL[lv])
    grid_square(s, XN, y, sd, NCELL[lv], "s-ac f-acs")
    # 백본 이름·크기(가로 화살표 위)
    s.text(XB + sd / 2 + 8, y - 22, f"C{lv} · {GRID[lv]}×{GRID[lv]}", "f-fg", 11, anchor="start")
    s.text(XB + sd / 2 + 8, y - 8, f"스트라이드 {2 ** lv}", "f-mu", 11, anchor="start")
    # 옆 연결
    arrow(s, XB + sd / 2 + 4, y, XN - sd / 2 - 4, y, "s-fg", "f-fg", 1.3)
    s.text((XB + XN) / 2 + 8, y + 16, "1×1", "f-mu", 11)
    s.text(XN, y + sd / 2 + 14, f"P{lv}", "f-ac", 11, weight="600")
    # 헤드와 출력
    arrow(s, XN + sd / 2 + 4, y, XH - 30, y, "s-mu", "f-mu", 1.2)
    s.rect(XH - 30, y - 13, 60, 26, "s-fg f-bds", 1.2, rx=4)
    s.text(XH, y + 4.5, "헤드", "f-fg", 11)
    arrow(s, XH + 30, y, XO - 4, y, "s-mu", "f-mu", 1.2)
    s.text(XO, y - 4, f"{GRID[lv]}×{GRID[lv]}칸", "f-fg", 11, anchor="start")
    s.text(XO, y + 11, "칸마다 K + 4", "f-mu", 11, anchor="start")
# 백본: 아래에서 위로(해상도 절반씩)
arrow(s, XB, ROWS[3] - SIDE[3] / 2 - 2, XB, ROWS[4] + SIDE[4] / 2 + 2, "s-fg", "f-fg", 1.3)
arrow(s, XB, ROWS[4] - SIDE[4] / 2 - 2, XB, ROWS[5] + SIDE[5] / 2 + 2, "s-fg", "f-fg", 1.3)
# 넥: 위에서 아래로(두 배 업샘플 후 더함)
arrow(s, XN - 14, ROWS[5] + SIDE[5] / 2 + 2, XN - 14, ROWS[4] - SIDE[4] / 2 - 2, "s-ac", "f-ac", 1.5)
arrow(s, XN - 14, ROWS[4] + SIDE[4] / 2 + 2, XN - 14, ROWS[3] - SIDE[3] / 2 - 2, "s-ac", "f-ac", 1.5)
for a, b in [(5, 4), (4, 3)]:
    ym = (ROWS[a] + SIDE[a] / 2 + ROWS[b] - SIDE[b] / 2) / 2
    s.text(XN - 20, ym + 4, "×2 업샘플", "f-ac", 11, anchor="end")
s.text(XN + 6, (ROWS[5] + SIDE[5] / 2 + ROWS[4] - SIDE[4] / 2) / 2 + 4, "+ 옆 연결", "f-mu", 11, anchor="start")
s.text(XN + 6, (ROWS[4] + SIDE[4] / 2 + ROWS[3] - SIDE[3] / 2) / 2 + 4, "+ 옆 연결", "f-mu", 11, anchor="start")
s.save(os.path.join(OUT, "33-fpn.svg"))

# ---------------------------------------------------------------- 2. 작은 U-Net 도식
s = Svg(780, 340, "작은 U-Net. 인코더가 32×32 → 16×16 → 8×8로 줄이며 채널을 16 → 32 → 64로 늘리고, 디코더가 전치 합성곱으로 "
                  "두 배씩 키운 뒤 같은 해상도의 인코더 특징을 이어 붙임(스킵 연결). 끝의 1×1 합성곱이 화소마다 로짓 6개를 냄")
H = {32: 96, 16: 48, 8: 24}                  # 한 변 → 그림 높이
W = {4: 6, 6: 8, 16: 12, 32: 20, 64: 34}     # 채널 → 그림 너비
LY = {32: 102, 16: 212, 8: 290}               # 해상도별 세로 중심


def block(cx, n, c, cls="s-fg f-acs", label=None):
    s.rect(cx - W[c] / 2, LY[n] - H[n] / 2, W[c], H[n], cls, 1.2)
    if label is not None:
        s.text(cx, LY[n] + H[n] / 2 + 15, label, "f-mu", 11)


XIN, XE1, XE2, XE3, XD2, XD1, XOUT = 40, 118, 212, 390, 560, 650, 735
block(XIN, 32, 4, "s-fg f-sf", "4×32×32")
s.text(XIN, LY[32] - H[32] / 2 - 8, "입력", "f-fg", 11)
block(XE1, 32, 16, label="16×32×32")
block(XE2, 16, 32, label="32×16×16")
block(XE3, 8, 64, label="64×8×8")
# 디코더: 이어 붙인 부분(주황) + 업샘플한 부분(파랑)
for xc, n, c in [(XD2, 16, 32), (XD1, 32, 16)]:
    s.rect(xc - W[c], LY[n] - H[n] / 2, W[c], H[n], "s-bd f-bds", 1.2)
    s.rect(xc, LY[n] - H[n] / 2, W[c], H[n], "s-fg f-acs", 1.2)
s.text(XD2 + W[32] + 6, LY[16] + 4, "32+32 → 32", "f-mu", 11, anchor="start")
s.text(XD1, LY[32] - H[32] / 2 - 8, "16+16 → 16", "f-mu", 11)
block(XOUT, 32, 6, "s-ac f-acs", "6×32×32")
s.text(XOUT, LY[32] - H[32] / 2 - 8, "로짓", "f-fg", 11)
s.text(XE1, 22, "인코더", "f-fg", 13, weight="600")
s.text(XD1, 22, "디코더", "f-fg", 13, weight="600")
arrow(s, XIN + 5, LY[32], XE1 - W[16] / 2 - 3, LY[32], "s-mu", "f-mu", 1.2)
arrow(s, XD1 + W[16] + 3, LY[32], XOUT - W[6] / 2 - 3, LY[32], "s-mu", "f-mu", 1.2)
s.text((XD1 + W[16] + XOUT) / 2, LY[32] - 8, "1×1", "f-mu", 11)
# 내려가기(최대 풀링), 올라가기(전치 합성곱)
arrow(s, XE1 + 4, LY[32] + H[32] / 2 + 22, XE2 - 6, LY[16] - H[16] / 2 - 4, "s-fg", "f-fg", 1.3)
s.text((XE1 + XE2) / 2 - 10, (LY[32] + H[32] / 2 + 22 + LY[16] - H[16] / 2) / 2 + 16, "풀링", "f-fg", 11, anchor="end")
arrow(s, XE2 + 10, LY[16] + H[16] / 2 + 22, XE3 - W[64] / 2 - 4, LY[8] - 2, "s-fg", "f-fg", 1.3)
s.text((XE2 + XE3) / 2 - 8, (LY[16] + H[16] / 2 + 22 + LY[8]) / 2 + 16, "풀링", "f-fg", 11, anchor="end")
arrow(s, XE3 + W[64] / 2 + 4, LY[8] - 2, XD2 - W[32] - 6, LY[16] + H[16] / 2 + 4, "s-ac", "f-ac", 1.5)
s.text((XE3 + XD2) / 2 + 12, (LY[16] + H[16] / 2 + LY[8]) / 2 + 18, "전치 합성곱", "f-ac", 11, anchor="start")
arrow(s, XD2 + W[32] + 2, LY[16] - H[16] / 2 - 4, XD1 - W[16] - 4, LY[32] + H[32] / 2 + 4, "s-ac", "f-ac", 1.5)
s.text((XD2 + XD1) / 2 + 14, (LY[16] - H[16] / 2 + LY[32] + H[32] / 2) / 2 + 12, "전치 합성곱", "f-ac", 11, anchor="start")
# 스킵 연결(가로 점선)
for n, xa, ca, xb, cb in [(32, XE1, 16, XD1, 16), (16, XE2, 32, XD2, 32)]:
    arrow(s, xa + W[ca] / 2 + 3, LY[n], xb - W[cb] - 3, LY[n], "s-bd", "f-bd", 1.5, dash="6 4")
    s.text((xa + xb) / 2, LY[n] - 8, "스킵 연결(이어 붙임)", "f-bd", 11)
s.save(os.path.join(OUT, "33-unet.svg"))

# ---------------------------------------------------------------- 3. 본문 수치 확인(손 계산 예)
from scipy.optimize import linear_sum_assignment

C = np.array([[0.2, 0.4], [0.3, 1.5], [1.4, 1.3]])   # 쿼리 3개 × 정답(선박 A, B) 비용
r, c = linear_sum_assignment(C)
print("헝가리안:", list(zip(r.tolist(), c.tolist())), "합", C[r, c].sum(), "/ 욕심쟁이(A 먼저) 합", 0.2 + 1.3)
print("RoI 풀링 반올림(스트라이드 16):", 203 / 16, "→", round(203 / 16) * 16, "/", 331 / 16, "→", round(331 / 16) * 16)
print("FPN 격자(640):", [640 // st for st in (8, 16, 32)], "칸 합", sum((640 // st) ** 2 for st in (8, 16, 32)))

try:
    import torchvision.models as M
    import torchvision.models.detection as TD
    n = lambda m: sum(p.numel() for p in m.parameters())
    r50 = M.resnet50()
    body = n(r50) - n(r50.fc)
    for name, f in [("faster", TD.fasterrcnn_resnet50_fpn), ("mask", TD.maskrcnn_resnet50_fpn),
                    ("retina", TD.retinanet_resnet50_fpn), ("fcos", TD.fcos_resnet50_fpn)]:
        m = f(weights=None, weights_backbone=None)
        parts = {k: n(v) for k, v in m.named_children() if n(v)}
        parts["body"], parts["fpn"] = n(m.backbone.body), n(m.backbone.fpn)
        print(name, n(m), "백본 비중", round(body / n(m), 3), parts)
except Exception as e:  # torchvision이 없으면 건너뜀
    print("torchvision 확인 건너뜀:", e)

# ---------------------------------------------------------------- 4. 분할 결과 예시(PNG) + 경계 정확도
W_SKIP = os.path.join(DATA, ".part5_unet_skip.pt")
W_NOSKIP = os.path.join(DATA, ".part5_unet_noskip.pt")
if not (os.path.exists(W_SKIP) and os.path.exists(W_NOSKIP)):
    print("U-Net 가중치가 없음: content/lessons/data에서 'python3 run_part5.py unet'을 먼저 돌리라. PNG 생성을 건너뜀")
    sys.exit(0)

import torch
from PIL import Image
from scipy import ndimage as ndi
from part5_common import data, TinyUNet, evaluate, miou
from make_part5 import load

D = data()
RAW = load()


def edges(m):
    """참 라벨 경계 화소(3×3 이웃에 다른 라벨이 있음)"""
    e = np.zeros(m.shape, bool)
    for i, mm in enumerate(m):
        e[i] = (mm != ndi.maximum_filter(mm, 3)) | (mm != ndi.minimum_filter(mm, 3))
    return e


PRED = {}
for split in ["val", "test"]:
    m = D[f"m_{split}"].numpy()
    e = edges(m)
    for name, skip, wpath in [("skip", True, W_SKIP), ("noskip", False, W_NOSKIP)]:
        net = TinyUNet(skip=skip)
        net.load_state_dict(torch.load(wpath))
        _, acc, pred = evaluate(net, D[f"x_{split}"], D[f"m_{split}"])
        pred = pred.numpy()
        ok = pred == m
        mi, ious = miou(torch.tensor(pred), torch.tensor(m))
        wrong_tile = (~ok).reshape(len(m), -1).mean(1) > 0.5          # 타일 절반 넘게 틀림(통째 혼동)
        print(f"{split} {name}: 화소 정확도 {acc:.4f} mIoU {mi:.4f} 경계 {ok[e].mean():.4f} 안쪽 {ok[~e].mean():.4f} "
              f"경계 비율 {e.mean():.4f} 절반 넘게 틀린 타일 {wrong_tile.sum()} (틀린 화소 중 {(~ok)[wrong_tile].sum() / (~ok).sum():.2f}) "
              f"IoU {[round(v, 3) for v in ious]}")
        if split == "test":
            PRED[name] = pred

# 그림용 타일: 경계가 있고, 두 모델 모두 타일을 통째로 틀리지 않았고, 스킵 없음의 경계 오차가 더 큰 타일
m = D["m_test"].numpy()
e = edges(m)
x = RAW["x_test"]
cand = []
for i in range(len(m)):
    if not e[i].any():
        continue
    es = (PRED["skip"][i] != m[i]).mean()
    en = (PRED["noskip"][i] != m[i]).mean()
    if es > 0.1 or en > 0.25:
        continue
    gap = (PRED["skip"][i] == m[i])[e[i]].mean() - (PRED["noskip"][i] == m[i])[e[i]].mean()
    pair = tuple(sorted(np.unique(m[i]).tolist()))
    cand.append((gap, i, pair))
cand.sort(reverse=True)
chosen, seen = [], set()
for gap, i, pair in cand:                    # 클래스 쌍이 겹치지 않게 4개
    if pair in seen or len(pair) != 2:
        continue
    chosen.append(i)
    seen.add(pair)
    if len(chosen) == 4:
        break
print("그림 타일", chosen, [tuple(RAW["classes"][k] for k in np.unique(m[i])) for i in chosen])

# 클래스 고정 팔레트: 산림 짙은 초록, 농경지 연두, 시가지 빨강, 수계 파랑, 나지 황토, 갯벌 회보라
PAL = np.array([[27, 110, 50], [176, 214, 92], [214, 60, 60], [40, 95, 210], [200, 160, 100], [150, 130, 175]], np.uint8)
fc = np.stack([x[i][[3, 2, 1]] for i in chosen])            # 폴스컬러 B8, B4, B3 → R, G, B
lo = np.percentile(fc, 1, axis=(0, 2, 3))
hi = np.percentile(fc, 99, axis=(0, 2, 3))
SC, G, N = 5, 8, 32 * 5
rows = len(chosen)
canvas = np.zeros((rows * N + (rows - 1) * G, 4 * N + 3 * G, 4), np.uint8)   # 틈은 투명
for r_, i in enumerate(chosen):
    v = np.clip((fc[r_] - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1) ** 0.8
    rgb = np.round(v * 255).astype(np.uint8).transpose(1, 2, 0)
    panels = [rgb, PAL[m[i]], PAL[PRED["skip"][i]], PAL[PRED["noskip"][i]]]
    for c_, p in enumerate(panels):
        big = p.repeat(SC, 0).repeat(SC, 1)
        r0, c0 = r_ * (N + G), c_ * (N + G)
        canvas[r0:r0 + N, c0:c0 + N, :3] = big
        canvas[r0:r0 + N, c0:c0 + N, 3] = 255
img = Image.fromarray(canvas, "RGBA").quantize(colors=256, method=Image.Quantize.FASTOCTREE)
img.save(os.path.join(OUT, "33-seg.png"), optimize=True)
for i in chosen:
    ok_s = (PRED["skip"][i] == m[i])[e[i]].mean()
    ok_n = (PRED["noskip"][i] == m[i])[e[i]].mean()
    print(f"  타일 {i}: 경계 정확도 스킵 {ok_s:.3f} / 없음 {ok_n:.3f}")
print("ok")
