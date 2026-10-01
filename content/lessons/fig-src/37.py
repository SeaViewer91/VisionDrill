"""37강 그림과 본문 수치: NMS 전후 상자(PNG), 슬라이스 겹침과 잘린 조각 도식(SVG)

- 본문 수치 대부분은 data/part6_runs.json의 nms·slice(계산: data/run_part6.py)에서 가져옴
- JSON에 없는 것(한 객체당 원시 상자 수, 중복 상자와 최고점 상자의 IoU 분포, 45° 주차 차량 두 대의 HBB IoU,
  중복·최고점 상자의 정답 IoU, 한 모델 WBF의 묶음 점수 평균 vs 최댓값,
  Soft-NMS 감쇠 배율, 축소 추론 때 차량 크기, 탐욕 NMS 코드 확인)은 여기서 계산해 출력함
"""
import math, os, sys
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg
from make_part6 import tiles, detect, render, big_scene
from part6_eval import iou_matrix, nms_dets, nms as nms_ref

OUT = os.path.join(HERE, "..", "fig")
T = tiles()
RAW = [detect(t["objs"], t["clutter"], 1.0, seed=t["image_id"]) for t in T]

# ---------------------------------------------------------------- 1. 본문 수치 확인
src = [d["src"] for r in RAW for d in r]
print("원시 상자", len(src), "객체에서", sum(isinstance(s, int) for s in src), "방해물", src.count("clutter"), "배경", src.count("bg"))
per, dup, top_g, dup_g = [], [], [], []
for t, r in zip(T, RAW):
    by = {}
    for d in r:
        if isinstance(d["src"], int):
            by.setdefault(d["src"], []).append(d)
    for v in by.values():
        per.append(len(v))
        v = sorted(v, key=lambda d: -d["score"])
        if len(v) > 1:
            dup += iou_matrix([v[0]["hbb"]], [x["hbb"] for x in v[1:]])[0].tolist()
            g = iou_matrix([t["objs"][v[0]["src"]]["hbb"]], [x["hbb"] for x in v])[0]   # 정답과의 IoU
            top_g.append(g[0]); dup_g += g[1:].tolist()
per, dup = np.array(per), np.array(dup)
print("잡힌 객체", len(per), "객체당 상자 평균", round(per.mean(), 2), "/ 정답", sum(len(t["objs"]) for t in T))
print("중복 상자", len(dup), "최고점 상자와 IoU > 0.7·0.5·0.3 비율", [round(float((dup > k).mean()), 3) for k in (0.7, 0.5, 0.3)])
from collections import Counter
for thr in (0.3, 0.5, 0.7):                      # NMS 뒤 객체에 붙어 남은 여분 상자(중복)와 상자가 하나라도 남은 객체 수
    extra = kept = 0
    for r in RAW:
        c = Counter(d["src"] for d in nms_dets(r, thr) if isinstance(d["src"], int))
        extra += sum(v - 1 for v in c.values()); kept += len(c)
    print(f"NMS {thr}: 남은 중복 {extra}, 상자가 남은 객체 {kept}")
L, W, gap = 4.5, 1.8, 2.6                       # 승용차 길이·폭, 주차 간격(m), 45° 줄
side = (L + W) / math.sqrt(2)
off = gap / math.sqrt(2)
inter = (side - off) ** 2
print("45° 이웃 차량 HBB 한 변", round(side, 2), "어긋남", round(off, 2), "IoU", round(inter / (2 * side ** 2 - inter), 3))
print("Soft-NMS 가우시안 σ 0.5 배율", {k: round(math.exp(-k ** 2 / 0.5), 3) for k in (0.2, 0.6, 0.9)})
print("2,048 → 640 축소 때 차량 길이(화소)", 8 * 640 / 2048, "~", 10 * 640 / 2048, "/ 512 타일 25장 화소 수 ÷ 640²", 25 * 512 ** 2 / 640 ** 2)
print("중복 상자 정답 IoU 평균(최고점 / 나머지)", round(float(np.mean(top_g)), 3), round(float(np.mean(dup_g)), 3))
from ensemble_boxes import weighted_boxes_fusion       # 한 모델 원시 출력에 WBF: 묶음 점수 평균(기본) vs 최댓값
from part6_eval import coco_eval


def wbf1(dets, thr=0.55, conf_type="avg"):
    Bx = np.clip(np.array([d["hbb"] for d in dets]) / 512, 0, 1).tolist()
    b, s_, l = weighted_boxes_fusion([Bx], [[d["score"] for d in dets]], [[d["cls"] for d in dets]],
                                     iou_thr=thr, skip_box_thr=0.0, conf_type=conf_type)
    return [dict(cls=int(c), score=float(x), hbb=(bb * 512).tolist()) for bb, x, c in zip(b, s_, l)]


for ct in ("avg", "max"):
    im = [(t["objs"], wbf1(r, conf_type=ct)) for t, r in zip(T, RAW)]
    print(f"한 모델 WBF 0.55 conf_type {ct}: 상자", sum(len(d) for _, d in im), "mAP50", coco_eval(im)["ap50"])
B = big_scene()
print("큰 장면 선박 길이(화소)", sorted(round(o["w"]) for o in B["objs"] if o["cls"] == 0))


def iou(b, bs):                                   # 상자 하나와 여러 개의 IoU
    x1 = np.maximum(b[0], bs[:, 0]); y1 = np.maximum(b[1], bs[:, 1])
    x2 = np.minimum(b[2], bs[:, 2]); y2 = np.minimum(b[3], bs[:, 3])
    it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
    return it / (a(b) + a(bs) - it)


def nms(boxes, scores, thr=0.5):                  # 본문 코드 블록과 같음
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        ious = iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= thr]
    return keep


r = [d for d in RAW[0] if d["cls"] == 2]
bx, sc = np.array([d["hbb"] for d in r]), np.array([d["score"] for d in r])
print("본문 코드 = part6_eval.nms:", sorted(int(i) for i in nms(bx, sc)) == sorted(nms_ref(bx, sc, 0.5)))

# ---------------------------------------------------------------- 2. NMS 전후(PNG)
import cv2
from PIL import Image

t = T[0]
img = render(t["objs"], t["clutter"], "harbor", seed=t["image_id"])
WINS = [(0, 14, 124, 150), (232, 14, 52, 150)]   # 자를 창(x0, y0, 너비, 높이): 왼쪽 위 45° 주차장, 계류 선박
Z = 3                                            # 확대 배율
COL = {0: (255, 200, 40), 1: (60, 220, 255), 2: (255, 70, 200)}


def crop(dets, x0, y0, w, h):
    p = cv2.resize(img[y0:y0 + h, x0:x0 + w], (w * Z, h * Z), interpolation=cv2.INTER_NEAREST)
    for d in sorted(dets, key=lambda d: d["score"]):
        x1, y1, x2, y2 = [(v - o) * Z for v, o in zip(d["hbb"], (x0, y0, x0, y0))]
        if x2 < 0 or y2 < 0 or x1 > w * Z or y1 > h * Z:
            continue
        cv2.rectangle(p, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), COL[d["cls"]], 1, cv2.LINE_AA)
    return p


def panel(dets):
    parts = []
    for k, wdw in enumerate(WINS):
        if k:
            parts.append(np.full((wdw[3] * Z, 6, 3), 255, np.uint8))
        parts.append(crop(dets, *wdw))
    return np.concatenate(parts, 1)


kept = Counter(d["src"] for d in nms_dets(RAW[0], 0.5))
inw = lambda h: WINS[0][0] <= (h[0] + h[2]) / 2 < WINS[0][0] + WINS[0][2] and WINS[0][1] <= (h[1] + h[3]) / 2 < WINS[0][1] + WINS[0][3]
gi = [i for i, o in enumerate(t["objs"]) if inw(o["hbb"])]
print("그림 주차장 창: 차량", len(gi), "원시 상자", sum(inw(d["hbb"]) for d in RAW[0]), "NMS 뒤", sum(inw(d["hbb"]) for d in nms_dets(RAW[0], 0.5)),
      "상자 2개 이상 남은 차", sum(kept[i] >= 2 for i in gi), "놓친 차", sum(kept[i] == 0 for i in gi))
a, b = panel(RAW[0]), panel(nms_dets(RAW[0], 0.5))
gapc = np.full((a.shape[0], 24, 3), 255, np.uint8)
im = Image.fromarray(np.concatenate([a, gapc, b], 1)).quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
im.save(os.path.join(OUT, "37-nms.png"), optimize=True)
print("PNG", im.size, os.path.getsize(os.path.join(OUT, "37-nms.png")) // 1024, "KB")



def orect(s, x, y, w, h, cls, width=1.5, dash=None):
    """채우지 않은 사각형(선만)"""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return s.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" class="{cls}" fill="none" stroke-width="{width}"{d}/>')

# ---------------------------------------------------------------- 3. 슬라이스 겹침과 잘린 조각(SVG)
s = Svg(780, 290, "가로로 이웃한 타일 A와 B가 겹침 구간을 공유함. 겹침 안의 차는 두 타일이 모두 잡아 전역 좌표로 옮긴 뒤 NMS로 하나를 지움. "
                  "겹침보다 긴 선박은 어느 타일에도 온전히 들어가지 않아 두 타일이 각각 잘린 조각 상자를 내고, 두 조각의 IoU가 0.33이라 NMS로 지워지지 않음. "
                  "전체를 줄여 한 번 추론하면 선박 전체를 덮는 상자 하나가 나옴")
TA, TB, TY0, TY1 = (40, 320), (240, 520), 52, 262
s.rect(TB[0] - 0, TY0, TA[1] - TB[0], TY1 - TY0, "s-mu f-sf", 0)                 # 겹침 구간
orect(s, TA[0], TY0, TA[1] - TA[0], TY1 - TY0, "s-ac", 1.8)
orect(s, TB[0], TY0 + 4, TB[1] - TB[0], TY1 - TY0 - 8, "s-bd", 1.8)
s.text(TA[0] + 4, TY0 - 10, "타일 A", "f-ac", 13, anchor="start", weight="600")
s.text(TB[1] - 4, TY0 - 10, "타일 B", "f-bd", 13, anchor="end", weight="600")
s.text((TB[0] + TA[1]) / 2, TY0 - 10, "겹침", "f-mu", 12)
# 겹침 안의 차: 두 타일이 모두 잡음
s.rect(270, 104, 20, 10, "s-fg f-mu", 1)
orect(s, 266, 100, 27, 18, "s-ac", 1.4)
orect(s, 268, 98, 26, 19, "s-bd", 1.4)
s.text(280, 140, "두 번 잡힘", "f-fg", 11)
s.text(280, 155, "→ NMS로 하나", "f-fg", 11)
# 겹침보다 긴 선박: 조각 둘 + 전체 축소 추론 상자
SX0, SX1, SY0, SY1 = 180, 420, 196, 216
s.rect(SX0, SY0, SX1 - SX0, SY1 - SY0, "s-fg f-mu", 1)
orect(s, SX0 - 6, SY0 - 14, SX1 - SX0 + 12, SY1 - SY0 + 28, "s-ok", 1.6)
orect(s, SX0 - 1, SY0 - 6, TA[1] - SX0 + 1, SY1 - SY0 + 12, "s-ac", 1.6)
orect(s, TB[0], SY0 - 9, SX1 - TB[0] + 1, SY1 - SY0 + 18, "s-bd", 1.6)
s.text((SX0 + TB[0]) / 2, SY1 + 30, "조각 A", "f-ac", 11)
s.text((TA[1] + SX1) / 2, SY1 + 30, "조각 B", "f-bd", 11)
# 범례와 IoU
LX = 560
for k, (cls, lab, w) in enumerate([("s-ac", "타일 A의 상자", 1.6), ("s-bd", "타일 B의 상자", 1.6), ("s-ok", "전체 축소 추론의 상자", 1.6)]):
    y = 80 + k * 26
    s.line(LX, y, LX + 26, y, cls, w)
    s.text(LX + 34, y + 4, lab, "f-fg", 12, anchor="start")
fa, fb = (SX0, TA[1]), (TB[0], SX1)
iou_frag = (fa[1] - fb[0]) / (fb[1] - fa[0])
s.text(LX, 186, f"조각 A·B의 IoU {iou_frag:.2f}", "f-fg", 12, anchor="start")
s.text(LX, 204, "→ NMS 0.5로 안 지워짐", "f-fg", 12, anchor="start")
s.save(os.path.join(OUT, "37-slice.svg"))
print("조각 IoU(도식)", round(iou_frag, 3))
