"""34강 그림과 본문 수치: 태스크별 출력 예(PNG), 두 시기 변화 지도 예(PNG), 태스크·출력·지표 지도(SVG)

- 영상은 6부 공통 자료(make_part6)의 항만 타일 5번(차량 158·소형선박 6·선박 2)을 make_part6.render로 그림
- 탐지·분할·변화 패널은 모두 정답(OBB·HBB)으로 그린 '이상적인 출력'임(가상 탐지기 출력이 아님)
- 바다·땅 경계는 render와 같은 식으로 정함(그림용)
- 본문 수치(자료 요약은 part6_runs.json의 data, PQ 계산 예, 45° 승용차의 상자 넓이 배율)도 출력함
"""
import json, math, os, sys
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg
from make_part6 import tiles, render, obb_corners, TILE

OUT = os.path.join(HERE, "..", "fig")
T = tiles()
TID = 5
t = T[TID - 1]
objs, clutter = t["objs"], t["clutter"]

# 클래스 색(RGB): 바다, 땅, 선박, 소형선박, 차량
SEA, LAND = (24, 44, 96), (150, 150, 146)
CCOL = {0: (240, 120, 30), 1: (250, 220, 40), 2: (40, 210, 220)}


def land_x(objs, size=TILE):
    """render와 같은 바다·땅 경계(열 번호)"""
    lx = int(size * (max([o["cx"] + o["w"] for o in objs if o["cls"] == 2] + [size * 0.4]) / size))
    return min(max(lx, int(size * 0.35)), int(size * 0.55))


def fill(img, o, col):
    p = obb_corners(o["cx"], o["cy"], o["w"], o["h"], o["theta"])
    cv2.fillPoly(img, [np.round(p * 16).astype(np.int32)], col, lineType=cv2.LINE_8, shift=4)


def semantic(objs, size=TILE):
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = SEA
    img[:, :land_x(objs, size)] = LAND
    for o in objs:
        fill(img, o, CCOL[o["cls"]])
    return img


def instance(objs, size=TILE):
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = (16, 16, 18)
    rng = np.random.default_rng(7)
    for o in objs:
        h = rng.uniform(0, 180)
        hsv = np.uint8([[[h, rng.uniform(150, 255), rng.uniform(190, 255)]]])
        col = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])
        fill(img, o, col)
    return img


def boxes(img, objs):
    img = img.copy()
    for o in objs:
        x1, y1, x2, y2 = o["hbb"]
        cv2.rectangle(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), CCOL[o["cls"]], 1)
    return img


def crop(img, x0, y0, s):
    return img[y0:y0 + s, x0:x0 + s]


def mosaic(panels, ncol, path, gap=12, colors=256):
    h, w = panels[0].shape[:2]
    nrow = math.ceil(len(panels) / ncol)
    canvas = np.full((nrow * h + (nrow - 1) * gap, ncol * w + (ncol - 1) * gap, 3), 255, np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, ncol)
        canvas[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = p
    Image.fromarray(canvas).quantize(colors, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    return canvas.shape, os.path.getsize(path) // 1024


# ---------------------------------------------------------------- 1. 태스크별 출력 (2×2)
img = render(objs, clutter, "harbor", seed=TID)
X0, Y0, S = 0, 20, 400
panels = [crop(img, X0, Y0, S), crop(boxes(img, objs), X0, Y0, S),
          crop(semantic(objs), X0, Y0, S), crop(instance(objs), X0, Y0, S)]
print("tasks png", mosaic(panels, 2, os.path.join(OUT, "34-tasks.png")))

# ---------------------------------------------------------------- 2. 두 시기 변화 (3패널)
ships = [i for i, o in enumerate(objs) if o["cls"] != 2]
print("ships in tile", [(i, objs[i]["cls"], [round(v) for v in objs[i]["hbb"]]) for i in ships])
o2 = [dict(o) for o in objs]
gone = [i for i in ships if objs[i]["cls"] == 0 and objs[i]["hbb"][0] > 400][0]        # 오른쪽 선박이 떠남
moved = [i for i in ships if objs[i]["cls"] == 1 and abs(objs[i]["hbb"][0] - 344) < 2][0]  # 소형선박 하나가 옮김
o2[moved].update(cx=objs[moved]["cx"] + 30, cy=objs[moved]["cy"] + 70, theta=objs[moved]["theta"] - 40)
new = [dict(cls=1, cx=430.0, cy=95.0, w=26.0, h=9.0, theta=20.0), dict(cls=1, cx=330.0, cy=330.0, w=22.0, h=8.0, theta=-60.0)]
o2 = [o for i, o in enumerate(o2) if i != gone] + new
for o in o2:
    p = obb_corners(o["cx"], o["cy"], o["w"], o["h"], o["theta"])
    o["hbb"] = [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
img2 = render(o2, clutter, "harbor", seed=TID)
m1 = np.zeros((TILE, TILE), np.uint8)
m2 = np.zeros((TILE, TILE), np.uint8)
for o in objs:
    if o["cls"] != 2:
        fill(m1, o, 1)
for o in o2:
    if o["cls"] != 2:
        fill(m2, o, 1)
chg = np.zeros((TILE, TILE, 3), np.uint8)
chg[:] = (16, 16, 18)
chg[:, :land_x(objs)] = (52, 52, 54)
chg[(m1 == 1) & (m2 == 0)] = (240, 120, 30)     # 선박 → 바다(사라짐)
chg[(m1 == 0) & (m2 == 1)] = (70, 150, 255)     # 바다 → 선박(새로 생김)
same = (img == img2).all(-1)
print("render identical outside changes:", bool(same[(m1 == m2)].all()))
CX, CY, CS = 140, 40, 360
print("change png", mosaic([crop(img, CX, CY, CS), crop(img2, CX, CY, CS), crop(chg, CX, CY, CS)], 3,
                           os.path.join(OUT, "34-change.png")))
print("changed pixels in crop %", round(100 * float(((m1 != m2)[CY:CY + CS, CX:CX + CS]).mean()), 2))

# ---------------------------------------------------------------- 3. 태스크 지도 (SVG)


def arrow(s, x1, y1, x2, y2, cls="s-mu", fcls="f-mu", width=1.3, head=6, dash=None):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    bx, by = x2 - ux * head, y2 - uy * head
    s.line(x1, y1, bx, by, cls, width, dash)
    px, py = -uy * head * 0.55, ux * head * 0.55
    s.path(f"M {x2:.1f} {y2:.1f} L {bx + px:.1f} {by + py:.1f} L {bx - px:.1f} {by - py:.1f} Z", fcls, 0)


W, H = 780, 390
NW, NH = 142, 64
s = Svg(W, H, "비전 태스크 지도. 한 시기 영상에서 이미지 분류에 위치를 더하면 객체 탐지, 화소를 더하면 의미 분할, "
              "탐지에 객체별 마스크를 더하면 인스턴스 분할, 둘을 합치면 파놉틱 분할. 시간 축을 더하면 변화탐지와 다중 객체 추적")
COLX = [86, 288, 490, 692]
N = {  # 이름: (x, y, 태스크, 출력, 지표)
    "cls": (COLX[0], 125, "이미지 분류", "타일마다 라벨", "정확도·F1"),
    "sem": (COLX[1], 70, "의미 분할", "화소마다 클래스", "mIoU·Dice"),
    "det": (COLX[1], 180, "객체 탐지", "상자·클래스·점수", "mAP"),
    "ins": (COLX[2], 180, "인스턴스 분할", "객체마다 마스크", "마스크 AP"),
    "pan": (COLX[3], 125, "파놉틱 분할", "화소마다 클래스·번호", "PQ"),
    "chg": (COLX[0] + 40, 330, "변화탐지", "두 시기 → 변화 지도", "변화 클래스 F1·IoU"),
    "mot": (COLX[2], 330, "다중 객체 추적", "프레임을 넘는 같은 ID", "MOTA·IDF1·HOTA"),
}
s.text(14, 24, "한 시기 영상", "f-mu", 12, anchor="start", weight="600")
s.line(14, 244, W - 14, 244, "s-mu", 1, dash="4 4")
s.text(14, 268, "시간 축을 더함", "f-mu", 12, anchor="start", weight="600")
for k, (x, y, name, out, met) in N.items():
    s.rect(x - NW / 2, y - NH / 2, NW, NH, "s-fg f-acs" if k not in ("chg", "mot") else "s-bd f-bds", 1.3, rx=6)
    s.text(x, y - 11, name, "f-fg", 13, weight="600")
    s.text(x, y + 7, out, "f-fg", 11)
    s.text(x, y + 23, met, "f-mu", 11)


def side_pt(k, sx, sy):
    """노드 k의 가장자리 점: sx, sy는 -1·0·1 (왼·가운데·오른, 위·가운데·아래)"""
    x, y = N[k][:2]
    return x + sx * NW / 2, y + sy * NH / 2


def edge(a, b, pa, pb, label, lx, ly, anchor="middle", dash=None):
    x1, y1 = side_pt(a, *pa)
    x2, y2 = side_pt(b, *pb)
    arrow(s, x1 + 3 * pa[0], y1 + 3 * pa[1], x2 + 3 * pb[0], y2 + 3 * pb[1], dash=dash)
    s.text(lx, ly, label, "f-ac", 11, anchor=anchor)


# 노드 가장자리: (가로 위치, 세로 위치) — 오른쪽 가운데 = (1, 0), 왼쪽 가운데 = (-1, 0)
edge("cls", "sem", (1, -0.5), (-1, 0.3), "+화소", 186, 83)
edge("cls", "det", (1, 0.5), (-1, -0.3), "+위치", 186, 177)
edge("det", "ins", (1, 0), (-1, 0), "+마스크", 389, 172)
edge("sem", "pan", (1, 0.3), (-1, -0.5), "+객체 번호", 490, 82)
edge("ins", "pan", (1, -0.3), (-1, 0.5), "+배경(stuff)", 590, 184, anchor="start")
edge("det", "mot", (0.4, 1), (-1, 0), "+프레임 간 같은 ID", 344, 300, anchor="end", dash="4 3")
s.save(os.path.join(OUT, "34-map.svg"))

# ---------------------------------------------------------------- 본문 수치
R = json.load(open(os.path.join(DATA, "part6_runs.json"), encoding="utf-8"))
d = R["data"]
small = sum(v[0] for v in d["size_bins_small_medium_large"].values())
tot = sum(d["per_class"])
print("data", d["kinds"], d["per_class"], "total", tot, "small", small, round(100 * small / tot, 1), "%")
print("length", d["length_m_min_med_max"], "max per tile", max(d["objs_per_tile"]),
      "tiles with >100 vehicles", sum(v > 100 for v in d["vehicles_per_tile"]))
print("base mAP50", R["map"]["coco"]["ap50"], "mAP50-95", R["map"]["coco"]["ap"])
# PQ 예: 맞힌 쌍 IoU 0.9, 0.8, 0.7 / 오탐 1 / 미탐 1
ious, fp, fn = [0.9, 0.8, 0.7], 1, 1
tp = len(ious)
sq = sum(ious) / tp
rq = tp / (tp + fp / 2 + fn / 2)
print("PQ", round(sum(ious) / (tp + fp / 2 + fn / 2), 4), "SQ", sq, "RQ", rq, "SQ*RQ", round(sq * rq, 4))
# 45° 승용차(4.5 m × 1.85 m)의 HBB 넓이 / 실제 넓이
L, Wd = 4.5, 1.85
side = (L + Wd) * math.cos(math.radians(45))
print("45deg car hbb side", round(side, 2), "area", round(side ** 2, 2), "true", L * Wd, "ratio", round(side ** 2 / (L * Wd), 2))
# 의미 분할 정답 마스크(그림과 같은 채우기)의 차량을 8-연결 연결요소로 나누면 몇 덩어리인가
vm = np.zeros((TILE, TILE), np.uint8)
for o in objs:
    if o["cls"] == 2:
        fill(vm, o, 1)
print("vehicles", sum(o["cls"] == 2 for o in objs), "components 8-conn", cv2.connectedComponents(vm, connectivity=8)[0] - 1)
# 사후 분류 비교: 두 지도 정확도 0.9, 독립이면 둘 다 맞는 화소 비율
print("both correct", 0.9 * 0.9)
# 변화 화소 2%일 때 '모두 무변화' 정확도
print("all-unchanged acc", 1 - 0.02)
