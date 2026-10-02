"""44강 그림과 본문 수치: 같은 상자의 네 형식 좌표 도식(SVG), 라벨 잡음별 시험 정확도·갯벌 재현율 막대(SVG)

본문 수치 대부분은 data/part7_runs.json의 formats·noise·missing(계산: data/run_part7.py)에서 가져옴.
JSON에 없는 것(YOLO 변환 손 계산, VOC 1부터 세는 정수 좌표, 지오트랜스폼 역변환, 고리 방향,
대칭 잡음의 학습 손실 바닥, 누락 라벨에서 점수 0.5 이상 탐지의 정밀도·재현율, 표본 검수 신뢰구간)은 여기서 계산해 출력함
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg

OUT = os.path.join(HERE, "..", "fig")
R = json.load(open(os.path.join(DATA, "part7_runs.json"), encoding="utf-8"))
F, N, M = R["formats"], R["noise"], R["missing"]

# ---------------------------------------------------------------- 1. 형식 변환 손 계산
x1, y1, x2, y2 = F["hbb_x1y1x2y2"]
W = H = 512
print("YOLO (반올림 좌표로 계산):", round((x1 + x2) / 2, 3), round((x1 + x2) / 2 / W, 5), round((y1 + y2) / 2 / H, 5),
      round((x2 - x1) / W, 5), round((y2 - y1) / H, 5), "/ JSON", F["yolo"])
print("COCO 폭 23.62 ÷ 512 =", round(23.62 / 512, 5), "| VOC xmax−xmin =", round(x2 - x1, 2), "| 반올림 전 폭 = JSON yolo w × 512 =", round(F["yolo"][3] * 512, 3))
voc1 = [math.floor(v) + 1 for v in (x1, y1, x2, y2)]                       # 1부터 세는 정수 화소 번호
print("VOC 원래 규약(1부터 정수):", voc1, "폭 xmax−xmin+1 =", voc1[2] - voc1[0] + 1, "→ 연속 좌표 x1 = xmin−1 =", voc1[0] - 1, "x2 = xmax =", voc1[2])
print("35강 IoU: 9화소 1칸 어긋남 (9−1)/(9+1) =", 8 / 10, "/ +1 규약이면 폭 10:", round(9 / 11, 3))
gt = F["geotransform"]
c = np.array(F["yolo_obb"][1:]).reshape(4, 2) * [W, H]                    # 꼭짓점(화소)
geo = np.array(F["geojson_polygon"][:4])
print("꼭짓점 화소:", c.round(2).tolist())
print("지오트랜스폼 순방향 첫 점:", round(gt[0] + gt[1] * c[0, 0], 2), round(gt[3] + gt[5] * c[0, 1], 2), "/ JSON", F["geojson_polygon"][0])
print("역방향 첫 점: x =", round((geo[0, 0] - gt[0]) / gt[1], 2), "y =", round((geo[0, 1] - gt[3]) / gt[5], 2))


def shoelace(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


print("신발끈 넓이: 화소(y 아래, +면 화면 시계 방향)", round(shoelace(c), 1), "/ 지도(y 위, −면 시계 방향)", round(shoelace(geo), 1), "m²")
print("타일 오른쪽 아래 지도 좌표:", gt[0] + gt[1] * W, gt[3] + gt[5] * H, "| 길이·폭(m)", F["obj_obb"][2] * 0.5, F["obj_obb"][3] * 0.5)

# ---------------------------------------------------------------- 2. 라벨 잡음
KEYS = [("clean", "깨끗함"), ("sym10", "대칭 10%"), ("sym20", "대칭 20%"), ("sym40", "대칭 40%"), ("tidal_to_water40", "갯벌→수계 40%")]
for k, nm in KEYS:
    r = N[k]
    print(f"{nm}: 바꾼 라벨 {r['n_flipped']} ({r['n_flipped'] / 4200:.3f}) 시험 best {r['test_best']} "
          f"재현율 {r['recall_best']} (150타일 중 {[round(v * 150) for v in r['recall_best']]}) 학습 손실 {r['train_loss_last']}")
for k in ("sym10", "sym20", "sym40"):
    e = N[k]["n_flipped"] / 4200
    print(f"{k} 손실 바닥 −(1−ε)ln(1−ε) − ε ln(ε/5), ε = {e:.3f}:", round(-(1 - e) * math.log(1 - e) - e * math.log(e / 5), 3),
          "| 40%에서 참 클래스 몫 0.6, 틀린 클래스 각", round(0.4 / 5, 2) if k == "sym40" else "")
print("갯벌 타일 중 바뀐 비율", round(N["tidal_to_water40"]["n_flipped"] / 700, 3))

# ---------------------------------------------------------------- 3. 누락 라벨: 같은 탐지, 정답 일부 삭제
from make_part6 import tiles, detect
from part6_eval import nms_dets, operating_point

T = tiles()
base = [(t["objs"], nms_dets(detect(t["objs"], t["clutter"], 1.0, seed=t["image_id"]), 0.5)) for t in T]
print("기본 탐지 결과 상자 수", sum(len(d) for _, d in base))
for frac in (0.0, 0.1, 0.2, 0.3):
    rng = np.random.default_rng(5)                                      # run_part7.exp_missing과 같은 삭제
    imgs = [([o for o in g if rng.random() >= frac], d) for g, d in base]
    ops = [operating_point(imgs, cl, 0.5) for cl in range(3)]
    tp, fp, fn = (sum(o[k] for o in ops) for k in ("tp", "fp", "fn"))
    j = M[f"drop{int(frac * 100)}"]
    print(f"지운 비율 {frac}: 정답 {j['n_gt']} mAP50 {j['ap50']} | 점수 0.5 이상 탐지 {tp + fp}: 정탐 {tp} 오탐 {fp} "
          f"정밀도 {tp / (tp + fp):.3f} 재현율 {tp / (tp + fn):.3f}")

from statsmodels.stats.proportion import proportion_confint
print("표본 검수 200장 중 6장 오류: 95% 윌슨 구간", [round(v, 4) for v in proportion_confint(6, 200, method="wilson")])


# ---------------------------------------------------------------- 4. 그림: 같은 상자의 네 형식 좌표(SVG)
def orect(s, x, y, w, h, cls, width=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return s.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" class="{cls}" fill="none" stroke-width="{width}"{d}/>')


def arrow(s, xa, ya, xb, yb, cls="s-fg", fcls="f-fg", width=1.4, head=6):
    L = math.hypot(xb - xa, yb - ya)
    ux, uy = (xb - xa) / L, (yb - ya) / L
    bx, by = xb - ux * head, yb - uy * head
    s.line(xa, ya, bx, by, cls, width)
    s.add(f'<path d="M {xb:.1f} {yb:.1f} L {bx - uy * head * 0.5:.1f} {by + ux * head * 0.5:.1f} '
          f'L {bx + uy * head * 0.5:.1f} {by - ux * head * 0.5:.1f} Z" class="{fcls}"/>')


s = Svg(800, 360, "같은 상자를 네 형식으로 본 좌표. 왼쪽은 512×512 타일 전체로, 화소 원점은 왼쪽 위이고 y가 아래로 커짐. "
                  "정규화 좌표는 오른쪽 아래가 (1, 1). 지도 좌표는 왼쪽 위가 지오트랜스폼 원점이고 북 좌표가 위로 커짐. "
                  "오른쪽은 상자를 확대한 것으로 COCO·VOC는 왼쪽 위 모서리에서, YOLO는 중심에서 출발함. 점선은 회전 박스")
# 왼쪽: 타일 전체
TX, TY, TS = 110, 70, 230                       # 타일 왼쪽 위와 한 변 길이(그림 단위)
k = TS / 512
s.rect(TX, TY, TS, TS, "s-mu f-sf", 1.2)
arrow(s, TX, TY, TX + 70, TY, "s-ac", "f-ac")
arrow(s, TX, TY, TX, TY + 70, "s-ac", "f-ac")
s.text(TX + 76, TY + 4, "x", "f-ac", 13, anchor="start", weight="600")
s.text(TX + 7, TY + 86, "y", "f-ac", 13, anchor="start", weight="600")
s.text(TX, TY - 30, "화소 (0, 0)", "f-ac", 12, anchor="start")
s.text(TX, TY - 14, "지도 (420000, 3880000)", "f-bd", 12, anchor="start")
s.text(TX + TS, TY + TS + 20, "화소 (512, 512) · 정규화 (1, 1)", "f-ac", 12, anchor="end")
s.text(TX + TS, TY + TS + 36, "지도 (420256, 3879744)", "f-bd", 12, anchor="end")
# 지도 축: 왼쪽 바깥에 동·북 화살표
MX, MY = 40, TY + TS - 10
arrow(s, MX, MY, MX, MY - 60, "s-bd", "f-bd")
arrow(s, MX, MY, MX + 50, MY, "s-bd", "f-bd")
s.text(MX, MY - 66, "북", "f-bd", 12)
s.text(MX + 58, MY + 4, "동", "f-bd", 12, anchor="start")
# 타일 안의 상자
bx0, by0, bw, bh = TX + x1 * k, TY + y1 * k, (x2 - x1) * k, (y2 - y1) * k
orect(s, bx0, by0, bw, bh, "s-fg", 1.4)
# 오른쪽: 확대
Z = 4.6
RX, RY = 520, 76
P = lambda x, y: (RX + (x - x1) * Z, RY + (y - y1) * Z)
s.line(bx0 + bw, by0, *P(x1, y1), "s-mu", 1, "3 3")
s.line(bx0 + bw, by0 + bh, *P(x1, y2), "s-mu", 1, "3 3")
pts = [P(*p) for p in c]
s.path("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts) + " Z", "s-mu", 1.3, "none")
s.add(s.parts.pop().replace("/>", ' stroke-dasharray="4 3"/>'))
X0, Y0 = P(x1, y1)
X1, Y1 = P(x2, y2)
orect(s, X0, Y0, X1 - X0, Y1 - Y0, "s-fg", 1.6)
CX, CY = P((x1 + x2) / 2, (y1 + y2) / 2)
s.circle(X0, Y0, 4, "f-ac")
s.circle(X1, Y1, 4, "f-ac")
s.circle(CX, CY, 4, "f-bd")
s.text(X0 - 8, Y0 - 22, "(138.94, 217.11)", "f-fg", 12, anchor="end")
s.text(X0 - 8, Y0 - 7, "COCO x·y, VOC 시작", "f-mu", 11, anchor="end")
s.text(X1 + 8, Y1 + 4, "(162.57, 258.09)", "f-fg", 12, anchor="start")
s.text(X1 + 8, Y1 + 19, "VOC 끝", "f-mu", 11, anchor="start")
s.line(CX + 4, CY, X1 + 22, CY, "s-mu", 1)
s.text(X1 + 26, CY - 3, "중심 (150.75, 237.60)", "f-bd", 12, anchor="start")
s.text(X1 + 26, CY + 13, "YOLO: ÷ 512", "f-mu", 11, anchor="start")
# 폭·높이 치수선
arrow(s, (X0 + X1) / 2, Y1 + 16, X1, Y1 + 16, "s-mu", "f-mu", 1, 5)
arrow(s, (X0 + X1) / 2, Y1 + 16, X0, Y1 + 16, "s-mu", "f-mu", 1, 5)
s.text((X0 + X1) / 2, Y1 + 34, "폭 23.62", "f-fg", 12)
arrow(s, X0 - 16, (Y0 + Y1) / 2, X0 - 16, Y0, "s-mu", "f-mu", 1, 5)
arrow(s, X0 - 16, (Y0 + Y1) / 2, X0 - 16, Y1, "s-mu", "f-mu", 1, 5)
s.text(X0 - 22, (Y0 + Y1) / 2 + 4, "높이 40.97", "f-fg", 12, anchor="end")
s.save(os.path.join(OUT, "44-formats.svg"))
print("도식 확대 상자", round(X0), round(Y0), round(X1), round(Y1))

# ---------------------------------------------------------------- 5. 그림: 라벨 잡음별 시험 정확도·갯벌 재현율(SVG)
s = Svg(720, 272, "학습 라벨 잡음별 시험 정확도(파랑)와 갯벌 재현율(주황). 대칭 잡음은 40%까지 넣어도 정확도가 0.974에서 0.940으로 내려가는 데 그침. "
                  "갯벌을 수계로 바꾼 체계적 잡음은 라벨 266개만 바꿨는데 갯벌 재현율이 0.733으로 무너짐")
PX0, PX1, PY0, PY1 = 60, 700, 40, 220         # 그림 영역
yv = lambda v: PY1 - v * (PY1 - PY0)
for t in (0, 0.25, 0.5, 0.75, 1.0):
    s.line(PX0, yv(t), PX1, yv(t), "s-mu", 0.6, "2 3" if t else None)
    s.text(PX0 - 8, yv(t) + 4, f"{t:g}", "f-mu", 11, anchor="end")
GW = (PX1 - PX0) / len(KEYS)
BW = 34
for i, (key, nm) in enumerate(KEYS):
    r = N[key]
    gx = PX0 + GW * (i + 0.5)
    for j, (v, cls) in enumerate([(r["test_best"], "f-ac"), (r["recall_best"][5], "f-bd")]):
        x = gx - BW - 2 + j * (BW + 4)
        s.rect(x, yv(v), BW, PY1 - yv(v), cls, 0)
        s.text(x + BW / 2, yv(v) - 5, f"{v:.3f}", "f-fg", 11)
    s.text(gx, PY1 + 20, nm, "f-fg", 12)
    s.text(gx, PY1 + 36, f"바꾼 라벨 {r['n_flipped']:,}", "f-mu", 11)
for j, (cls, lab) in enumerate([("f-ac", "시험 정확도"), ("f-bd", "갯벌 재현율")]):
    lx = PX0 + 10 + j * 130
    s.rect(lx, 12, 14, 12, cls, 0)
    s.text(lx + 20, 22, lab, "f-fg", 12, anchor="start")
s.save(os.path.join(OUT, "44-noise.svg"))
