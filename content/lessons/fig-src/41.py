"""41강 그림과 본문 수치: 항만 타일의 HBB·OBB(PNG), 5·8파라미터와 경계 불연속 도식(SVG), 각도 어긋남–회전 IoU 곡선(SVG)

본문 수치 대부분은 part6_runs.json의 'obb'(run_part6.py exp_obb)에서 가져옴. JSON에 없는 것
(같은 탐지를 회전 IoU로만 다시 평가한 값, 회전 NMS 뒤 상자 수, 선박 라벨 각도 분포, IoU 0.5·0.75 아래로 떨어지는 각도,
sin·cos 표현의 거리, 꼭짓점 순서, OpenCV minAreaRect 각도)은 여기서 계산해 출력함. 학습은 없음(몇 초)
"""
import math, os, sys
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg
from make_part6 import tiles, detect, render, obb_corners, obb_to_hbb, TILE
from part6_eval import riou, iou_matrix, nms_dets, coco_eval

OUT = os.path.join(HERE, "..", "fig")

# ---------------------------------------------------------------- 1. 본문 수치 확인
T = tiles()
RAW = [detect(t["objs"], t["clutter"], 1.0, seed=t["image_id"]) for t in T]
HB = [(t["objs"], nms_dets(r, 0.5)) for t, r in zip(T, RAW)]                   # 기본 탐지 결과(HBB NMS 0.5)
RO = [(t["objs"], nms_dets(r, 0.5, rotated=True)) for t, r in zip(T, RAW)]     # 회전 NMS 0.5
for name, imgs, rot in [("HBB NMS + HBB IoU 평가", HB, False), ("HBB NMS + 회전 IoU 평가(같은 탐지)", HB, True),
                        ("회전 NMS + 회전 IoU 평가", RO, True)]:
    e = coco_eval(imgs, rotated=rot)
    print(f"{name}: 상자 {sum(len(d) for _, d in imgs)}개 mAP50 {e['ap50']} mAP50-95 {e['ap']} "
          f"AP50 {[e['per_class'][c][0] for c in range(3)]} AP75 {[e['per_class'][c][5] for c in range(3)]}")
for thr in (0.3,):
    imgs = [(t["objs"], nms_dets(r, thr, rotated=True)) for t, r in zip(T, RAW)]
    e = coco_eval(imgs, rotated=True)
    print(f"회전 NMS {thr} + 회전 IoU: 상자 {sum(len(d) for _, d in imgs)}개 mAP50 {e['ap50']} mAP50-95 {e['ap']}")

th = np.array([o["theta"] for t in T for o in t["objs"] if o["cls"] == 0])
print("선박 라벨", len(th), "척 중 |θ|>80°:", int((np.abs(th) > 80).sum()), "(음수", int((th < -80).sum()), "양수", int((th > 80).sum()), ")")
fill = [o["w"] * o["h"] / o["area"] for t in T for o in t["objs"]]
print("채움 비율 전체 중앙값", round(float(np.median(fill)), 3))


def hbb_size(w, h, a):
    t = math.radians(a)
    return w * abs(math.cos(t)) + h * abs(math.sin(t)), w * abs(math.sin(t)) + h * abs(math.cos(t))


W_, H_ = hbb_size(200, 34, 45)
print("200×34, 45°: HBB", round(W_, 1), "×", round(H_, 1), "채움", round(200 * 34 / (W_ * H_), 4), "= 2wh/(w+h)^2", round(2 * 200 * 34 / 234 ** 2, 4))
a, b = (100, 100, 200, 34, 45), (100 - 38 / math.sqrt(2), 100 + 38 / math.sqrt(2), 200, 34, 45)
print("나란한 45° 선박: HBB IoU", round(float(iou_matrix([obb_to_hbb(a)], [obb_to_hbb(b)])[0, 0]), 4), "회전 IoU", round(riou(a, b), 4))


def cross(w, h, level):
    """각도 어긋남이 몇 도일 때 회전 IoU가 level 아래로 떨어지는가(0.01° 간격)"""
    for d in np.arange(0, 90, 0.01):
        if riou((0, 0, w, h, 0), (0, 0, w, h, d)) < level:
            return round(float(d), 2)


for nm, w, h in [("선박 200×34", 200, 34), ("소형선박 24×8", 24, 8), ("20×16", 20, 16)]:
    print(nm, "회전 IoU 0.75 아래:", cross(w, h, 0.75), "° / 0.5 아래:", cross(w, h, 0.5), "°",
          "| 5°", round(riou((0, 0, w, h, 0), (0, 0, w, h, 5)), 4), "10°", round(riou((0, 0, w, h, 0), (0, 0, w, h, 10)), 4))
print("경계 예 −88° vs 89°: 회전 IoU", round(riou((0, 0, 200, 34, -88), (0, 0, 200, 34, 89)), 4))
enc = lambda d: np.array([math.cos(math.radians(2 * d)), math.sin(math.radians(2 * d))])
print("(cos2θ, sin2θ): −88°", enc(-88).round(4), "89°", enc(89).round(4), "거리", round(float(np.linalg.norm(enc(-88) - enc(89))), 4),
      "/ 10° vs 13° 거리", round(float(np.linalg.norm(enc(10) - enc(13))), 4))
print("정사각형에 가까운 물체 20×19.6 @10° vs 19.6×20 @−80°: 회전 IoU", round(riou((0, 0, 20, 19.6, 10), (0, 0, 19.6, 20, -80)), 4))
P = obb_corners(100, 60, 40, 12, 30)
x, y = P[:, 0], P[:, 1]
print("8파라미터 (100,60,40,12,30):", P.round(1).ravel().tolist(),
      "신발끈 넓이(영상 좌표, +면 화면에서 시계 방향)", round(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)), 1))
try:
    import cv2
    r = [cv2.minAreaRect(obb_corners(0, 0, 200, 34, d).astype(np.float32)) for d in (0, 30, -30, 89)]
    print("OpenCV", cv2.__version__, "minAreaRect (θ=0, 30, −30, 89):", [(tuple(round(v) for v in s), round(ang, 1)) for _, s, ang in r])
except ImportError:
    pass

# ---------------------------------------------------------------- 2. 항만 타일의 HBB·OBB (PNG, 글자 없음)
import cv2
from PIL import Image

SC, CROP, GAP = 3, 190, 18
HBB_COL, OBB_COL = (255, 170, 40), (70, 225, 255)        # 주황 = HBB, 하늘 = OBB


def panel(t, x0, y0):
    img = render(t["objs"], t["clutter"], t["kind"], seed=t["image_id"])
    crop = img[y0:y0 + CROP, x0:x0 + CROP]
    big = cv2.resize(crop, (CROP * SC, CROP * SC), interpolation=cv2.INTER_NEAREST)
    for o in t["objs"]:
        hb = np.array(o["hbb"]) - [x0, y0, x0, y0]
        if hb[2] < 0 or hb[3] < 0 or hb[0] > CROP or hb[1] > CROP:
            continue
        p = (obb_corners(o["cx"] - x0, o["cy"] - y0, o["w"], o["h"], o["theta"]) * SC * 16).round().astype(np.int32)
        q = np.array([[hb[0], hb[1]], [hb[2], hb[1]], [hb[2], hb[3]], [hb[0], hb[3]]]) * SC * 16
        cv2.polylines(big, [q.round().astype(np.int32)], True, HBB_COL, 2, cv2.LINE_AA, shift=4)
        cv2.polylines(big, [p], True, OBB_COL, 2, cv2.LINE_AA, shift=4)
    return big


A = panel(T[4], 0, 8)                       # 항만(5번 타일): 45° 주차장
B = panel(T[12], 300, 10)                   # 외해(13번 타일): 비스듬한 선박
N = CROP * SC
canvas = np.zeros((N, 2 * N + GAP, 4), np.uint8)
canvas[:, :N, :3], canvas[:, N + GAP:, :3] = A, B
canvas[:, :N, 3] = canvas[:, N + GAP:, 3] = 255
Image.fromarray(canvas, "RGBA").quantize(colors=128, method=Image.Quantize.FASTOCTREE).save(os.path.join(OUT, "41-hbb-obb.png"), optimize=True)
print("PNG", canvas.shape, os.path.getsize(os.path.join(OUT, "41-hbb-obb.png")) // 1024, "KB")


# ---------------------------------------------------------------- 3. 5·8파라미터와 경계 불연속 도식 (SVG)
def poly_d(pts):
    return "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts) + " Z"


def arrow(s, x1, y1, x2, y2, cls="s-fg", fcls="f-fg", width=1.4, head=6, dash=None):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    bx, by = x2 - ux * head, y2 - uy * head
    s.line(x1, y1, bx, by, cls, width, dash)
    px, py = -uy * head * 0.55, ux * head * 0.55
    s.path(f"M {x2:.1f} {y2:.1f} L {bx + px:.1f} {by + py:.1f} L {bx - px:.1f} {by - py:.1f} Z", fcls, 0)


s = Svg(780, 300, "왼쪽: 회전 박스의 5파라미터(중심 cx·cy, 긴 변 w, 짧은 변 h, 긴 변이 x축과 이루는 각 θ). 영상 좌표는 y가 아래라 "
                  "θ = +30°는 화면에서 시계 방향. 가운데: 같은 상자의 8파라미터(꼭짓점 1→4를 시계 방향으로). "
                  "오른쪽: 거의 세로인 두 상자 −88°와 89°는 실제로 3° 차이지만 값은 177 차이")
TITLE_Y = 26
for cx_, lab in [(130, "5파라미터"), (390, "8파라미터"), (650, "경계 불연속")]:
    s.text(cx_, TITLE_Y, lab, "f-fg", 13, weight="600")
s.line(260, 40, 260, 285, "s-mu", 0.8, "3 4")
s.line(520, 40, 520, 285, "s-mu", 0.8, "3 4")

# 왼쪽: 축과 5파라미터
OX, OY = 22, 52
arrow(s, OX, OY, OX + 46, OY, "s-mu", "f-mu", 1.2)
arrow(s, OX, OY, OX, OY + 46, "s-mu", "f-mu", 1.2)
s.text(OX + 52, OY + 4, "x", "f-mu", 11, anchor="start")
s.text(OX, OY + 60, "y", "f-mu", 11)
C1 = (112, 168)
WB, HB_ = 150, 44
TH = 30
P1 = obb_corners(C1[0], C1[1], WB, HB_, TH)
s.path(poly_d(P1), "s-ac f-acs", 1.6)
s.line(C1[0], C1[1], C1[0] + 112, C1[1], "s-mu", 1.0, "4 3")         # x축과 나란한 기준선
r_ = 96
ex, ey = C1[0] + r_ * math.cos(math.radians(TH)), C1[1] + r_ * math.sin(math.radians(TH))
s.line(C1[0] + 75 * math.cos(math.radians(TH)), C1[1] + 75 * math.sin(math.radians(TH)),
       C1[0] + 108 * math.cos(math.radians(TH)), C1[1] + 108 * math.sin(math.radians(TH)), "s-mu", 1.0, "4 3")  # 긴 변 방향 연장
s.path(f"M {C1[0] + r_:.1f} {C1[1]:.1f} A {r_} {r_} 0 0 1 {ex:.1f} {ey:.1f}", "s-bd", 1.5)
s.text(C1[0] + 106 * math.cos(math.radians(14)) + 2, C1[1] + 106 * math.sin(math.radians(14)) + 5, "θ", "f-bd", 13, anchor="start", weight="600")
s.circle(C1[0], C1[1], 3, "f-fg")
s.text(C1[0] - 6, C1[1] - 8, "(cx, cy)", "f-fg", 11, anchor="end")
# w: 위쪽 긴 변(1→2)의 바깥, h: 오른쪽 짧은 변(2→3)의 바깥
t_ = math.radians(TH)
nrm = np.array([math.sin(t_), -math.cos(t_)])                          # 위쪽 긴 변 바깥 방향
mw = (P1[0] + P1[1]) / 2 + nrm * 14
s.text(mw[0], mw[1], "w (긴 변)", "f-ac", 11)
mh = (P1[3] + P1[0]) / 2 - np.array([math.cos(t_), math.sin(t_)]) * 12      # 왼쪽 짧은 변 바깥
s.text(mh[0] - 2, mh[1] + 4, "h", "f-ac", 12, anchor="end")
s.text(130, 282, "θ = +30° (화면에서 시계 방향)", "f-mu", 11)

# 가운데: 같은 상자, 꼭짓점 번호
C2 = (390, 168)
P2 = obb_corners(C2[0], C2[1], WB, HB_, TH)
s.path(poly_d(P2), "s-ac f-acs", 1.6)
off = [(-12, -8), (10, -8), (10, 16), (-12, 14)]
for k, (px, py) in enumerate(P2):
    s.circle(px, py, 4, "f-bd")
    s.text(px + off[k][0], py + off[k][1], str(k + 1), "f-bd", 12, weight="600")
for k in range(3):                                                    # 순서 화살표(변 안쪽)
    a_, b_ = P2[k], P2[k + 1]
    m_ = (a_ + b_) / 2
    d_ = (b_ - a_) / np.linalg.norm(b_ - a_)
    inn = np.array(C2) - m_
    inn = inn / np.linalg.norm(inn) * 9
    arrow(s, *(m_ + inn - d_ * 10), *(m_ + inn + d_ * 10), "s-bd", "f-bd", 1.2, 5)
s.text(390, 282, "(x1, y1, …, x4, y4)", "f-mu", 11)

# 오른쪽: −88°와 89°
C3 = (650, 158)
s.path(poly_d(obb_corners(C3[0], C3[1], 190, 30, -88)), "s-ac", 1.6)
s.add(f'<path d="{poly_d(obb_corners(C3[0], C3[1], 190, 30, 89))}" class="s-bd" stroke-width="1.4" fill="none" stroke-dasharray="5 3"/>')
# 위쪽 끝: −88° 상자는 오른쪽으로, 89° 상자는 왼쪽으로 기울어 있으므로 이름도 그 쪽에 붙임
s.text(C3[0] + 24, 70, "−88°", "f-ac", 12, anchor="start", weight="600")
s.text(C3[0] - 24, 70, "89°", "f-bd", 12, anchor="end", weight="600")
s.text(650, 268, "값 차이 177", "f-mu", 11)
s.text(650, 284, "실제 차이 3°", "f-mu", 11)
s.save(os.path.join(OUT, "41-param.svg"))

# ---------------------------------------------------------------- 4. 각도 어긋남–회전 IoU 곡선 (SVG)
s = Svg(640, 300, "같은 중심·같은 크기의 상자를 0~30° 돌렸을 때의 회전 IoU. 길쭉한 선박(200×34)이 가장 빨리 떨어지고, "
                  "소형선박(24×8), 정사각형에 가까운 물체(20×16) 순으로 덜 떨어짐. 점선은 IoU 0.5와 0.75")
X0, X1, Y0, Y1 = 60, 500, 260, 30
fx = lambda d: X0 + (X1 - X0) * d / 30
fy = lambda v: Y0 - (Y0 - Y1) * v
s.line(X0, Y0, X1, Y0, "s-fg", 1.2)
s.line(X0, Y0, X0, Y1, "s-fg", 1.2)
for d in range(0, 31, 5):
    s.line(fx(d), Y0, fx(d), Y0 + 4, "s-fg", 1)
    s.text(fx(d), Y0 + 17, f"{d}", "f-mu", 11)
for v in (0, 0.25, 0.5, 0.75, 1.0):
    s.line(X0 - 4, fy(v), X0, fy(v), "s-fg", 1)
    s.text(X0 - 8, fy(v) + 4, f"{v:g}", "f-mu", 11, anchor="end")
for v in (0.5, 0.75):
    s.line(X0, fy(v), X1, fy(v), "s-mu", 0.9, "4 4")
s.text((X0 + X1) / 2, 293, "각도 어긋남(°)", "f-mu", 11)
s.text(X0 - 2, Y1 - 12, "회전 IoU", "f-mu", 11, anchor="start")
D = np.arange(0, 30.01, 0.25)
for (w, h, cls, fcls, lab) in [(200, 34, "s-bd", "f-bd", "선박 200×34"), (24, 8, "s-ac", "f-ac", "소형선박 24×8"),
                               (20, 16, "s-ok", "f-ok", "20×16")]:
    v = [riou((0, 0, w, h, 0), (0, 0, w, h, d)) for d in D]
    s.path("M " + " L ".join(f"{fx(d):.1f} {fy(u):.1f}" for d, u in zip(D, v)), cls, 2)
    s.text(X1 + 8, fy(v[-1]) + 4, lab, fcls, 11, anchor="start")
s.save(os.path.join(OUT, "41-riou-angle.svg"))
print("ok")
