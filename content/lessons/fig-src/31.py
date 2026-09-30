"""31강 그림: 백본 계보 연표, 잔차·병목·밀집 블록 도식, 깊은 망 학습 손실 곡선(SVG)

곡선 값은 data/part5_runs.json의 deep 실험(plain8·plain20·res20, Adam 1e-3, 20에폭)에서 읽음.
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
R = json.load(open(os.path.join(HERE, "..", "data", "part5_runs.json"), encoding="utf-8"))


def arrow(s, x1, y1, x2, y2, cls="s-fg", fcls="f-fg", width=1.4, head=6):
    """직선 화살표(머리는 채운 삼각형 경로, marker를 쓰지 않음: id 금지)"""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    bx, by = x2 - ux * head, y2 - uy * head
    s.line(x1, y1, bx, by, cls, width)
    px, py = -uy * head * 0.55, ux * head * 0.55
    s.path(f"M {x2:.1f} {y2:.1f} L {bx + px:.1f} {by + py:.1f} L {bx - px:.1f} {by - py:.1f} Z", fcls, 0)


# ---------------------------------------------------------------- 1. 계보 연표
s = Svg(780, 258, "2012년 AlexNet부터 2022년 ConvNeXt까지 대표 백본이 처음 공개된 해와 핵심 아이디어")
X = lambda yr: 60 + (yr - 2012) * 62
AX = 130
s.line(X(2012) - 20, AX, X(2022) + 20, AX, "s-mu", 2)
for yr in range(2012, 2023):
    s.line(X(yr), AX - 4, X(yr), AX + 4, "s-mu", 1.2)
# (이름, 해, 한 줄, 줄 위치 A~D, 흐리게)
ITEMS = [("AlexNet", 2012, "GPU로 학습한 8층 CNN", "B", False),
         ("VGG", 2014, "3×3 합성곱을 깊게", "C", False),
         ("ResNet", 2015, "잔차 연결 x + F(x)", "A", False),
         ("DenseNet", 2016, "앞 층 특징을 모두 이어 붙임", "D", False),
         ("MobileNet", 2017, "깊이별 분리 합성곱", "B", False),
         ("EfficientNet", 2019, "깊이·폭·해상도를 함께 키움", "C", False),
         ("ViT", 2020, "트랜스포머를 영상에 (32강)", "A", True),
         ("ConvNeXt", 2022, "트랜스포머 설계를 합성곱 망에", "D", False)]
ROW = {"A": (26, 42, 50), "B": (78, 94, 102), "C": (176, 192, 158), "D": (226, 242, 208)}   # 이름 y, 설명 y, 줄기 끝 y
for name, yr, idea, row, muted in ITEMS:
    x = X(yr)
    ny, iy, sy = ROW[row]
    s.line(x, AX, x, sy, "s-mu" if muted else "s-ac", 1.4, "3 3" if muted else None)
    s.circle(x, AX, 5, "f-mu" if muted else "f-ac")
    s.text(x, ny, f"{name} {yr}", "f-mu" if muted else "f-fg", 13, weight="600")
    s.text(x, iy, idea, "f-mu", 11)
s.save(os.path.join(OUT, "31-timeline.svg"))


# ---------------------------------------------------------------- 2. 잔차·병목·밀집 블록
s = Svg(780, 318, "기본 잔차 블록, 병목 블록, 밀집 블록의 구조 비교")
BW, BH = 136, 28


def box(cx, y, label, cls="s-fg f-acs"):
    s.rect(cx - BW / 2, y, BW, BH, cls, 1.3, rx=4)
    s.text(cx, y + BH / 2 + 4.5, label, "f-fg", 12)


def plus(cx, cy, r=11):
    s.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" class="s-bd f-bg" stroke-width="1.6"/>')
    s.line(cx - 6, cy, cx + 6, cy, "s-bd", 1.8)
    s.line(cx, cy - 6, cx, cy + 6, "s-bd", 1.8)


def res_panel(cx, title, in_label, boxes, note):
    s.text(cx, 22, title, "f-fg", 13, weight="600")
    s.text(cx, 50, in_label, "f-fg", 12)
    ys = [72 + i * 40 for i in range(len(boxes))] if len(boxes) == 3 else [82, 142]
    top = 56
    for y, lab in zip(ys, boxes):
        arrow(s, cx, top, cx, y)
        box(cx, y, lab)
        top = y + BH
    PY = 212
    arrow(s, cx, top, cx, PY - 11)
    plus(cx, PY)
    # 지름길: 입력에서 오른쪽으로 돌아 ⊕로
    rx = cx + BW / 2 + 16
    s.path(f"M {cx:.1f} 60 H {rx:.1f} V {PY:.1f}", "s-bd", 1.6)
    arrow(s, rx, PY, cx + 12, PY, "s-bd", "f-bd", 1.6)
    s.text(rx + 6, 140, "x", "f-bd", 13, anchor="start", weight="600")
    s.text(cx - BW / 2 - 8, (ys[0] + top) / 2 + 4, "F(x)", "f-mu", 12, anchor="end")
    arrow(s, cx, PY + 11, cx, 236)
    box(cx, 236, "ReLU", "s-mu f-sf")
    arrow(s, cx, 264, cx, 282)
    s.text(cx, 296, "출력 = ReLU(x + F(x))", "f-fg", 12)
    s.text(cx, 314, note, "f-mu", 11)


res_panel(135, "기본 잔차 블록", "입력 x (64채널)", ["3×3 합성곱 + ReLU", "3×3 합성곱"], "가중치 73,728개")
res_panel(395, "병목 블록", "입력 x (256채널)", ["1×1 합성곱 → 64", "3×3 합성곱 → 64", "1×1 합성곱 → 256"],
          "가중치 69,632개")

# 밀집 블록: 층마다 32채널을 만들어 뒤에 이어 붙임
cx0 = 640
s.text(cx0, 22, "밀집 블록 (성장률 32)", "f-fg", 13, weight="600")
x0, SC, H = 572, 0.8, 22
for i, ch in enumerate([64, 96, 128, 160]):
    y = 50 + i * 60
    segs = [64] + [32] * i
    x = x0
    for j, w in enumerate(segs):
        new = (i > 0 and j == len(segs) - 1)
        s.rect(x, y, w * SC, H, "s-bd f-bds" if new else "s-ac f-acs", 1.2)
        x += w * SC
    s.text(x + 6, y + H / 2 + 4.5, f"{ch}", "f-fg", 12, anchor="start")
    if i < 3:
        arrow(s, x0 + 20, y + H + 2, x0 + 20, y + 58)
        s.text(x0 + 30, y + H + 22, f"층 {i + 1}: 전부 받아 32채널 만듦", "f-mu", 11, anchor="start")
s.text(cx0, 296, "이어 붙임: 채널 = 64 + 32 × 층 수", "f-fg", 12)
s.text(cx0, 314, "(ResNet은 더하기라 채널 수 그대로)", "f-mu", 11)
s.save(os.path.join(OUT, "31-blocks.svg"))


# ---------------------------------------------------------------- 3. 학습 손실 곡선
d = R["deep"]
W, Hh = 700, 320
L, Rm, T, B = 62, 20, 20, 50
s = Svg(W, Hh, "평범한 8층·20층과 잔차 연결 20층의 에폭별 학습 손실")
x_of = lambda e: L + (e - 1) / 19 * (W - L - Rm)
y_of = lambda v: T + (1 - v / 2.0) * (Hh - T - B)
for v in [0, 0.5, 1.0, 1.5, 2.0]:
    s.line(L, y_of(v), W - Rm, y_of(v), "s-mu", 0.6, "2 4")
    s.text(L - 8, y_of(v) + 4, f"{v:.1f}", "f-mu", 11, anchor="end")
for e in [1, 5, 10, 15, 20]:
    s.text(x_of(e), Hh - B + 18, str(e), "f-mu", 11)
s.line(L, Hh - B, W - Rm, Hh - B, "s-fg", 1.2)
s.line(L, T, L, Hh - B, "s-fg", 1.2)
s.text((L + W - Rm) / 2, Hh - 8, "에폭", "f-mu", 12)
s.text(L - 30, T - 6, "손실", "f-mu", 12, anchor="start")
ln6 = math.log(6)
s.line(L, y_of(ln6), W - Rm, y_of(ln6), "s-fg", 1.2, "6 4")
s.text(W - Rm - 4, y_of(ln6) - 6, "ln 6 ≈ 1.79 (여섯 클래스 찍기 수준)", "f-fg", 11, anchor="end")
STY = {"plain20": ("s-bd", "f-bd", None, "평범한 20층"), "res20": ("s-ac", "f-ac", None, "잔차 연결 20층"),
       "plain8": ("s-mu", "f-mu", "5 3", "평범한 8층")}
for k, (sc, fc, dash, lab) in STY.items():
    v = d[k]["hist"]["train_loss"]
    pts = " L ".join(f"{x_of(i + 1):.1f} {y_of(y):.1f}" for i, y in enumerate(v))
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    s.add(f'<path d="M {pts}" class="{sc}" stroke-width="2.2" fill="none"{dd}/>')
    for i, y in enumerate(v):
        s.circle(x_of(i + 1), y_of(y), 2.6, fc)
def r3(v):
    """반올림(사사오입)으로 소수 셋째 자리: 0.9155 → 0.916"""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(v)).quantize(Decimal("0.001"), ROUND_HALF_UP))


# 범례: 곡선이 모두 0.45 아래인 오른쪽 위(손실 1.0~1.5 구간)
lx, ly = x_of(12), y_of(1.5)
for i, (k, (sc, fc, dash, lab)) in enumerate(STY.items()):
    yy = ly + i * 20
    s.line(lx, yy, lx + 28, yy, sc, 2.2, dash)
    s.text(lx + 36, yy + 4, f"{lab} (1에폭 {r3(d[k]['hist']['train_loss'][0])})", "f-fg", 12, anchor="start")
s.save(os.path.join(OUT, "31-deep-loss.svg"))
print("ok")
