"""32강 그림: 셀프 어텐션 계산 흐름, 작은 ViT 구조, 전역 어텐션 vs 창 어텐션(SVG)

어텐션 수치는 본문 1절의 손 계산 예(쿼리·키·값 각 3개, 차원 2)를 여기서 다시 계산해 넣음.
"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")


def arrow(s, x1, y1, x2, y2, cls="s-fg", fcls="f-fg", width=1.4, head=6):
    """직선 화살표(머리는 채운 삼각형 경로, marker를 쓰지 않음: id 금지)"""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    bx, by = x2 - ux * head, y2 - uy * head
    s.line(x1, y1, bx, by, cls, width)
    px, py = -uy * head * 0.55, ux * head * 0.55
    s.path(f"M {x2:.1f} {y2:.1f} L {bx + px:.1f} {by + py:.1f} L {bx - px:.1f} {by - py:.1f} Z", fcls, 0)


def box(s, cx, cy, w, h, label, cls="s-fg f-acs", size=12, tcls="f-fg"):
    s.rect(cx - w / 2, cy - h / 2, w, h, cls, 1.3, rx=4)
    s.text(cx, cy + size * 0.36, label, tcls, size)


# ---------------------------------------------------------------- 1. 셀프 어텐션 계산 흐름
Q = np.array([[1, 0], [0, 1], [1, 1.0]])
K = np.array([[2, 0], [0, 2], [1, 1.0]])
V = np.array([[1, 0], [0, 1], [0.5, 0.5]])
d = 2
S = Q @ K.T
Sc = S / math.sqrt(d)
A = np.exp(Sc) / np.exp(Sc).sum(1, keepdims=True)
O = A @ V
f2 = lambda v: f"{v:.2f}"
fv = lambda v: "(" + ", ".join(f"{t:g}" for t in v) + ")"

s = Svg(780, 250, "토큰 1의 쿼리가 키 세 개와 유사도를 계산하고 소프트맥스 가중치로 값을 섞는 과정")
ROWS = [78, 132, 186]
COL = {"q": 62, "k": 180, "s": 282, "sc": 372, "w": 478, "v": 596, "o": 716}
HEAD = [("k", "키 k"), ("s", "내적 q·k"), ("sc", "÷ √2"), ("w", "소프트맥스"), ("v", "값 v")]
for c, lab in HEAD:
    s.text(COL[c], 36, lab, "f-fg", 12, weight="600")
s.text(COL["q"], 36, "쿼리", "f-fg", 12, weight="600")
s.text(COL["o"], 36, "출력", "f-fg", 12, weight="600")
box(s, COL["q"], ROWS[1], 84, 30, "q1 = (1, 0)", "s-bd f-bds")
for i, y in enumerate(ROWS):
    arrow(s, COL["q"] + 42, ROWS[1], COL["k"] - 44, y, "s-mu", "f-mu", 1.2)
    box(s, COL["k"], y, 84, 30, f"k{i + 1} = {fv(K[i])}")
    arrow(s, COL["k"] + 42, y, COL["s"] - 26, y, "s-mu", "f-mu", 1.2)
    box(s, COL["s"], y, 50, 30, f"{S[0, i]:g}", "s-mu f-sf")
    arrow(s, COL["s"] + 25, y, COL["sc"] - 30, y, "s-mu", "f-mu", 1.2)
    box(s, COL["sc"], y, 58, 30, f2(Sc[0, i]), "s-mu f-sf")
    arrow(s, COL["sc"] + 29, y, COL["w"] - 44, y, "s-mu", "f-mu", 1.2)
    # 가중치: 막대 길이 = 가중치
    bw = 84
    s.rect(COL["w"] - bw / 2, y - 15, bw, 30, "s-mu f-bg", 1.2, rx=4)
    s.rect(COL["w"] - bw / 2, y - 15, bw * A[0, i], 30, "s-ac f-acs", 1.2, rx=4)
    s.text(COL["w"], y + 4.5, f2(A[0, i]), "f-fg", 12, weight="600")
    s.text(COL["w"] + bw / 2 + 12, y + 4.5, "×", "f-fg", 13)
    box(s, COL["v"], y, 92, 30, f"v{i + 1} = {fv(V[i])}")
    arrow(s, COL["v"] + 46, y, COL["o"] - 50, ROWS[1], "s-ac", "f-ac", 1.3)
box(s, COL["o"], ROWS[1], 96, 34, f"({f2(O[0, 0])}, {f2(O[0, 1])})", "s-ac f-acs", 13)
s.text(COL["o"], ROWS[1] + 36, "가중합", "f-mu", 11)
s.text(COL["w"], 230, f"합 = {A[0].sum():.0f}", "f-mu", 11)
s.save(os.path.join(OUT, "32-attention.svg"))


# ---------------------------------------------------------------- 2. 작은 ViT 구조
s = Svg(780, 350, "타일을 4×4 패치 64개로 잘라 패치 임베딩하고 분류 토큰과 위치 임베딩을 더해 인코더 층 4개를 지나 분류하는 구조")
# 타일(8×8 패치 격자)
TX, TY, C = 20, 60, 14
for r in range(8):
    for c in range(8):
        cls = "s-mu f-acs" if (r + c) % 2 == 0 else "s-mu f-sf"
        s.rect(TX + c * C, TY + r * C, C, C, cls, 0.8)
s.add(f'<rect x="{TX}" y="{TY}" width="{8 * C}" height="{8 * C}" class="s-fg" fill="none" stroke-width="1.4"/>')
s.rect(TX, TY, C, C, "s-bd f-bds", 1.4)
s.text(TX + 4 * C, TY + 8 * C + 20, "타일 4×32×32", "f-fg", 12)
s.text(TX + 4 * C, TY + 8 * C + 38, "4×4 패치 64개", "f-mu", 11)
arrow(s, TX + 8 * C + 6, TY + 4 * C, 178, TY + 4 * C)

# 토큰 열
TKX, TKW = 240, 110
s.text(TKX, 36, "패치 임베딩", "f-fg", 12, weight="600")
s.text(347, 36, "위치 임베딩", "f-fg", 12, weight="600")
TOK = [(56, "분류 토큰", "s-bd f-bds"), (94, "패치 1 → 64차원", "s-fg f-acs"),
       (132, "패치 2 → 64차원", "s-fg f-acs"), (170, "⋮", None), (204, "패치 64 → 64차원", "s-fg f-acs")]
for i, (y, lab, cls) in enumerate(TOK):
    if cls is None:
        s.text(TKX, y + 4, lab, "f-mu", 14)
        s.text(347, y + 4, "⋮", "f-mu", 14)
        continue
    box(s, TKX, y, TKW, 26, lab, cls, 11)
    s.text(311, y + 4.5, "+", "f-fg", 14)
    pos = {56: "위치 0", 94: "위치 1", 132: "위치 2", 204: "위치 64"}[y]
    box(s, 347, y, 50, 26, pos, "s-mu f-sf", 11)
# 묶음 괄호와 인코더 입력
s.line(382, 43, 382, 217, "s-fg", 1.3)
s.line(376, 43, 382, 43, "s-fg", 1.3)
s.line(376, 217, 382, 217, "s-fg", 1.3)
s.text(392, 128, "토큰", "f-mu", 11, anchor="start")
s.text(392, 142, "65개", "f-mu", 11, anchor="start")
s.path("M 382 160 H 420 V 42 H 520", "s-fg", 1.4)

# 인코더 층
EX, EW, ET, EB = 440, 190, 30, 282
s.rect(EX, ET, EW, EB - ET, "f-bg", 0, rx=6)
s.add(f'<rect x="{EX}" y="{ET}" width="{EW}" height="{EB - ET}" rx="6" class="s-mu" fill="none" stroke-width="1.2" stroke-dasharray="5 4"/>')
s.text(EX + EW / 2 + 90, 22, "인코더 층 × 4", "f-fg", 12, anchor="end", weight="600")
cx = 520
arrow(s, cx, 42, cx, 62)
box(s, cx, 74, 130, 24, "레이어 정규화", "s-mu f-sf", 11)
arrow(s, cx, 86, cx, 100)
box(s, cx, 112, 130, 24, "멀티헤드 어텐션", "s-ac f-acs", 11)


def plus(x, y, r=10):
    s.add(f'<circle cx="{x}" cy="{y}" r="{r}" class="s-bd f-bg" stroke-width="1.6"/>')
    s.line(x - 5.5, y, x + 5.5, y, "s-bd", 1.8)
    s.line(x, y - 5.5, x, y + 5.5, "s-bd", 1.8)


arrow(s, cx, 124, cx, 138)
plus(cx, 148)
s.path(f"M {cx} 52 H 608 V 148", "s-bd", 1.5)
arrow(s, 608, 148, cx + 11, 148, "s-bd", "f-bd", 1.5)
arrow(s, cx, 158, cx, 172)
box(s, cx, 184, 130, 24, "레이어 정규화", "s-mu f-sf", 11)
arrow(s, cx, 196, cx, 210)
box(s, cx, 222, 130, 24, "MLP 64→128→64", "s-ac f-acs", 11)
arrow(s, cx, 234, cx, 248)
plus(cx, 258)
s.path(f"M {cx} 163 H 608 V 258", "s-bd", 1.5)
arrow(s, 608, 258, cx + 11, 258, "s-bd", "f-bd", 1.5)
s.text(614, 205, "잔차", "f-bd", 11, anchor="start")
# 분류 헤드
arrow(s, cx, 268, cx, 308)
box(s, cx, 322, 150, 28, "분류 토큰 출력만", "s-bd f-bds", 12)
arrow(s, cx + 75, 322, 628, 322)
box(s, 700, 322, 136, 28, "LN → 선형 64→6", "s-fg f-acs", 12)
s.text(700, 348, "로짓 6개", "f-mu", 11)
s.save(os.path.join(OUT, "32-vit.svg"))


# ---------------------------------------------------------------- 3. 전역 어텐션 vs 창 어텐션
s = Svg(780, 272, "전역 어텐션, 4×4 창 어텐션, 창을 2칸 민 어텐션에서 한 토큰이 보는 범위")
G, N = 22, 8
QR, QC = 1, 2          # 쿼리 토큰(행, 열)


def grid(x0, y0, look, bounds, old=None):
    """look(r, c) -> 이 토큰을 보는가, bounds = 창 경계(토큰 단위, 양 끝 포함)"""
    for r in range(N):
        for c in range(N):
            if (r, c) == (QR, QC):
                cls = "s-mu f-bd"
            elif look(r, c):
                cls = "s-mu f-acs"
            else:
                cls = "s-mu f-bg"
            s.rect(x0 + c * G, y0 + r * G, G, G, cls, 0.7)
    if old:
        for b in old[1:-1]:
            s.line(x0 + b * G, y0, x0 + b * G, y0 + N * G, "s-mu", 1.6, "4 3")
            s.line(x0, y0 + b * G, x0 + N * G, y0 + b * G, "s-mu", 1.6, "4 3")
    for b in bounds:
        s.line(x0 + b * G, y0, x0 + b * G, y0 + N * G, "s-fg", 2.4)
        s.line(x0, y0 + b * G, x0 + N * G, y0 + b * G, "s-fg", 2.4)


Y0 = 44
PX = [40, 302, 564]
grid(PX[0], Y0, lambda r, c: True, [0, 8])
grid(PX[1], Y0, lambda r, c: r < 4 and c < 4, [0, 4, 8])
grid(PX[2], Y0, lambda r, c: r < 2 and 2 <= c < 6, [0, 2, 6, 8], old=[0, 4, 8])
TIT = ["전역 어텐션", "창 어텐션 (창 4×4)", "다음 층: 창을 2칸 밀어 나눔"]
L1 = ["64 × 64 = 4,096칸", "64 × 16 = 1,024칸", "옛 경계(점선)를 넘어 섞임"]
L2 = ["토큰 수의 제곱", "토큰 수에 비례", "창끼리 정보가 오감"]
for x, t, a, b in zip(PX, TIT, L1, L2):
    cx = x + N * G / 2
    s.text(cx, 28, t, "f-fg", 13, weight="600")
    s.text(cx, Y0 + N * G + 24, a, "f-fg", 12)
    s.text(cx, Y0 + N * G + 42, b, "f-mu", 11)
s.save(os.path.join(OUT, "32-window.svg"))
print("ok", A[0].round(3), O[0].round(3))
