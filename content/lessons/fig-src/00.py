"""0강 그림: 비전 모델 개발 한 바퀴(단계와 다루는 강), 이 책을 읽는 순서(부 갈래와 공통 예제 자료)

개념 도식이라 자료를 읽지 않음. 강 번호·강 수는 content/lessons/*.md 머리말과 parts.json에서 확인한 값
(1부 1~5강, 2부 6~12강, 3부 13~19강, 4부 20~26강, 5부 27~33강, 6부 34~42강, 7부 43~50강, 8부 51~61강)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

OUT = os.path.join(os.path.dirname(__file__), "..", "fig")


def arrow(s, x1, y1, x2, y2, cls="s-fg", fcls="f-fg", width=1.6, dash=None, head=7):
    """직선 화살표. 화살촉은 삼각형 경로로 직접 그림(marker의 id를 쓰지 않기 위해)"""
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
    s.line(x1, y1, bx, by, cls, width, dash)
    px, py = -math.sin(ang) * head * 0.5, math.cos(ang) * head * 0.5
    s.path(f"M {x2:.1f} {y2:.1f} L {bx + px:.1f} {by + py:.1f} L {bx - px:.1f} {by - py:.1f} Z", fcls, 0)


def poly_arrow(s, pts, cls="s-fg", fcls="f-fg", width=1.6, dash=None, head=7):
    """꺾인 화살표: 마지막 구간에만 화살촉"""
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        s.line(x1, y1, x2, y2, cls, width, dash)
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    arrow(s, x1, y1, x2, y2, cls, fcls, width, dash, head)


def orect(s, x, y, w, h, cls="s-fg", width=1.2, rx=0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return s.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" class="{cls}" fill="none" stroke-width="{width}"{d}/>')


# ---------------------------------------------------------------- 1. 개발 한 바퀴
# 위 줄은 왼쪽 → 오른쪽(1~4), 아래 줄은 오른쪽 → 왼쪽(5~8)으로 돌아 한 바퀴가 됨
STEPS = [("1 과제 정의", "13·34강"), ("2 자료 수집", "20~26강"), ("3 라벨링", "44강"), ("4 데이터셋 분할", "15·43강"),
         ("5 학습", "27~33·45~49강"), ("6 평가", "17·35~42강"), ("7 진단", "51~61강"), ("8 배포·운영", "50·60·61강")]
BW, BH, GX = 168, 58, 44
X0, Y_TOP, Y_BOT = 24, 20, 168
W = X0 * 2 + BW * 4 + GX * 3
s = Svg(W, Y_BOT + BH + 22, "비전 모델 개발 한 바퀴: 과제 정의, 자료 수집, 라벨링, 데이터셋 분할, 학습, 평가, 진단, 배포·운영과 각 단계를 다루는 강")


def col_x(c):
    return X0 + c * (BW + GX)


pos = {}
for i, (name, ref) in enumerate(STEPS):
    c, y = (i, Y_TOP) if i < 4 else (7 - i, Y_BOT)
    x = col_x(c)
    pos[i] = (x, y)
    cls = "s-ac f-acs" if i == 6 else "s-mu f-sf"
    s.rect(x, y, BW, BH, cls=cls, width=1.4, rx=8)
    s.text(x + BW / 2, y + 24, name, size=14, weight="bold")
    s.text(x + BW / 2, y + 44, ref, cls="f-mu", size=12)

# 정방향 흐름
for i in range(7):
    (xa, ya), (xb, yb) = pos[i], pos[i + 1]
    if i < 3:      # 위 줄 → 오른쪽
        arrow(s, xa + BW + 3, ya + BH / 2, xb - 3, yb + BH / 2)
    elif i == 3:   # 4 → 5 (오른쪽 끝에서 아래로)
        arrow(s, xa + BW / 2, ya + BH + 3, xb + BW / 2, yb - 3)
    else:          # 아래 줄 ← 왼쪽
        arrow(s, xa - 3, ya + BH / 2, xb + BW + 3, yb + BH / 2)

# 되돌아가는 흐름: 운영 → 과제 정의(다음 바퀴), 진단 → 자료 수집·라벨링(점선)
(x8, y8), (x1, y1) = pos[7], pos[0]
arrow(s, x8 + BW / 2, y8 - 3, x1 + BW / 2, y1 + BH + 3, cls="s-ac", fcls="f-ac", width=1.8)
s.text(x8 + BW / 2 + 10, (y1 + BH + y8) / 2 + 4, "다음 바퀴", cls="f-ac", size=12, anchor="start")
(x7, y7), (x2, y2) = pos[6], pos[1]
arrow(s, x7 + BW / 2, y7 - 3, x2 + BW / 2, y2 + BH + 3, cls="s-bd", fcls="f-bd", width=1.6, dash="5 4")
s.text(x7 + BW / 2 + 10, (y2 + BH + y7) / 2 - 4, "자료·라벨", cls="f-bd", size=12, anchor="start")
s.text(x7 + BW / 2 + 10, (y2 + BH + y7) / 2 + 12, "다시 보기", cls="f-bd", size=12, anchor="start")
s.save(os.path.join(OUT, "00-cycle.svg"))

# ---------------------------------------------------------------- 2. 읽는 순서
UP = [("1부 통계 기초", "1~5강", "강마다 작은 예"), ("2부 회귀·진단", "6~12강", "행정동 LST"),
      ("3부 ML 기본기", "13~19강", "토지피복 표본")]
LO = [("4부 영상·기하", "20~26강", "연안 장면"), ("5부 딥러닝", "27~33강", "타일·VD-CNN"),
      ("6부 태스크·평가", "34~42강", "항만 타일"), ("7부 학습 실무", "43~50강", "장면 10장")]
ZW, PW, PG, EW, GAP_SIDE = 104, 128, 22, 118, 44
LX0 = 10 + ZW + GAP_SIDE                 # 갈래 영역 시작
SPAN = PW * 4 + PG * 3                   # 아래 줄 폭
UW = 158
UG = (SPAN - UW * 3) / 2
EX = LX0 + SPAN + GAP_SIDE               # 8부 x
W2 = EX + EW + 10
BH2 = 66
YU, YL = 34, 150
YMID = (YU + BH2 + YL) / 2
YA = YL + BH2 + 30                       # 부록 띠
s = Svg(W2, YA + 52, "이 책을 읽는 순서: 0부 다음 통계·ML 갈래(1~3부)와 영상·DL 갈래(4~7부)로 나뉘고 8부에서 합쳐짐. 부록은 어디서나 참조")


def pbox(x, y, w, a, b, c, cls="s-mu f-sf"):
    s.rect(x, y, w, BH2, cls=cls, width=1.4, rx=8)
    s.text(x + w / 2, y + 22, a, size=13.5, weight="bold")
    s.text(x + w / 2, y + 40, b, size=12)
    s.text(x + w / 2, y + 57, c, cls="f-mu", size=11)


s.text(LX0, YU - 12, "통계·ML 갈래", cls="f-mu", size=12, anchor="start", weight="bold")
s.text(LX0, YL - 12, "영상·DL 갈래", cls="f-mu", size=12, anchor="start", weight="bold")
zy = YMID - BH2 / 2
pbox(10, zy, ZW, "0부 시작하기", "0강", "흐름·읽는 순서", cls="s-fg f-bg")
ey = YMID - BH2 / 2
pbox(EX, ey, EW, "8부 모델 진단", "51~61강", "앞 부 자료·모델", cls="s-ac f-acs")

ux = [LX0 + k * (UW + UG) for k in range(3)]
lx = [LX0 + k * (PW + PG) for k in range(4)]
for x, (a, b, c) in zip(ux, UP):
    pbox(x, YU, UW, a, b, c)
for x, (a, b, c) in zip(lx, LO):
    pbox(x, YL, PW, a, b, c)
for k in range(2):
    arrow(s, ux[k] + UW + 3, YU + BH2 / 2, ux[k + 1] - 3, YU + BH2 / 2)
for k in range(3):
    arrow(s, lx[k] + PW + 3, YL + BH2 / 2, lx[k + 1] - 3, YL + BH2 / 2)
# 0부 → 두 갈래
xm = 10 + ZW + GAP_SIDE / 2
s.line(10 + ZW + 3, YMID, xm, YMID, "s-fg", 1.6)
poly_arrow(s, [(xm, YMID), (xm, YU + BH2 / 2), (LX0 - 3, YU + BH2 / 2)])
poly_arrow(s, [(xm, YMID), (xm, YL + BH2 / 2), (LX0 - 3, YL + BH2 / 2)])
# 두 갈래 → 8부
xe = EX - GAP_SIDE / 2
s.line(ux[2] + UW + 3, YU + BH2 / 2, xe, YU + BH2 / 2, "s-fg", 1.6)
s.line(lx[3] + PW + 3, YL + BH2 / 2, xe, YL + BH2 / 2, "s-fg", 1.6)
s.line(xe, YU + BH2 / 2, xe, YL + BH2 / 2, "s-fg", 1.6)
arrow(s, xe, YMID, EX - 3, YMID)
# 부록 띠
orect(s, 10, YA, W2 - 20, 40, cls="s-mu", width=1.2, rx=8, dash="5 4")
s.text(W2 / 2, YA + 25, "부록 A 지시문 읽는 법 · B 헷갈리는 용어 쌍 · C 수식·기호표 · D Diagnostics 용어를 밖에서 말할 때",
       cls="f-mu", size=12.5)
s.save(os.path.join(OUT, "00-order.svg"))
print("W1", W, "W2", W2)
