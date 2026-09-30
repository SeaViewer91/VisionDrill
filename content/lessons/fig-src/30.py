"""30강 그림과 본문 수치 검산

그림: VD-CNN 구조도, 수용영역이 커지는 모습(1차원 단면), 표준 합성곱 vs 깊이별 분리 합성곱 도식 (모두 SVG)
검산: VD-CNN 층별 출력 모양·파라미터·곱셈-누산(MAC), 화소 펼침 MLP 파라미터, 출력 크기 공식, 수용영역,
      이동 등변성(4화소 이동 vs 1화소 이동), 깊이별 분리 합성곱 절감 비율. 학습은 하지 않음
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from svglib import Svg
from part5_common import TinyCNN, FlatMLP, n_params

OUT = os.path.join(HERE, "..", "fig")


# ---------------------------------------------------------------- 검산
def check():
    torch.manual_seed(0)
    m = TinyCNN().eval()
    x = torch.zeros(1, 4, 32, 32)
    print("층 | 출력 모양 | 파라미터 | 버퍼 | MAC")
    tot_p, tot_mac = 0, 0
    for mod in list(m.features) + list(m.head):
        x = mod(x)
        p = sum(q.numel() for q in mod.parameters())
        b = sum(q.numel() for q in mod.buffers())
        mac = 0
        if isinstance(mod, nn.Conv2d):
            mac = x.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, nn.Linear):
            mac = mod.in_features * mod.out_features
        tot_p += p
        tot_mac += mac
        print(type(mod).__name__, tuple(x.shape[1:]), p, b, mac)
    print("합계 파라미터", tot_p, "(배치 정규화 제외", tot_p - 2 * (16 + 32 + 64), ") MAC", tot_mac)
    print("화소 펼침 MLP", n_params(FlatMLP()), "첫 층", 4096 * 128 + 128)
    print("위치마다 가중치를 따로 두는 첫 층(국소 연결)", 4 * 9 * 16 * 32 * 32)
    print("펼침 헤드(64×8×8→6)", 64 * 8 * 8 * 6 + 6)

    def out_size(n, k, s=1, p=0, d=1):
        return (n + 2 * p - (d * (k - 1) + 1)) // s + 1
    for args in [(32, 3, 1, 1), (32, 3, 1, 0), (32, 3, 2, 1), (31, 3, 2, 1), (32, 2, 2, 0), (256, 7, 2, 3)]:
        print("출력 크기", args, out_size(*args))
    print("팽창 2, 패딩 2:", out_size(32, 3, 1, 2, 2))
    print("전치 합성곱 8→", (8 - 1) * 2 + 2)

    # 수용영역: 층 (k, s) 목록으로 계산
    rf, jump = 1, 1
    for name, k, s in [("conv1", 3, 1), ("pool1", 2, 2), ("conv2", 3, 1), ("pool2", 2, 2), ("conv3", 3, 1)]:
        rf += (k - 1) * jump
        jump *= s
        print("수용영역", name, rf, "간격", jump)
    print("18×18 / 32×32 =", 18 * 18 / 1024)
    print("팽창 1,2,4 세 층:", 1 + 2 * (1 + 2 + 4), " 팽창 1,1,1:", 1 + 2 * 3)

    # 이동 등변성: 4화소 이동은 안쪽 특징이 정확히 1칸 이동, 1화소 이동은 어느 칸과도 맞지 않음
    big = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        fa = m.features(big[:, :, 8:40, 8:40])
        fb = m.features(big[:, :, 4:36, 4:36])
        fc = m.features(big[:, :, 7:39, 7:39])
    print("4화소 이동 차이", (fa[:, :, 2:5, 2:5] - fb[:, :, 3:6, 3:6]).abs().max().item())
    print("1화소 이동 최소 차이", min((fa[:, :, 2:5, 2:5] - fc[:, :, 2 + i:5 + i, 2 + j:5 + j]).abs().max().item()
                                   for i in (-1, 0, 1) for j in (-1, 0, 1)), "특징 평균 크기", fa.abs().mean().item())

    # 깊이별 분리 합성곱 (VD-CNN 3번째 층 32→64, 8×8)
    std_w, dw_w, pw_w = 9 * 32 * 64, 9 * 32, 32 * 64
    print("표준 가중치", std_w, "깊이별+점별", dw_w, pw_w, dw_w + pw_w, "비율", (dw_w + pw_w) / std_w, 1 / 64 + 1 / 9)
    print("MAC 표준", std_w * 64, "분리", dw_w * 64, pw_w * 64, (dw_w + pw_w) * 64)
    print("2번째 층 16→32 비율", 1 / 32 + 1 / 9, "MAC", 9 * 16 * 16 * 16, 16 * 32 * 16 * 16)
    print("전체 파라미터 4342/24342 =", 4342 / 24342, " MAC 907648/2949504 =", 907648 / 2949504)
    print("1×1 64→16 파라미터", 64 * 16 + 16)


# ---------------------------------------------------------------- 그림 공통
def arrow(s, x1, y, x2, cls="s-fg", head="f-fg"):
    s.line(x1, y, x2 - 7, y, cls, 1.4)
    s.path(f"M {x2:.1f} {y:.1f} l -8 -4 l 0 8 Z", head, 1)


# ---------------------------------------------------------------- 1. VD-CNN 구조도
def fig_arch():
    s = Svg(760, 250, "VD-CNN 구조: 입력 4×32×32에서 합성곱·풀링을 거쳐 로짓 6개까지 층마다 바뀌는 모양과 파라미터 수")
    # (채널, 공간 크기, 아래 글자)
    T = [(4, 32, "4×32×32"), (16, 32, "16×32×32"), (16, 16, "16×16×16"), (32, 16, "32×16×16"),
         (32, 8, "32×8×8"), (64, 8, "64×8×8"), (64, 1, "64"), (6, 1, "6")]
    OPS = [("합성곱 3×3", "592+32"), ("최대 풀링", "0"), ("합성곱 3×3", "4,640+64"), ("최대 풀링", "0"),
           ("합성곱 3×3", "18,496+128"), ("전역 평균", "0"), ("선형", "390")]
    ym = 138
    xs = [52 + i * 94 for i in range(len(T))]
    wid = lambda c: {4: 8, 6: 8, 16: 14, 32: 22, 64: 34}[c]
    hei = lambda n: 8 if n == 1 else n * 3.4
    for i, (c, n, lab) in enumerate(T):
        w, h = wid(c), hei(n)
        cls = "s-ac f-acs" if i == 0 else ("s-bd f-bds" if i == len(T) - 1 else "s-fg f-sf")
        s.rect(xs[i] - w / 2, ym - h / 2, w, h, cls, 1.4)
        s.text(xs[i], ym + 32 * 3.4 / 2 + 22, lab, "f-fg", 12)
    for i, (op, p) in enumerate(OPS):
        x1 = xs[i] + wid(T[i][0]) / 2 + 4
        x2 = xs[i + 1] - wid(T[i + 1][0]) / 2 - 4
        arrow(s, x1, ym, x2, "s-mu", "f-mu")
        xm = (xs[i] + xs[i + 1]) / 2
        s.text(xm, 30, op, "f-fg", 12, weight="600")
        s.text(xm, 48, p, "f-ac" if p != "0" else "f-mu", 12)
    s.text(380, 238, "위: 층 종류와 파라미터 수(합성곱+배치 정규화)   아래: 출력 모양(채널×세로×가로)", "f-mu", 11.5)
    s.save(os.path.join(OUT, "30-vdcnn.svg"))


# ---------------------------------------------------------------- 2. 수용영역 (1차원 단면)
def fig_rf():
    s = Svg(760, 380, "VD-CNN 마지막 합성곱의 한 칸이 입력에서 보는 범위가 층을 내려갈수록 넓어져 18화소가 되는 모습")
    # 위에서 아래로: 합성곱3 → 풀링2 → 합성곱2 → 풀링1 → 합성곱1 → 입력
    rows = [("합성곱 3", 8, (4, 4), 18), ("풀링 2", 8, (3, 5), 10), ("합성곱 2", 16, (6, 11), 8),
            ("풀링 1", 16, (5, 12), 4), ("합성곱 1", 32, (10, 25), 3), ("입력", 32, (9, 26), 1)]
    x0, W = 96, 576          # 32칸 × 18
    rh, gap, y0 = 26, 28, 46
    s.text(x0 - 12, 26, "층", "f-mu", 12, anchor="end")
    s.text(x0 + W + 14, 26, "수용영역", "f-mu", 12, anchor="start")
    spans = []
    for r, (name, n, (a, b), rf) in enumerate(rows):
        y = y0 + r * (rh + gap)
        cw = W / n
        for i in range(n):
            on = a <= i <= b
            cls = ("s-bd f-bds" if r == 0 else "s-ac f-acs") if on else "s-mu f-bg"
            s.rect(x0 + i * cw, y, cw, rh, cls, 1)
        s.text(x0 - 12, y + rh / 2 + 5, name, "f-fg", 12, anchor="end")
        s.text(x0 + W + 14, y + rh / 2 + 5, f"{rf}화소", "f-bd" if r == 0 else "f-fg", 12, anchor="start",
               weight="600" if r == 0 else None)
        spans.append((x0 + a * cw, x0 + (b + 1) * cw, y))
    for (l1, r1, y1), (l2, r2, y2) in zip(spans[:-1], spans[1:]):
        s.line(l1, y1 + rh, l2, y2, "s-mu", 1, "3 3")
        s.line(r1, y1 + rh, r2, y2, "s-mu", 1, "3 3")
    yb = spans[-1][2] + rh + 12
    l, r = spans[-1][0], spans[-1][1]
    s.path(f"M {l:.1f} {yb:.1f} v 6 H {r:.1f} v -6", "s-ac", 1.4)
    s.text((l + r) / 2, yb + 24, "18화소 (32화소 타일의 가로 56%, 넓이 32%)", "f-ac", 12)
    s.save(os.path.join(OUT, "30-rf.svg"))


# ---------------------------------------------------------------- 3. 표준 vs 깊이별 분리 합성곱
def stack(s, x, y, w, h, n, cls, off=5):
    """뒤에서 앞으로 겹친 사각형 n장 (채널 묶음). (x, y)는 맨 앞 장의 왼쪽 위"""
    for i in reversed(range(n)):
        s.rect(x + i * off, y - i * off, w, h, cls, 1.1)


def fig_dw():
    s = Svg(760, 380, "표준 3×3 합성곱과 깊이별 분리 합성곱(깊이별 3×3 + 점별 1×1)의 계산 구조와 가중치 수 비교, 32채널에서 64채널")
    # 윗줄: 표준
    y = 70
    s.text(16, 28, "표준 합성곱", "f-fg", 13, anchor="start", weight="600")
    stack(s, 30, y, 56, 56, 6, "s-ac f-acs")
    s.text(72, y + 82, "입력 32×8×8", "f-fg", 12)
    arrow(s, 132, y + 28, 196)
    stack(s, 214, y + 14, 26, 26, 6, "s-fg f-sf", 4)
    s.text(240, y + 82, "3×3×32 커널", "f-fg", 12)
    s.text(240, y + 100, "64개", "f-mu", 12)
    arrow(s, 280, y + 28, 346)
    stack(s, 364, y, 56, 56, 9, "s-bd f-bds", 4)
    s.text(410, y + 82, "출력 64×8×8", "f-fg", 12)
    s.text(500, y + 16, "가중치 3×3×32×64", "f-fg", 12, anchor="start")
    s.text(500, y + 36, "= 18,432", "f-fg", 12, anchor="start", weight="600")
    s.text(500, y + 60, "곱셈-누산 1,179,648", "f-mu", 12, anchor="start")
    s.line(16, 196, 744, 196, "s-mu", 1, "4 4")
    # 아랫줄: 깊이별 분리
    y = 262
    s.text(16, 222, "깊이별 분리 합성곱", "f-fg", 13, anchor="start", weight="600")
    stack(s, 30, y, 56, 56, 6, "s-ac f-acs")
    s.text(72, y + 82, "입력 32×8×8", "f-fg", 12)
    arrow(s, 120, y + 28, 146)
    for i in range(3):
        s.rect(154 + i * 20, y + 18, 16, 16, "s-fg f-sf", 1.1)
    s.text(184, y + 82, "3×3×1 커널", "f-fg", 12)
    s.text(184, y + 100, "32개(채널마다)", "f-mu", 12)
    arrow(s, 222, y + 28, 250)
    stack(s, 262, y, 56, 56, 6, "s-mu f-sf")
    s.text(304, y + 82, "중간 32×8×8", "f-fg", 12)
    arrow(s, 350, y + 28, 376)
    stack(s, 386, y + 20, 10, 10, 6, "s-fg f-sf", 3)
    s.text(402, y + 82, "1×1×32 커널", "f-fg", 12)
    s.text(402, y + 100, "64개", "f-mu", 12)
    arrow(s, 420, y + 28, 446)
    stack(s, 456, y, 56, 56, 9, "s-bd f-bds", 4)
    s.text(500, y + 82, "출력 64×8×8", "f-fg", 12)
    s.text(572, y + 16, "가중치 288 + 2,048", "f-fg", 12, anchor="start")
    s.text(572, y + 36, "= 2,336 (12.7%)", "f-fg", 12, anchor="start", weight="600")
    s.text(572, y + 60, "곱셈-누산 149,504", "f-mu", 12, anchor="start")
    s.save(os.path.join(OUT, "30-dwsep.svg"))


if __name__ == "__main__":
    check()
    fig_arch()
    fig_rf()
    fig_dw()
