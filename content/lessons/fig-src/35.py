"""35강 그림과 본문 수치 확인: IoU·GIoU·DIoU 도식, 크기별 어긋남에 따른 IoU 곡선

결과 수치는 data/part6_runs.json의 iou(run_part6.py exp_iou)에서 읽음. 학습 없음.
본문의 손계산(교집합 예, 어긋남 공식, CIoU의 v·α, 마스크 경계 1화소, Dice 환산, 좌표 +1 규약)은 check()에서 다시 계산함.
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from svglib import Svg
from part6_eval import iou_matrix, iou_family

OUT = os.path.join(HERE, "..", "fig")
R = json.load(open(os.path.join(HERE, "..", "data", "part6_runs.json"), encoding="utf-8"))


# ================================================================ 본문 수치 확인
def check():
    I = R["iou"]
    g = I["family_gt"]
    print("정답 소형선박", g)
    b = [106, 100, 130, 108]
    ix = max(0, min(g[2], b[2]) - max(g[0], b[0]))
    iy = max(0, min(g[3], b[3]) - max(g[1], b[1]))
    print("교집합 예: 폭 %g 높이 %g 넓이 %g, 합집합 %g, IoU %.4f" % (ix, iy, ix * iy, 192 * 2 - ix * iy, ix * iy / (384 - ix * iy)))
    for k, v in I["family"].items():
        print("family", k, v)
    # 겹치지 않는 두 경우의 C, ρ², c²
    for b in ([128, 100, 152, 108], [164, 100, 188, 108], [106, 100, 130, 108]):
        cw, ch = max(g[2], b[2]) - min(g[0], b[0]), max(g[3], b[3]) - min(g[1], b[1])
        rho = abs((g[0] + g[2]) / 2 - (b[0] + b[2]) / 2)
        print(b, "C %g×%g=%g" % (cw, ch, cw * ch), "rho", rho, "rho2/c2 %.4f" % (rho ** 2 / (cw ** 2 + ch ** 2)))
    v = 4 / math.pi ** 2 * (math.atan(16 / 12) - math.atan(24 / 8)) ** 2
    a = v / (1 - 0.5 + v)
    print("모양 다름(16×12): v %.4f α %.4f αv %.4f, C 24×12=288 합집합 256 → GIoU %.4f" % (v, a, a * v, 0.5 - 32 / 288))
    print("10%% 확대 IoU 1/1.21 = %.4f" % (1 / 1.21))
    for name, d in I["shift"].items():
        print("shift", name, d)
    for w in (9, 24, 200):
        print("w", w, "(w-d)/(w+d):", [round((w - d) / (w + d), 4) for d in (1, 2, 3)], "0.5 경계 d=w/3 %.2f, 0.75 경계 d=w/7 %.2f" % (w / 3, w / 7))
    print("차량 폭 방향 1화소", iou_matrix([[0, 0, 9, 4]], [[0, 1, 9, 5]])[0, 0], "폭 4/7 = %.2f" % (4 / 7))
    print("차량 대각 1화소: 교집합 8×3=24, 합집합 72-24=48 →", 24 / 48)
    for c, v in I["best_iou_dist"].items():
        print("best_iou", c, v)
    print("차량 정답", R["data"]["per_class"][2], "짝 있는 정답", I["best_iou_dist"]["2"]["n"])
    ap = R["map"]["coco"]["per_class"]
    print("클래스별 AP at IoU 0.5 / 0.75:", {c: (ap[c][0], ap[c][5]) for c in ap})
    for w, h in ((9, 4), (24, 8), (200, 34)):
        print("마스크 %d×%d 한 화소 두껍게: IoU %.3f, 한 화소 얇게: %.3f" % (w, h, w * h / ((w + 2) * (h + 2)), (w - 2) * (h - 2) / (w * h)))
    for i in (0.5, 0.75):
        print("Dice from IoU", i, round(2 * i / (1 + i), 4))

    def iou_plus1(a, b):
        ix = min(a[2], b[2]) - max(a[0], b[0]) + 1
        iy = min(a[3], b[3]) - max(a[1], b[1]) + 1
        A = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
        B = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        return ix * iy / (A + B - ix * iy)
    try:
        import torch
        from torchvision.ops import box_iou, generalized_box_iou, complete_box_iou_loss
        gt, pr = torch.tensor([[100., 100, 124, 108]]), torch.tensor([[104., 98, 120, 110]])
        print("torchvision 확인", box_iou(gt, pr).item(), generalized_box_iou(gt, pr).item(), complete_box_iou_loss(pr, gt).item())
    except ImportError:
        pass
    print("차량 1화소, 끝 화소 포함(+1) 공식 %.4f vs 연속 좌표 %.4f" % (iou_plus1([0, 0, 9, 4], [1, 0, 10, 4]), 0.8))


def hollow(s, r, cls):
    s.rect(*r, cls=cls, width=2)
    s.parts[-1] = s.parts[-1].replace(f'class="{cls}"', f'class="{cls}" fill="none"')


# ================================================================ 그림 1: IoU·GIoU·DIoU 도식
def fig_family():
    s = Svg(720, 215, "IoU는 교집합과 합집합, GIoU는 두 상자를 감싸는 최소 상자 C의 빈 공간, DIoU는 중심 거리 ρ와 C의 대각선 c를 씀")
    # 패널마다 같은 두 상자: A(정답), B(예측). 단위 화소를 확대해 그림
    A = (0, 0, 110, 70)
    B = (60, 40, 160, 120)
    titles = ["IoU", "GIoU", "DIoU"]
    for k in range(3):
        ox, oy = 30 + k * 236, 40
        T = lambda r: (ox + r[0], oy + r[1], r[2] - r[0], r[3] - r[1])
        C = (min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3]))
        if k >= 1:
            s.rect(*T(C), cls="s-mu f-sf", width=1.3)
            s.parts[-1] = s.parts[-1].replace("/>", ' stroke-dasharray="5 4"/>')
            s.rect(*T(A), cls="f-bg", width=0)
            s.rect(*T(B), cls="f-bg", width=0)
        inter = (max(A[0], B[0]), max(A[1], B[1]), min(A[2], B[2]), min(A[3], B[3]))
        s.rect(*T(inter), cls="f-bds", width=0)
        hollow(s, T(A), "s-ac")
        hollow(s, T(B), "s-bd")
        s.text(ox + 8, oy + A[3] - 8, "A", "f-ac", 13, anchor="start", weight="bold")
        s.text(ox + B[0] + 8, oy + B[3] - 8, "B", "f-bd", 13, anchor="start", weight="bold")
        s.text(ox + 80, oy - 16, titles[k], "f-fg", 14, weight="bold")
        if k == 0:
            s.text(ox + (inter[0] + inter[2]) / 2, oy + (inter[1] + inter[3]) / 2 + 5, "A∩B", "f-bd", 11)
            s.text(ox + 80, oy + 156, "교집합 ÷ 합집합", "f-mu", 12)
        if k == 1:
            s.text(ox + C[2] - 6, oy + C[1] + 16, "C", "f-mu", 13, anchor="end", weight="bold")
            s.text(ox + 80, oy + 156, "빈 공간(회색) ÷ C 를 뺌", "f-mu", 12)
        if k == 2:
            ca = (ox + (A[0] + A[2]) / 2, oy + (A[1] + A[3]) / 2)
            cb = (ox + (B[0] + B[2]) / 2, oy + (B[1] + B[3]) / 2)
            s.line(ox + C[0], oy + C[1], ox + C[2], oy + C[3], "s-mu", 1.3, dash="3 3")
            s.line(*ca, *cb, "s-fg", 2)
            s.circle(*ca, 3.5, "f-ac")
            s.circle(*cb, 3.5, "f-bd")
            s.text((ca[0] + cb[0]) / 2 - 8, (ca[1] + cb[1]) / 2 + 2, "ρ", "f-fg", 14, anchor="end", weight="bold")
            s.text(ox + 146, oy + 99, "c", "f-mu", 14, weight="bold")
            s.text(ox + 80, oy + 156, "ρ² ÷ c² 를 뺌", "f-mu", 12)
    s.save(os.path.join(OUT, "35-family.svg"))


# ================================================================ 그림 2: 크기별 어긋남과 IoU
def fig_shift():
    s = Svg(640, 300, "가로 길이가 w화소인 상자를 그 방향으로 d화소 밀었을 때의 IoU (w−d)/(w+d). 선박 200, 소형선박 24, 차량 길이 9, 차량 폭 4")
    L, Rr, T, B = 60, 470, 30, 250
    sx = lambda d: L + (Rr - L) * d / 4
    sy = lambda v: B - (B - T) * v
    s.line(L, B, Rr, B, "s-mu", 1.2)
    s.line(L, T, L, B, "s-mu", 1.2)
    for d in range(5):
        s.line(sx(d), B, sx(d), B + 4, "s-mu", 1.2)
        s.text(sx(d), B + 17, f"{d}", "f-mu", 11)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        s.line(L - 4, sy(v), L, sy(v), "s-mu", 1.2)
        s.text(L - 7, sy(v) + 4, f"{v:g}", "f-mu", 11, anchor="end")
    for v in (0.5, 0.75):
        s.line(L, sy(v), Rr, sy(v), "s-mu", 1, dash="4 4")
    s.text((L + Rr) / 2, B + 38, "어긋남 d (화소, 1화소 = 0.5 m)", "f-mu", 12)
    s.text(L, T - 12, "IoU", "f-mu", 12, anchor="start")
    SER = [(200, "s-ok", None, "f-ok", "선박 길이 200"),
           (24, "s-ac", None, "f-ac", "소형선박 길이 24"),
           (9, "s-bd", None, "f-bd", "차량 길이 9"),
           (4, "s-bd", "5 3", "f-bd", "차량 폭 4")]
    for w, cls, dash, tcls, name in SER:
        ds = np.linspace(0, 4, 81)
        pts = [(sx(d), sy(max((w - d) / (w + d), 0))) for d in ds]
        s.path("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts), cls, 2)
        if dash:
            s.parts[-1] = s.parts[-1].replace("/>", f' stroke-dasharray="{dash}"/>')
        for d in (1, 2, 3):
            v = (w - d) / (w + d)
            s.circle(sx(d), sy(v), 3.2, tcls)
        yend = sy(max((w - 4) / (w + 4), 0))
        s.text(Rr + 8, yend + 4, name, tcls, 12, anchor="start")
    s.save(os.path.join(OUT, "35-shift.svg"))


if __name__ == "__main__":
    check()
    fig_family()
    fig_shift()
