"""59강 그림과 검산.

학습 결과는 data/part8_runs.json(run_part8.py의 causal 실험)에서 읽음. 새 학습은 하지 않음.
    python3 59.py        # 검산 출력 + 그림 3개
그림: 59-scm.svg(바로잡은 구조적 인과 모델, 관측 vs 개입), 59-cue.svg(표식 실험의 세 시험), 59-results.svg(두 모델의 세 시험 정확도)
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from svglib import Svg

OUT = os.path.join(HERE, "..", "fig")
DATA = os.path.join(HERE, "..", "data")
C = json.load(open(os.path.join(DATA, "part8_runs.json"), encoding="utf-8"))["causal"]
P5 = json.load(open(os.path.join(DATA, "part5_runs.json"), encoding="utf-8"))
N_TEST = 900
CUE_POS = [(2, 3 + 5 * c) for c in range(6)]          # run_part8.CUE_POS와 같음(행, 열), 3×3
CLS = ["산림", "농경지", "시가지", "수계", "나지", "갯벌"]


# ================================================================ 검산
def check():
    print(C["note"])
    for k in ("shortcut", "cue_randomized"):
        r = C[k]
        print(k, r)
        print("   장수: 표식 없음 %.1f  맞는 자리 %.1f  틀린 자리 %.1f  뒤집힘 %.1f  예측 바뀜 %.1f" % tuple(
            r[x] * N_TEST for x in ("acc_clean", "acc_cue_aligned", "acc_cue_counter", "flip_to_cue_rate", "changed_pred_rate")))
        cbr = (r["acc_cue_aligned"] - r["acc_cue_counter"]) / r["acc_cue_aligned"]
        print("   정확도로 계산한 CBR 대응값 %.4f" % cbr)
        print("   맞는 자리 − 틀린 자리 %.4f" % (r["acc_cue_aligned"] - r["acc_cue_counter"]))
    print("원래 VD-CNN(5부 base, best) 시험 정확도", P5["base"]["best"]["test_acc"])
    print("표식 없음: 무작위화 − 바로가기 %.4f" % (C["cue_randomized"]["acc_clean"] - C["shortcut"]["acc_clean"]))
    print("ACE 경고선 0.40의 예: 0.95 →", round(0.95 - 0.40, 2))
    print("표식 자리(행, 열):", CUE_POS)


# ================================================================ 공통: 화살표·노드
def arrow(s, x1, y1, x2, y2, r1, r2, cls="s-fg", fcls="s-fg f-fg", width=1.6, dash=None):
    import math
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    sx, sy = x1 + ux * r1, y1 + uy * r1
    ex, ey = x2 - ux * (r2 + 1), y2 - uy * (r2 + 1)
    hx, hy = ex - ux * 9, ey - uy * 9                 # 화살촉 밑변 중심
    s.line(sx, sy, hx, hy, cls, width, dash)
    px, py = -uy * 4.5, ux * 4.5
    s.path(f"M{ex:.1f},{ey:.1f} L{hx + px:.1f},{hy + py:.1f} L{hx - px:.1f},{hy - py:.1f} Z", fcls, 1.0)
    return (sx + ex) / 2, (sy + ey) / 2


def node(s, x, y, lab, cls="s-fg f-bg", r=18):
    s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{cls}" stroke-width="1.6"/>')
    s.text(x, y + 5, lab, "f-fg", 15, weight="600")


# ================================================================ 그림 1: 구조적 인과 모델
def fig_scm():
    s = Svg(720, 360, "바로잡은 구조적 인과 모델. 왼쪽 학습 자료(관측)에서는 환경이 배경과 표적에 모두 영향을 주어 지름길이 생기고, 오른쪽 개입에서는 환경에서 배경으로 가는 화살표가 끊김")
    R = 18
    for k, ox in enumerate((0, 370)):
        pos = {"E": (ox + 170, 66), "C": (ox + 80, 146), "O": (ox + 260, 146), "X": (ox + 170, 216),
               "Yh": (ox + 170, 290), "Y": (ox + 260, 290)}
        title = "(가) 학습 자료: 관측" if k == 0 else "(나) 개입 do(C = c′)"
        s.text(ox + 170, 22, title, "f-fg", 14, weight="600")
        sc, sc_f = "s-bd", "s-bd f-bd"
        ok, ok_f = "s-ac", "s-ac f-ac"
        nu, nu_f = "s-mu", "s-mu f-mu"
        # 화살표
        if k == 0:
            arrow(s, *pos["E"], *pos["C"], R, R, sc, sc_f, 2.2)
            arrow(s, *pos["E"], *pos["O"], R, R, sc, sc_f, 2.2)
        else:
            mx, my = arrow(s, *pos["E"], *pos["C"], R, R, nu, nu_f, 1.3, "4 4")
            s.line(mx - 7, my - 7, mx + 7, my + 7, "s-bd", 2.4)
            s.line(mx - 7, my + 7, mx + 7, my - 7, "s-bd", 2.4)
            arrow(s, *pos["E"], *pos["O"], R, R, nu, nu_f, 1.6)
        arrow(s, *pos["E"], *pos["X"], R, R, nu, nu_f, 1.6)
        arrow(s, *pos["C"], *pos["X"], R, R, sc, sc_f, 2.2)
        arrow(s, *pos["O"], *pos["X"], R, R, ok, ok_f, 2.2)
        arrow(s, *pos["O"], *pos["Y"], R, R, ok, ok_f, 2.2)
        arrow(s, *pos["X"], *pos["Yh"], R, R, "s-fg", "s-fg f-fg", 2.2)
        # 노드
        node(s, *pos["E"], "E")
        node(s, *pos["C"], "C", "s-bd f-bds" if k == 1 else "s-fg f-bg")
        node(s, *pos["O"], "O")
        node(s, *pos["X"], "X")
        node(s, *pos["Yh"], "Ŷ")
        node(s, *pos["Y"], "Y")
        # 설명 글자
        s.text(pos["E"][0] + 26, pos["E"][1] + 5, "환경", "f-mu", 12, anchor="start")
        s.text(pos["C"][0], pos["C"][1] + 40, "c′로 고정" if k == 1 else "배경·단서", "f-bd" if k == 1 else "f-mu", 12)
        s.text(pos["O"][0] + 26, pos["O"][1] + 5, "표적", "f-mu", 12, anchor="start")
        s.text(pos["X"][0] + 26, pos["X"][1] + 18, "영상", "f-mu", 12, anchor="start")
        s.text(pos["Yh"][0] - 26, pos["Yh"][1] + 5, "예측", "f-mu", 12, anchor="end")
        s.text(pos["Y"][0] + 26, pos["Y"][1] + 5, "정답", "f-mu", 12, anchor="start")
    s.line(355, 40, 355, 310, "s-mu", 1.0, "2 4")
    # 범례
    ly = 340
    for x, cls, lab in ((150, "s-ac", "바른 길: 표적 → 영상 → 예측"), (420, "s-bd", "지름길: 환경 → 배경 → 영상")):
        s.line(x, ly - 4, x + 26, ly - 4, cls, 2.4)
        s.text(x + 34, ly, lab, "f-mu", 12, anchor="start")
    s.save(os.path.join(OUT, "59-scm.svg"))


# ================================================================ 그림 2: 표식 실험의 세 시험
def fig_cue():
    s = Svg(700, 256, "표식 실험. 32×32 타일 위쪽 가장자리의 여섯 자리가 여섯 클래스에 대응함. 시가지 타일을 예로 표식 없음, 맞는 자리, 틀린 자리(수계 자리)")
    k = 5.0                                            # 화소 하나 = 5단위
    tiles = [("표식 없음", None, None), ("맞는 자리", 2, "s-ok"), ("틀린 자리", 3, "s-bd")]
    for t, (title, slot, ocls) in enumerate(tiles):
        ox, oy = 40 + t * 225, 50
        s.text(ox + 80, 26, title, "f-fg", 14, weight="600")
        s.rect(ox, oy, 32 * k, 32 * k, "s-mu f-sf", 1.4)
        for c, (r, cc) in enumerate(CUE_POS):
            x, y = ox + cc * k, oy + r * k
            if c == slot:
                s.rect(x, y, 3 * k, 3 * k, f"{ocls} f-fg", 1.6)
            else:
                s.rect(x, y, 3 * k, 3 * k, "s-mu f-bg", 0.9, rx=0)
        s.text(ox + 80, oy + 32 * k - 16, "시가지 타일", "f-mu", 12)
        if slot is not None:
            r, cc = CUE_POS[slot]
            cx = ox + (cc + 1.5) * k
            lab = "시가지 자리" if slot == 2 else "수계 자리"
            s.line(cx, oy + (r + 3) * k + 2, cx, oy + 70, "s-mu", 1.0)
            s.text(cx, oy + 84, lab, "f-bd" if slot == 3 else "f-ok", 12)
    # 자리 이름(첫 타일 아래)
    ox, oy = 40, 50
    s.text(ox + 80, oy + 32 * k + 22, "왼쪽부터 산림·농경지·시가지", "f-mu", 11)
    s.text(ox + 80, oy + 32 * k + 37, "수계·나지·갯벌 자리", "f-mu", 11)
    s.text(40 + 225 + 80, oy + 32 * k + 22, "학습 자료의 90%가 이 상태", "f-mu", 11)
    s.text(40 + 450 + 80, oy + 32 * k + 22, "do(단서 = 다른 클래스)", "f-mu", 11)
    s.save(os.path.join(OUT, "59-cue.svg"))


# ================================================================ 그림 3: 두 모델의 세 시험 정확도
def fig_results():
    s = Svg(700, 250, "바로가기 모델과 단서 무작위화 모델의 표식 없음, 맞는 자리, 틀린 자리 시험 정확도")
    L, Rr = 150, 600
    lo, hi = 0.85, 1.0
    sx = lambda v: L + (Rr - L) * (v - lo) / (hi - lo)
    tests = [("acc_clean", "표식 없음", "s-mu f-mu"), ("acc_cue_aligned", "맞는 자리", "s-ok f-ok"), ("acc_cue_counter", "틀린 자리", "s-bd f-bd")]
    groups = [("shortcut", "바로가기 모델"), ("cue_randomized", "단서 무작위화")]
    y, bh = 40, 18
    for key, name in groups:
        y0 = y
        for t, lab, cls in tests:
            v = C[key][t]
            s.rect(L, y, sx(v) - L, bh, cls, 1.0)
            s.text(sx(v) + 6, y + bh - 4, f"{v:.3f}", "f-fg", 11, anchor="start")
            y += bh + 4
        s.text(L - 12, (y0 + y - 4) / 2 + 1, name, "f-fg", 13, anchor="end")
        r = C[key]
        s.text(L - 12, (y0 + y - 4) / 2 + 17, f"ACE {r['ace_true_class_conf']:.3f}", "f-mu", 11, anchor="end")
        y += 20
    yb = y - 10
    s.line(L, yb, Rr, yb, "s-mu", 1.2)
    s.line(L, 30, L, yb, "s-mu", 1.2)
    for v in (0.85, 0.90, 0.95, 1.00):
        s.line(sx(v), yb, sx(v), yb + 4, "s-mu", 1.2)
        s.text(sx(v), yb + 17, f"{v:.2f}", "f-mu", 11)
    s.text((L + Rr) / 2, yb + 36, "시험 정확도 (가로축은 0.85부터)", "f-mu", 12)
    lx = L
    for i, (t, lab, cls) in enumerate(tests):
        s.rect(lx + i * 110, 10, 16, 11, cls, 1.0)
        s.text(lx + i * 110 + 22, 20, lab, "f-mu", 11, anchor="start")
    s.save(os.path.join(OUT, "59-results.svg"))


if __name__ == "__main__":
    check()
    fig_scm()
    fig_cue()
    fig_results()
