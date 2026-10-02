"""48강 그림과 검산.

학습 결과는 data/part7_runs.json(run_part7.py의 tl·rgb4 실험)에서 읽음.
    python3 48.py        # 검산 출력 + 그림 3개
    python3 48.py bn     # (약 30초, CPU 1스레드) 미세조정한 모델의 러닝 통계만 원천 것으로 되돌려 원래·대상 시험을 다시 잼
                         #  run_part7.exp_tl의 전체 미세조정(60·300장)을 같은 시드로 다시 돌림. JSON 값과 같게 나오는지도 확인
                         #  가중치(러닝 통계 제외)의 원천 대비 상대 L2 변화도 출력(60장 약 0.016, 300장 약 0.007)
그림: 48-methods.svg(무엇을 얼리고 학습하나), 48-results.svg(라벨 수·방법별 대상 정확도), 48-channels.svg(첫 합성곱 채널 확장)
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from svglib import Svg

OUT = os.path.join(HERE, "..", "fig")
DATA = os.path.join(HERE, "..", "data")
R = json.load(open(os.path.join(DATA, "part7_runs.json"), encoding="utf-8"))
TL, RGB4 = R["tl"], R["rgb4"]
METHODS = [("scratch", "처음부터"), ("linear_probe", "선형 탐침"), ("freeze_block1", "첫 블록 동결"),
           ("finetune_all", "전체 미세조정"), ("finetune_all_adabn", "AdaBN 후 미세조정")]


# ================================================================ 검산
def check():
    print("원천 모델 그대로", TL["source_only"])
    for k in ("adabn_60", "adabn_600"):
        print(k, TL[k])
    for n in ("n60", "n300"):
        print(n)
        for key, name in METHODS:
            v = TL[n][key]
            print("  %-8s 학습 파라미터 %6d  best 에폭 %2d  대상 %.4f  원래 %.4f  (last 대상 %.4f 원래 %.4f)" % (
                name, v["trainable_params"], v["best_epoch"], v["shift_test_best"], v["orig_test_best"],
                v["shift_test_last"], v["orig_test_last"]))
    # VD-CNN 블록별 파라미터(합성곱 가중치+편향, 배치 정규화 γ·β)
    blocks = [4 * 16 * 9 + 16 + 2 * 16, 16 * 32 * 9 + 32 + 2 * 32, 32 * 64 * 9 + 64 + 2 * 64, 64 * 6 + 6]
    print("블록별 파라미터", blocks, "합", sum(blocks), " 첫 블록 동결 시 학습", sum(blocks) - blocks[0])
    print("첫 합성곱 가중치(4밴드)", 16 * 4 * 9, " 3밴드", 16 * 3 * 9)
    print("rgb4", RGB4)
    print("원래 시험 하락: 0.9744 →", TL["source_only"]["shift_test"], "차", round(0.9744 - TL["source_only"]["shift_test"], 4))
    print("미세조정 원래 시험 범위", min(TL[n][k]["orig_test_best"] for n in ("n60", "n300") for k, _ in METHODS[2:]),
          max(TL[n][k]["orig_test_best"] for n in ("n60", "n300") for k, _ in METHODS[2:]))
    # 연무·계절 변환(make_part5): 청·녹·적·근적외에 0.035·0.022·0.010·0.004를 더하고 식생 화소 근적외 ×0.78
    print("근적외 감소 비율", round(1 - 0.78, 2))


def check_bn():
    """미세조정 뒤 원래 시험이 떨어진 원인 가르기: 러닝 통계만 원천 모델 것으로 되돌림"""
    import copy
    sys.argv = sys.argv[:1]
    sys.path.insert(0, DATA)
    os.environ["PART7_OUT"] = os.path.join(DATA, "part7_runs.json")   # 읽기만 함(save를 부르지 않음)
    import torch
    import run_part7 as RP
    from part5_common import evaluate
    from part7_common import fit
    D = RP.D
    test = {"shift_test": (D["x_shift"], D["y_shift"]), "orig_test": (D["x_test"], D["y_test"])}
    xv, yv = RP.target_tiles(25, 77)
    src_stats = {k: v.clone() for k, v in RP.base_model().state_dict().items() if "running" in k}
    for n in (10, 50):
        xt, yt = RP.target_tiles(n, 55)
        torch.manual_seed(0)
        m = RP.base_model()
        res = fit(m, xt, yt, xv, yv, test, epochs=60 if n == 10 else 40, bs=32, lr=1e-4)
        print(n * 6, "장 미세조정 best", res["best"], "에폭", res["best_epoch"],
              " JSON", TL[f"n{n * 6}"]["finetune_all"]["shift_test_best"], TL[f"n{n * 6}"]["finetune_all"]["orig_test_best"])
        m2 = copy.deepcopy(m)
        m2.load_state_dict({k: (src_stats[k] if k in src_stats else v) for k, v in m.state_dict().items()})
        print("   러닝 통계만 원천으로", {k: round(evaluate(m2, x, y)[1], 4) for k, (x, y) in test.items()})
        # 학습하는 파라미터(러닝 통계 제외)가 원천 대비 얼마나 움직였나: 상대 L2 변화
        b, s = RP.base_model().state_dict(), m.state_dict()
        keys = [k for k in s if "running" not in k and "num_batches" not in k]
        num = sum(((s[k] - b[k]) ** 2).sum().item() for k in keys)
        den = sum((b[k] ** 2).sum().item() for k in keys)
        print("   가중치 상대 변화(L2) %.4f" % (num / den) ** 0.5)


# ================================================================ 그림 1: 무엇을 얼리고 무엇을 학습하나
def fig_methods():
    s = Svg(700, 300, "전이 방법 네 가지가 VD-CNN의 어느 부분을 사전학습 가중치로 시작하고, 얼리고, 학습하는지")
    cols = [("블록 1", "624"), ("블록 2", "4,704"), ("블록 3", "18,624"), ("헤드", "390")]
    rows = [("처음부터", ["n", "n", "n", "n"], "24,342"),
            ("선형 탐침", ["z", "z", "z", "t"], "390"),
            ("첫 블록 동결", ["z", "t", "t", "t"], "23,718"),
            ("전체 미세조정", ["t", "t", "t", "t"], "24,342")]
    L, cw, ch, gx, gy, top = 130, 104, 34, 10, 12, 62
    for j, (name, p) in enumerate(cols):
        x = L + j * (cw + gx)
        s.text(x + cw / 2, 26, name, "f-fg", 13, weight="600")
        s.text(x + cw / 2, 44, p, "f-mu", 11)
    xr = L + 4 * (cw + gx) + 46
    s.text(xr, 26, "학습", "f-fg", 13, weight="600")
    s.text(xr, 44, "파라미터", "f-mu", 11)
    style = {"n": ("s-fg f-bg", "무작위", "f-fg"), "z": ("s-mu f-sf", "얼림", "f-mu"), "t": ("s-ac f-acs", "학습", "f-ac")}
    for i, (name, cells, n) in enumerate(rows):
        y = top + i * (ch + gy)
        s.text(L - 12, y + ch / 2 + 5, name, "f-fg", 13, anchor="end")
        for j, c in enumerate(cells):
            x = L + j * (cw + gx)
            cls, lab, tcls = style[c]
            s.rect(x, y, cw, ch, cls, 1.4, rx=4)
            s.text(x + cw / 2, y + ch / 2 + 5, lab, tcls, 12, weight="600" if c != "z" else None)
        s.text(xr, y + ch / 2 + 5, n, "f-fg", 12)
    yl = top + 4 * (ch + gy) + 14
    items = [("s-fg f-bg", "무작위 가중치로 시작해 학습"), ("s-ac f-acs", "사전학습 가중치로 시작해 학습"), ("s-mu f-sf", "사전학습 가중치 그대로 고정")]
    xs = [L - 100, L + 140, L + 380]
    for (cls, lab), x in zip(items, xs):
        s.rect(x, yl, 18, 14, cls, 1.2, rx=3)
        s.text(x + 26, yl + 12, lab, "f-mu", 11, anchor="start")
    s.save(os.path.join(OUT, "48-methods.svg"))


# ================================================================ 그림 2: 라벨 수·방법별 대상 정확도
def fig_results():
    s = Svg(700, 410, "대상 도메인 시험 정확도. 라벨 없는 두 경우(원천 모델 그대로, AdaBN만)와 라벨 60장·300장에서 방법 다섯 가지")
    L, Rr = 200, 610
    lo, hi = 0.5, 1.0
    sx = lambda v: L + (Rr - L) * (v - lo) / (hi - lo)
    groups = [("원천 그대로 (라벨 0)", [(TL["source_only"]["shift_test"], "s-bd f-bd")]),
              ("AdaBN만 (라벨 0)", [(TL["adabn_60"]["shift_test"], "s-ok f-ok")])]
    for key, name in METHODS:
        groups.append((name, [(TL["n60"][key]["shift_test_best"], "s-mu f-mu"), (TL["n300"][key]["shift_test_best"], "s-ac f-ac")]))
    y, bh = 34, 15
    tops = []
    for gi, (name, bars) in enumerate(groups):
        if gi == 2:
            y += 14
            s.line(30, y - 9, Rr + 60, y - 9, "s-mu", 1.0, "2 3")
        y0 = y
        for v, cls in bars:
            s.rect(L, y, sx(v) - L, bh, cls, 1.0)
            s.text(sx(v) + 6, y + bh - 3, f"{v:.3f}", "f-fg", 11, anchor="start")
            y += bh + 3
        s.text(L - 10, (y0 + y - 3) / 2 + 5, name, "f-fg", 12, anchor="end")
        tops.append(y0)
        y += 12
    yb = y - 4
    s.line(L, yb, Rr, yb, "s-mu", 1.2)
    s.line(L, 26, L, yb, "s-mu", 1.2)
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s.line(sx(v), yb, sx(v), yb + 4, "s-mu", 1.2)
        s.text(sx(v), yb + 17, f"{v:.1f}", "f-mu", 11)
    s.text((L + Rr) / 2, yb + 36, "대상 도메인 시험 정확도 (가로축은 0.5부터)", "f-mu", 12)
    # 범례
    lx, ly = L, 16
    for cls, lab, dx in (("s-mu f-mu", "라벨 60장", 0), ("s-ac f-ac", "라벨 300장", 110)):
        s.rect(lx + dx, ly - 10, 16, 11, cls, 1.0)
        s.text(lx + dx + 22, ly, lab, "f-mu", 11, anchor="start")
    s.save(os.path.join(OUT, "48-results.svg"))


# ================================================================ 그림 3: 첫 합성곱 채널 확장
def grid(s, x, y, size, cls):
    c = size / 3
    s.rect(x, y, size, size, cls, 1.3)
    for k in (1, 2):
        s.line(x + k * c, y, x + k * c, y + size, cls.split()[0], 0.8)
        s.line(x, y + k * c, x + size, y + k * c, cls.split()[0], 0.8)


def fig_channels():
    import math
    s = Svg(700, 300, "RGB 3채널로 사전학습한 첫 합성곱 가중치를 B2·B3·B4·B8 4채널 입력용으로 늘리는 방법")
    g = 46
    xs_top, y_top = [260, 340, 420], 40
    xs_bot, y_bot = [260, 340, 420, 500], 196
    s.text(205, y_top + g / 2 - 2, "사전학습 첫 층", "f-fg", 13, anchor="end", weight="600")
    s.text(205, y_top + g / 2 + 16, "입력 3채널", "f-mu", 11, anchor="end")
    s.text(205, y_bot + g / 2 - 2, "새 첫 층", "f-fg", 13, anchor="end", weight="600")
    s.text(205, y_bot + g / 2 + 16, "입력 4채널", "f-mu", 11, anchor="end")
    for x, n in zip(xs_top, ["적(R)", "녹(G)", "청(B)"]):
        grid(s, x - g / 2, y_top, g, "s-ac f-acs")
        s.text(x, y_top + g + 16, n, "f-fg", 12)
    for k, (x, n) in enumerate(zip(xs_bot, ["B2 청", "B3 녹", "B4 적", "B8 근적외"])):
        grid(s, x - g / 2, y_bot, g, "s-bd f-bds" if k == 3 else "s-ac f-acs")
        s.text(x, y_bot + g + 18, n, "f-fg", 12)
    for a, b in [(0, 2), (1, 1), (2, 0)]:          # R→B4, G→B3, B→B2 (순서가 뒤집힘)
        x1, y1 = xs_top[a], y_top + g + 26
        x2, y2 = xs_bot[b], y_bot - 8
        s.line(x1, y1, x2, y2, "s-ac", 1.4)
        ang = math.atan2(y2 - y1, x2 - x1)
        for d in (0.45, -0.45):
            s.line(x2, y2, x2 - 9 * math.cos(ang + d), y2 - 9 * math.sin(ang + d), "s-ac", 1.4)
    s.text(205, 160, "자리를 맞춰 복사", "f-ac", 12, anchor="end", weight="600")
    bx, by, bw, bh = 568, y_bot - 6, 124, 58
    s.rect(bx, by, bw, bh, "s-bd f-bds", 1.2, rx=6)
    s.text(bx + bw / 2, by + 20, "추가 밴드 초기화", "f-bd", 12, weight="600")
    s.text(bx + bw / 2, by + 42, "0 · 평균 · 무작위", "f-fg", 12)
    s.line(xs_bot[3] + g / 2 + 4, y_bot + g / 2, bx - 2, y_bot + g / 2, "s-bd", 1.4, "4 3")
    s.text(350, 292, "출력 채널 수와 커널 크기는 그대로, 입력 채널 축만 늘어남", "f-mu", 11)
    s.save(os.path.join(OUT, "48-channels.svg"))


if __name__ == "__main__":
    if sys.argv[1:] == ["bn"]:
        check_bn()
    else:
        check()
        fig_methods()
        fig_results()
        fig_channels()
