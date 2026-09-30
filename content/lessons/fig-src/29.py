"""29강 그림과 본문 수치 검산

그림: 화소 펼침 MLP의 학습·검증 손실(규제 없음 vs 드롭아웃 0.5), 배치·레이어·그룹 정규화가 평균을 내는 범위 도식,
      학습 모드·추론 모드 추론 정확도 막대 (모두 SVG)
학습 결과는 data/part5_runs.json(run_part5.py의 reg·small·smallbs·mode·base 실험)에서 읽음. 새 학습은 하지 않음.
검산: 라벨 스무딩 목표·손실 바닥·로짓 차이, 가중치 감쇠 누적 배율, 조기 종료(참을성)별 멈춤 에폭, 표의 값
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from svglib import Svg

OUT = os.path.join(HERE, "..", "fig")
R = json.load(open(os.path.join(HERE, "..", "data", "part5_runs.json"), encoding="utf-8"))


# ================================================================ 검산
def early_stop(vl, patience):
    """검증 손실이 patience 에폭 연속 최솟값을 못 넘으면 멈춤 → (멈춘 에폭, 고른 에폭)"""
    best, be = float("inf"), 0
    for i, v in enumerate(vl):
        if v < best:
            best, be = v, i + 1
        elif i + 1 - be >= patience:
            return i + 1, be
    return len(vl), be


def check():
    G = R["reg"]
    print("설정 | 학습손실40 | 학습정확도40 | 검증손실 최소(에폭) | 검증손실40 | 검증정확도40 | 시험 last | best")
    for k, v in G.items():
        h = v["hist"]
        print(k, h["train_loss"][-1], h["train_acc"][-1], min(h["val_loss"]), v["best_epoch"], h["val_loss"][-1],
              h["val_acc"][-1], v["last"]["test_acc"], v["best"]["test_acc"])
    h = G["flat"]["hist"]
    va = np.array(h["val_acc"])
    print("규제 없음 검증 정확도 최대", va.max(), "에폭", va.argmax() + 1, " 3~40에폭 범위", va[2:].min(), va[2:].max())
    print("27강 30에폭 값과 같은지", R["mlp"]["flat"]["hist"]["val_loss"] == h["val_loss"][:30])
    for k in ("flat", "flat_dropout0.5", "flat_ls0.1"):
        print("참을성 3·5·10 →(멈춤, 고른 에폭)", k, [early_stop(G[k]["hist"]["val_loss"], p) for p in (3, 5, 10)])
    b = R["base"]["hist"]
    print("VD-CNN 참을성 3·5·10", [early_stop(b["val_loss"], p) for p in (3, 5, 10)],
          "검증 정확도 최대", max(b["val_acc"]), int(np.argmax(b["val_acc"])) + 1)

    # 라벨 스무딩(PyTorch 방식: (1−ε)·원-핫 + ε/K)
    K, e = 6, 0.1
    pt, po = 1 - e + e / K, e / K
    floor = -(pt * math.log(pt) + (K - 1) * po * math.log(po))
    print("라벨 스무딩 목표 정답 %.4f 나머지 %.4f 손실 바닥 %.3f 로짓 차이 %.2f" % (pt, po, floor, math.log(pt / po)))

    # 가중치 감쇠(AdamW) 누적 배율: 40에폭 × 66반복
    steps = 40 * math.ceil(4200 / 64)
    for lam in (0.1, 1.0):
        print("감쇠", lam, "걸음", steps, "배율 %.3f" % ((1 - 1e-3 * lam) ** steps))

    # 드롭아웃 크기 보정: 0.5로 끄고 남은 값에 2를 곱하면 기댓값이 그대로
    rng = np.random.default_rng(0)
    x = np.ones(1_000_000)
    keep = rng.random(x.size) > 0.5
    print("드롭아웃 0.5 보정 뒤 평균 %.4f" % (x * keep / 0.5).mean())

    print("배치 정규화 파라미터(채널마다 γ·β)", 2 * (16 + 32 + 64))
    S = R["small"]
    for k, v in S.items():
        print("small", k, "학습정확도", v["hist"]["train_acc"][-1], "검증정확도 최대", max(v["hist"]["val_acc"]),
              "last", v["last"]["test_acc"], "best", v["best"]["test_acc"], v["best_epoch"])
    for k, v in R["smallbs"].items():
        print("smallbs", k, "학습손실", v["hist"]["train_loss"][-1], "검증손실 최소", min(v["hist"]["val_loss"]),
              "last", v["last"]["test_acc"], "best", v["best"]["test_acc"])
    print("mode", R["mode"], "무작위 1/6 = %.3f" % (1 / 6))


# ================================================================ 그림 도구
def poly(s, pts, cls, w, dash=None):
    d = "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts)
    s.path(d, cls, w)
    if dash:
        s.parts[-1] = s.parts[-1].replace("/>", f' stroke-dasharray="{dash}"/>')


def axes(s, l, r, t, b, xt, sx, yt, sy, yfmt, xlabel, ylabel):
    s.line(l, b, r, b, "s-mu", 1.2)
    s.line(l, t, l, b, "s-mu", 1.2)
    for v in xt:
        s.line(sx(v), b, sx(v), b + 4, "s-mu", 1.2)
        s.text(sx(v), b + 17, f"{v}", "f-mu", 11)
    for v in yt:
        s.line(l - 4, sy(v), l, sy(v), "s-mu", 1.2)
        s.text(l - 7, sy(v) + 4, yfmt(v), "f-mu", 11, anchor="end")
    s.text((l + r) / 2, b + 36, xlabel, "f-mu", 12)
    s.text(l, t - 10, ylabel, "f-mu", 12, anchor="start")


# ---------------------------------------------------------------- 그림 1: 학습·검증 손실 곡선
def fig_curves():
    G = R["reg"]
    s = Svg(700, 300, "화소 펼침 MLP의 에폭별 학습 손실과 검증 손실. 왼쪽은 규제 없음, 오른쪽은 드롭아웃 0.5")
    panels = [("flat", 56, 330, "규제 없음", (9, 1.08), (11, 0.36), "middle"),
              ("flat_dropout0.5", 410, 684, "드롭아웃 0.5", (30, 0.84), (30, 0.10), "middle")]
    T, B = 40, 232
    for key, L, Rr, title, vpos, tpos, anc in panels:
        h = G[key]["hist"]
        sx = lambda e, L=L, Rr=Rr: L + (Rr - L) * (e - 1) / 39
        sy = lambda v: B - (B - T) * v / 1.6
        axes(s, L, Rr, T, B, (1, 10, 20, 30, 40), sx, (0, 0.4, 0.8, 1.2, 1.6), sy, lambda v: f"{v:.1f}", "에폭",
             f"손실 ({title})")
        poly(s, [(sx(e + 1), sy(v)) for e, v in enumerate(h["train_loss"])], "s-fg", 1.8)
        poly(s, [(sx(e + 1), sy(v)) for e, v in enumerate(h["val_loss"])], "s-bd", 1.8, "5 3")
        be = G[key]["best_epoch"]
        s.circle(sx(be), sy(h["val_loss"][be - 1]), 4.2, "f-bd")
        s.text(sx(vpos[0]), sy(vpos[1]), "검증", "f-bd", 12, anchor=anc, weight="600")
        s.text(sx(tpos[0]), sy(tpos[1]), "학습", "f-fg", 12, anchor=anc, weight="600")
    s.save(os.path.join(OUT, "29-curves.svg"))


# ---------------------------------------------------------------- 그림 2: 정규화가 평균을 내는 범위
def fig_norm():
    s = Svg(700, 250, "배치 정규화, 레이어 정규화, 그룹 정규화가 평균과 분산을 함께 내는 범위. "
                      "가로는 채널, 세로는 배치 안의 샘플, 칸 하나는 한 채널의 H×W 화소 전체")
    N, C, cell = 4, 6, 24
    panels = [("배치 정규화", 44, lambda n, c: c == 1, "채널마다, 배치 전체"),
              ("레이어 정규화", 282, lambda n, c: n == 2, "샘플마다, 모든 채널"),
              ("그룹 정규화", 520, lambda n, c: n == 2 and c in (2, 3), "샘플마다, 채널 묶음")]
    top = 70
    for title, x0, sel, sub in panels:
        gw = C * cell
        s.text(x0 + gw / 2, 26, title, "f-fg", 14, weight="600")
        for n in range(N):
            for c in range(C):
                cls = "s-mu f-bds" if sel(n, c) else "s-mu f-sf"
                s.rect(x0 + c * cell, top + n * cell, cell, cell, cls, 1.0)
        # 선택한 범위 테두리 강조
        cells = [(n, c) for n in range(N) for c in range(C) if sel(n, c)]
        n0, n1 = min(a for a, _ in cells), max(a for a, _ in cells)
        c0, c1 = min(b for _, b in cells), max(b for _, b in cells)
        s.rect(x0 + c0 * cell, top + n0 * cell, (c1 - c0 + 1) * cell, (n1 - n0 + 1) * cell, "s-bd f-bds", 2.4)
        for n, c in cells:   # 테두리 안쪽 칸 선을 다시 그림
            s.rect(x0 + c * cell, top + n * cell, cell, cell, "s-mu f-bds", 1.0)
        s.rect(x0 + c0 * cell, top + n0 * cell, (c1 - c0 + 1) * cell, (n1 - n0 + 1) * cell, "s-bd", 2.4)
        s.parts[-1] = s.parts[-1].replace('class="s-bd"', 'class="s-bd" fill="none"')
        # 축 표시
        s.text(x0 + gw / 2, top - 10, "채널 C →", "f-mu", 11)
        s.text(x0 - 8, top + N * cell / 2 + 4, "N", "f-mu", 11, anchor="end")
        s.line(x0 - 4, top + 4, x0 - 4, top + N * cell - 4, "s-mu", 1.0)
        s.text(x0 + gw / 2, top + N * cell + 24, sub, "f-mu", 12)
    s.text(350, 226, "칸 하나 = 샘플 하나의 한 채널(H×W 화소 전체)", "f-mu", 11)
    s.save(os.path.join(OUT, "29-norm.svg"))


# ---------------------------------------------------------------- 그림 3: 학습 모드 vs 추론 모드
def fig_mode():
    M = R["mode"]
    s = Svg(680, 210, "VD-CNN(best)의 시험 정확도. eval 모드, train 모드로 섞인 64장 배치, train 모드로 한 클래스만 모인 30장 배치")
    L, Rr = 250, 600
    sx = lambda v: L + (Rr - L) * v
    rows = [("eval 모드", M["eval_mode"], "f-ac"),
            ("train 모드, 섞인 64장", M["train_mode_mixed_batches64"], "f-mu"),
            ("train 모드, 한 클래스 30장", M["train_mode_single_class_batches"], "f-bd")]
    y0, bh, gap = 30, 30, 18
    for i, (name, v, cls) in enumerate(rows):
        y = y0 + i * (bh + gap)
        s.rect(L, y, sx(v) - L, bh, cls.replace("f-", "s-") + " " + cls, 1.0)
        s.text(L - 10, y + bh / 2 + 5, name, "f-fg", 12, anchor="end")
        s.text(sx(v) + 8, y + bh / 2 + 5, f"{v:.3f}", "f-fg", 12, anchor="start", weight="600")
    yb = y0 + 3 * (bh + gap) - gap + 8
    s.line(L, yb, Rr, yb, "s-mu", 1.2)
    s.line(L, y0 - 8, L, yb, "s-mu", 1.2)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        s.line(sx(v), yb, sx(v), yb + 4, "s-mu", 1.2)
        s.text(sx(v), yb + 17, f"{v:g}", "f-mu", 11)
    s.line(sx(1 / 6), y0 - 8, sx(1 / 6), yb, "s-mu", 1.2, "3 3")
    s.text(sx(1 / 6) + 4, y0 - 12, "찍기 1/6", "f-mu", 11, anchor="start")
    s.text((L + Rr) / 2, yb + 36, "시험 정확도", "f-mu", 12)
    s.save(os.path.join(OUT, "29-mode.svg"))


if __name__ == "__main__":
    check()
    fig_curves()
    fig_norm()
    fig_mode()
