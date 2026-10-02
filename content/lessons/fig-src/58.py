"""58강 그림과 검산.

실험 결과는 data/part8_runs.json의 fusion(run_part8.exp_fusion)에서 읽음. 학습은 다시 하지 않음.
    python3 58.py        # 검산 출력 + 그림 3개
검산
- 섀플리 값·DR·MBR을 JSON의 v(S)로 다시 계산해 JSON 값과 맞는지 확인
- 시험 화소 표본을 exp_fusion과 같은 난수 순서로 다시 뽑아 클래스 비율(입력을 모두 비웠을 때 0.3562 = 농경지 비율),
  구름 낀 표본 비율, SAR를 대각선으로 d화소 밀었을 때 중심 클래스가 그대로인 비율을 셈
- 코사인 특성 불일치 지수(D_align) 숫자 예: 무작위 특성 맵으로 채널 순서만 바꾼 경우·공간 이동·독립 특성을 비교
그림: 58-stages.svg(융합 단계), 58-missing.svg(조건별 정확도), 58-shapley.svg(섀플리 값)
"""
import json, os, sys
import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from svglib import Svg

OUT = os.path.join(HERE, "..", "fig")
DATA = os.path.join(HERE, "..", "data")
FU = json.load(open(os.path.join(DATA, "part8_runs.json"), encoding="utf-8"))["fusion"]
A = FU["acc"]


# ================================================================ 검산
def check():
    print("정확도", json.dumps(A, ensure_ascii=False))
    for k, v in FU["shapley"].items():
        vN, vE, vS, v0 = v["v_all"], v["v_eo"], v["v_sar"], v["v_none"]
        pe = 0.5 * (vE - v0) + 0.5 * (vN - vS)
        ps = 0.5 * (vS - v0) + 0.5 * (vN - vE)
        de, ds = (vN - vS) / vN, (vN - vE) / vN
        mcr = 100 * max(pe, 0) / (max(pe, 0) + max(ps, 0))
        print(f"{k:24s} phiE {pe:.4f}({v['phi_eo']}) phiS {ps:.4f}({v['phi_sar']}) 합 {pe + ps:.4f} = v(N)-v(0) {vN - v0:.4f} "
              f"DR_E {de:.4f} DR_S {ds:.4f} MBR {de / ds if ds else float('nan'):.1f}({v['mbr']}) MCR_E {mcr:.1f}")
    # 음의 전이 정의 A: 융합 < 단일 센서 모델 최고
    for m in ("fusion", "fusion_moddrop"):
        for c in ("clear", "cloudy"):
            best = max(A["eo"][c], A["sar"][c])
            print(f"정의A {m} {c}: 융합 {A[m][c]} vs 단일 최고 {best} -> {A[m][c] < best}")
    p = A["eo"]["clear"]
    print("맑음 표준오차(이항) 약", round(np.sqrt(p * (1 - p) / FU["n_test"]), 4), " 차이", round(A["eo"]["clear"] - A["fusion"]["clear"], 4))
    # 단독 센서 모델 두 개로 v(S)를 잡은 섀플리(참고)
    v0 = FU["shapley"]["fusion_clear"]["v_none"]
    vE, vS, vN = A["eo"]["clear"], A["sar"]["clear"], A["fusion"]["clear"]
    print("별도 모델 기준 phiE", round(0.5 * (vE - v0) + 0.5 * (vN - vS), 4), "phiS", round(0.5 * (vS - v0) + 0.5 * (vN - vE), 4))
    print("DR_EO 완화율", round(1 - FU["shapley"]["fusion_moddrop_clear"]["dr_eo"] / FU["shapley"]["fusion_clear"]["dr_eo"], 3))
    print("정렬 오차", FU["misalignment"], " 절반 기준", round(0.5 * A["fusion"]["clear"], 4))

    # exp_fusion의 표본 추출을 같은 난수 순서로 재현
    Z = np.load(os.path.join(DATA, "part4_scene.npz")); cls = Z["cls"]
    H, r = 300, 4
    rng = np.random.default_rng(0)
    def sample(rows, n):
        ys = rng.integers(rows[0] + r, rows[1] - r, n); xs = rng.integers(r, H - r, n)
        return ys, xs
    sample((0, 180), 12000)
    te_y, te_x = sample((196, 300), 5000)
    te_y = np.clip(te_y, 196 + 8, 300 - 9 - r); te_x = np.clip(te_x, 8 + r, H - 9 - r)
    cm = ndi.gaussian_filter(rng.normal(0, 1, (H, H)), 12); cloud = cm > np.quantile(cm[196:], 0.6)
    yt = cls[te_y, te_x]
    print("시험 클래스 비율(바다·하천·산림·농경지·시가지·도로·나지·갯벌·선박)", np.round(np.bincount(yt, minlength=9) / len(yt), 4))
    print("구름 낀 시험 표본 비율", cloud[te_y, te_x].mean(), " 면적 비율", round(cloud[196:].mean(), 3))
    for d in (1, 2, 4, 6, 8):
        print(f"SAR {d}화소 대각 이동: 중심 클래스 같음 {(cls[te_y + d, te_x + d] == yt).mean():.4f}")

    # D_align 숫자 예 (이 자습서에서 만든 무작위 특성 맵, 32채널 32×32, ReLU 뒤)
    g = np.random.default_rng(0)
    def feat():
        f = np.stack([ndi.gaussian_filter(g.normal(0, 1, (32, 32)), 2) for _ in range(32)])
        return np.maximum(f / f.std(), 0)
    def dalign(a, b):
        cos = (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-12)
        return 1 - cos.mean()
    F = feat()
    perm = g.permutation(32)
    print("D_align 같은 특성", round(dalign(F, F), 3), " 채널 순서만 바꿈", round(dalign(F, F[perm]), 3))
    for s in (1, 2, 4):
        print(f"D_align {s}화소 대각 이동", round(dalign(F, np.roll(F, (s, s), (1, 2))), 3))
    print("D_align 독립 특성", round(dalign(F, feat()), 3))


# ================================================================ 그림 1: 융합 단계
def arrow(s, x1, y, x2, cls="s-mu"):
    s.line(x1, y, x2, y, cls, 1.3)
    s.line(x2, y, x2 - 7, y - 4, cls, 1.3)
    s.line(x2, y, x2 - 7, y + 4, cls, 1.3)


def box(s, x, y, w, h, t, cls="s-fg f-sf", tcls="f-fg"):
    s.rect(x, y, w, h, cls, 1.2, rx=5)
    s.text(x + w / 2, y + h / 2 + 4, t, tcls, 12)


def fig_stages():
    s = Svg(700, 330, "다종 센서 융합의 세 단계. 초기 융합은 입력 채널을 이어 붙이고, 중간 융합은 센서별 백본의 특성을 합치고, 후기 융합은 센서별 모델의 결과를 합침")
    rows = [("초기 융합", 30), ("중간 융합", 130), ("후기 융합", 230)]
    for name, y in rows:
        s.text(18, y + 44, name, "f-fg", 13, anchor="start", weight="600")
    # 초기
    y = 30
    box(s, 110, y + 6, 70, 26, "광학 6")
    box(s, 110, y + 48, 70, 26, "SAR 1")
    s.line(180, y + 19, 205, y + 40, "s-mu", 1.3); s.line(180, y + 61, 205, y + 40, "s-mu", 1.3)
    box(s, 205, y + 27, 80, 26, "7채널 결합", "s-ac f-acs")
    arrow(s, 285, y + 40, 330)
    box(s, 330, y + 27, 110, 26, "백본 하나", "s-ac f-acs")
    arrow(s, 440, y + 40, 480)
    box(s, 480, y + 27, 70, 26, "출력")
    s.rect(570, y + 25, 118, 30, "s-bd f-bds", 1.2, rx=5)
    s.text(629, y + 45, "이 강의 실험", "f-bd", 12, weight="600")
    # 중간
    y = 130
    box(s, 110, y + 6, 70, 26, "광학")
    box(s, 110, y + 48, 70, 26, "SAR")
    arrow(s, 180, y + 19, 210); arrow(s, 180, y + 61, 210)
    box(s, 210, y + 6, 100, 26, "광학 백본")
    box(s, 210, y + 48, 100, 26, "SAR 백본")
    s.line(310, y + 19, 335, y + 40, "s-mu", 1.3); s.line(310, y + 61, 335, y + 40, "s-mu", 1.3)
    box(s, 335, y + 27, 105, 26, "특성 결합", "s-ac f-acs")
    arrow(s, 440, y + 40, 480)
    box(s, 480, y + 27, 70, 26, "출력")
    # 후기
    y = 230
    box(s, 110, y + 6, 70, 26, "광학")
    box(s, 110, y + 48, 70, 26, "SAR")
    arrow(s, 180, y + 19, 210); arrow(s, 180, y + 61, 210)
    box(s, 210, y + 6, 100, 26, "광학 모델")
    box(s, 210, y + 48, 100, 26, "SAR 모델")
    arrow(s, 310, y + 19, 335); arrow(s, 310, y + 61, 335)
    box(s, 335, y + 6, 70, 26, "결과")
    box(s, 335, y + 48, 70, 26, "결과")
    s.line(405, y + 19, 425, y + 40, "s-mu", 1.3); s.line(405, y + 61, 425, y + 40, "s-mu", 1.3)
    box(s, 425, y + 27, 55, 26, "병합", "s-ac f-acs")
    arrow(s, 480, y + 40, 500)
    box(s, 500, y + 27, 50, 26, "출력")
    for y in (118, 218):
        s.line(14, y, 690, y, "s-mu", 0.8, "2 4")
    s.save(os.path.join(OUT, "58-stages.svg"))


# ================================================================ 그림 2: 조건별 정확도
MODELS = [("eo", "광학 단독", "s-mu f-mu"), ("sar", "SAR 단독", "s-ok f-ok"),
          ("fusion", "융합", "s-bd f-bd"), ("fusion_moddrop", "융합+드롭아웃", "s-ac f-ac")]
CONDS = [("clear", "맑음"), ("cloudy", "구름"), ("cloudy_masked", "구름 마스킹"), ("eo_missing", "광학 결측")]


def fig_missing():
    s = Svg(700, 440, "조건별 시험 정확도. 광학 단독·SAR 단독·융합·모달리티 드롭아웃 융합 모델, 맑음·구름·구름 마스킹·광학 결측")
    L, Rr = 150, 620
    sx = lambda v: L + (Rr - L) * v
    y, bh = 44, 14
    for ci, (ck, cname) in enumerate(CONDS):
        y0 = y
        for mk, _, cls in MODELS:
            v = A[mk][ck]
            s.rect(L, y, sx(v) - L, bh, cls, 1.0)
            s.text(sx(v) + 6, y + bh - 3, f"{v:.3f}", "f-fg", 11, anchor="start")
            y += bh + 3
        s.text(L - 12, (y0 + y - 3) / 2 + 5, cname, "f-fg", 12, anchor="end")
        y += 14
    yb = y - 6
    s.line(L, yb, Rr, yb, "s-mu", 1.2)
    s.line(L, 34, L, yb, "s-mu", 1.2)
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        s.line(sx(v), yb, sx(v), yb + 4, "s-mu", 1.2)
        s.text(sx(v), yb + 17, f"{v:.1f}", "f-mu", 11)
    vn = FU["shapley"]["fusion_clear"]["v_none"]
    s.line(sx(vn), 34, sx(vn), yb, "s-fg", 1.0, "3 3")
    s.text(sx(vn) + 4, yb + 34, f"점선: 입력을 모두 비운 수준 {vn:.3f}", "f-mu", 11, anchor="middle")
    lx = L
    for i, (_, lab, cls) in enumerate(MODELS):
        x = lx + i * 118
        s.rect(x, 12, 14, 11, cls, 1.0)
        s.text(x + 20, 22, lab, "f-mu", 11, anchor="start")
    s.save(os.path.join(OUT, "58-missing.svg"))


# ================================================================ 그림 3: 섀플리 값
def fig_shapley():
    s = Svg(700, 300, "융합 모델과 모달리티 드롭아웃 모델의 맑음·구름 조건 섀플리 값. 드롭아웃 모델의 구름 조건에서 광학의 섀플리 값이 음수")
    L, Rr = 200, 640
    lo, hi = -0.2, 0.6
    sx = lambda v: L + (Rr - L) * (v - lo) / (hi - lo)
    cases = [("fusion_clear", "융합 · 맑음"), ("fusion_cloudy", "융합 · 구름"),
             ("fusion_moddrop_clear", "드롭아웃 · 맑음"), ("fusion_moddrop_cloudy", "드롭아웃 · 구름")]
    y, bh = 44, 15
    for k, name in cases:
        y0 = y
        for key, cls in (("phi_eo", "s-mu f-mu"), ("phi_sar", "s-ok f-ok")):
            v = FU["shapley"][k][key]
            x0, x1 = (sx(0), sx(v)) if v >= 0 else (sx(v), sx(0))
            s.rect(x0, y, max(x1 - x0, 1.0), bh, cls, 1.0)
            if v >= 0:
                s.text(x1 + 6, y + bh - 3, f"{v:+.4f}" if abs(v) < 0.01 else f"{v:+.3f}", "f-fg", 11, anchor="start")
            else:
                s.text(x0 - 6, y + bh - 3, f"{v:+.3f}", "f-bd", 11, anchor="end", weight="600")
            y += bh + 3
        s.text(L - 70, (y0 + y - 3) / 2 + 5, name, "f-fg", 12, anchor="end")
        y += 16
    yb = y - 8
    s.line(sx(0), 34, sx(0), yb, "s-fg", 1.2)
    s.line(L - 60, yb, Rr, yb, "s-mu", 1.2)
    for v in (-0.2, 0, 0.2, 0.4, 0.6):
        s.line(sx(v), yb, sx(v), yb + 4, "s-mu", 1.2)
        s.text(sx(v), yb + 17, f"{v:.1f}", "f-mu", 11)
    s.text((L + Rr) / 2, yb + 36, "섀플리 값 (정확도 단위)", "f-mu", 12)
    for i, (lab, cls) in enumerate((("광학 φ_EO", "s-mu f-mu"), ("SAR φ_SAR", "s-ok f-ok"))):
        x = L + i * 120
        s.rect(x, 12, 14, 11, cls, 1.0)
        s.text(x + 20, 22, lab, "f-mu", 11, anchor="start")
    s.save(os.path.join(OUT, "58-shapley.svg"))


if __name__ == "__main__":
    check()
    fig_stages()
    fig_missing()
    fig_shapley()
