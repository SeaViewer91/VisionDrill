"""56강 그림과 검산.

실험 수치는 data/part8_runs.json의 repr(run_part8.py의 exp_repr)에서 읽음. 새 학습은 하지 않음.
    python3 56.py        # 검산 출력 + 그림 3개 (ERF는 5부 기본 VD-CNN 가중치로 다시 계산, CPU 1스레드 약 10초)
그림: 56-spectrum.svg(기본 VD-CNN 마지막 특징의 특잇값 비율), 56-erf.svg(유효 수용영역 지도와 누적 비율),
      56-rfm.svg(객체 크기별 RFM과 판정 구간)
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from svglib import Svg

OUT = os.path.join(HERE, "..", "fig")
DATA = os.path.join(HERE, "..", "data")
R = json.load(open(os.path.join(DATA, "part8_runs.json"), encoding="utf-8"))["repr"]


def base_model_and_data():
    sys.path.insert(0, DATA)
    import torch
    torch.set_num_threads(1)
    import part5_common as P5C
    from part5_common import TinyCNN
    D = P5C.data()
    m = TinyCNN()
    m.load_state_dict(torch.load(os.path.join(DATA, ".part5_base.pt")))
    m.eval()
    return m, D, torch


def spectrum():
    """기본 VD-CNN 시험 900장의 전역 평균 풀링 특징(64차원) 특잇값. run_part8.eff_rank와 같은 계산"""
    m, D, torch = base_model_and_data()
    with torch.no_grad():
        F = m.features(D["x_test"]).mean(dim=(2, 3)).numpy()
    s = np.linalg.svd(F - F.mean(0), compute_uv=False)
    return s


def erf_map():
    """run_part8.exp_repr의 ERF 계산을 그대로 다시 함: 마지막 합성곱 (4,4) 칸의 입력 기울기 크기, 시험 300장 합"""
    m, D, torch = base_model_and_data()
    x = D["x_test"][:300].clone().requires_grad_(True)
    f = m.features(x)
    f[:, :, 4, 4].sum().backward()
    return x.grad.abs().sum(dim=(0, 1)).numpy()


def radii(g):
    yy, xx = np.mgrid[0:32, 0:32]
    cy, cx = (g * yy).sum() / g.sum(), (g * xx).sum() / g.sum()
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2), cy, cx


def r_at(g, rr, q):
    o = np.argsort(rr.ravel())
    cum = np.cumsum(g.ravel()[o]) / g.sum()
    return float(rr.ravel()[o][np.searchsorted(cum, q)])


# ================================================================ 검산
def check(s, g):
    b, d = R["base"], R["no_bn_lr0.02"]
    print("기본: 유효 랭크", b["eff_rank"], "CUR", b["cur"], "=", round(b["eff_rank"] / 64, 4), "DCR", b["dcr"], "NC1 학습/시험", b["nc1_train"], b["nc1_test"])
    print("정규화 없음·lr 0.02:", d)
    print("  DCR 채널 수:", [round(v * n) for v, n in zip(d["dcr"], (16, 32, 64))], "/ 16·32·64, 합", sum(round(v * n) for v, n in zip(d["dcr"], (16, 32, 64))), "/ 112")
    print("  CUR", round(d["eff_rank"] / 64, 4), " 살아 있는 마지막 블록 채널", 64 - round(d["dcr"][2] * 64))
    print("기본 VD-CNN 시험 정확도(best, part5_runs) 0.9744")
    # 손 계산 예: 특잇값 (4, 2, 1, 1)
    p = np.array([4, 2, 1, 1.]); p /= p.sum(); H = -(p * np.log(p)).sum()
    print("예 (4,2,1,1): p", p, "H", round(H, 4), "유효 랭크", round(math.exp(H), 3))
    # 특잇값 비율
    ps = s / s.sum(); ps2 = s ** 2 / (s ** 2).sum()
    print("특잇값 비율 위 8개", ps[:8].round(3), " 유효 랭크(σ)", round(math.exp(-(ps * np.log(ps + 1e-12)).sum()), 4),
          " (σ²로 정의하면)", round(math.exp(-(ps2 * np.log(ps2 + 1e-12)).sum()), 3))
    print("분산(σ²) 비율 위 5개 합", round(ps2[:5].sum(), 4), " 클래스 수 − 1 = 5, 5/64 =", 5 / 64)
    # ERF
    e = R["erf"]
    rr, cy, cx = radii(g)
    print("ERF 다시 계산: 중심", round(cy, 3), round(cx, 3), "R50", round(r_at(g, rr, .5), 4), "R95", round(r_at(g, rr, .95), 4), " JSON", e["center"], e["r50"], e["r95"])
    nz = np.argwhere(g > 0); print("기울기가 0이 아닌 범위(행·열)", nz.min(0), nz.max(0), " 이론 수용영역", e["theoretical_rf"])
    u = (g > 0).astype(float)
    print("TRF 18×18에 고르게 퍼졌다면 R50", round(r_at(u, rr, .5), 2), "R95", round(r_at(u, rr, .95), 2))
    yy, xx = np.mgrid[0:32, 0:32]
    sel = (abs(yy - cy) <= 4.5) & (abs(xx - cx) <= 4.5)
    print("가운데 9×9(", sel.sum(), "화소) 기울기 비율", round(float(g[sel].sum() / g.sum()), 3), " 넓이 비율", round(sel.sum() / 324, 3))
    print("R50 원 넓이 / TRF 넓이", round(math.pi * e["r50"] ** 2 / 324, 3), " 2·R95 =", round(2 * e["r95"], 3))
    # RFM과 넓이 기준
    for name, (w, h) in {"차량": (9, 4), "소형선박": (24, 8), "타일 전체": (32, 32), "선박": (200, 34), "정사각 6×6": (6, 6)}.items():
        S = math.hypot(w, h); A = w * h
        print(f"  {name} {w}×{h}: 대각 {S:.2f}  RFM {math.log(2 * e['r95'] / S):+.3f}  넓이 {A} 화소²  √넓이 {math.sqrt(A):.1f}  "
              f"EPS<16 {A < 16}  COCO 소형(<1024) {A < 1024}")
    print("  JSON rfm_examples", e["rfm_examples"])
    print("  선박 길이 대비 2R95 비율", round(2 * e["r95"] / 202.9, 3))
    # 어텐션 엔트로피
    E = np.array(R["vit_attention_entropy"]["normalized_entropy_by_layer_head"])
    print("어텐션 엔트로피(ln 65로 나눔) 층 평균", E.mean(1).round(3), "최소", E.min(), "최대", E.max(), " 시험 정확도", R["vit_attention_entropy"]["test_acc_best"])
    print("  exp(엔트로피) = 실효 토큰 수: 최소", round(65 ** E.min(), 1), "최대", round(65 ** E.max(), 1), " ln 65 =", round(math.log(65), 3))
    # 현장 예(가상 값)
    print("현장 예: ln(150/25.3) =", round(math.log(150 / 25.3), 3), " 넓이 12 화소² √", round(math.sqrt(12), 1))


# ================================================================ 그림
def fig_spectrum(s):
    p = s / s.sum()
    k = 16
    sv = Svg(700, 270, "기본 VD-CNN 마지막 특징 64차원의 특잇값 비율. 위 5개가 크고 그 뒤로 급히 작아짐")
    L, Rr, T, B = 70, 670, 30, 210
    ymax = 0.25
    sy = lambda v: B - (B - T) * v / ymax
    bw = (Rr - L) / k
    for i in range(k):
        x = L + i * bw + 4
        cls = "s-ac f-ac" if i < 5 else "s-mu f-sf"
        sv.rect(x, sy(p[i]), bw - 8, B - sy(p[i]), cls, 1)
        sv.text(L + i * bw + bw / 2, B + 16, str(i + 1), "f-mu", 11)
    for v in (0, 0.05, 0.10, 0.15, 0.20, 0.25):
        sv.line(L - 4, sy(v), L, sy(v), "s-mu", 1)
        sv.text(L - 8, sy(v) + 4, f"{v:.2f}", "f-mu", 11, anchor="end")
    sv.line(L, B, Rr, B, "s-mu", 1)
    sv.line(L, T, L, B, "s-mu", 1)
    sv.line(L, sy(1 / 64), Rr, sy(1 / 64), "s-bd", 1.2, dash="5 4")
    sv.text(Rr, sy(1 / 64) - 6, "64개가 모두 같다면 1/64", "f-bd", 11, anchor="end")
    sv.text(L + 2.5 * bw, sy(p[0]) - 10, "클래스 수 − 1 = 5개 방향", "f-ac", 12, weight="600")
    sv.text((L + Rr) / 2, B + 38, "특잇값 순위 k (64개 중 앞 16개)", "f-mu", 12)
    sv.text(L - 52, T - 12, "σ_k / Σσ", "f-mu", 11, anchor="start")
    sv.save(os.path.join(OUT, "56-spectrum.svg"))


def fig_erf(g):
    e = R["erf"]
    rr, cy, cx = radii(g)
    sv = Svg(700, 330, "VD-CNN 마지막 합성곱 가운데 칸의 유효 수용영역 지도와, 반지름별 누적 기울기 비율")
    c, X0, Y0 = 7, 40, 34
    gn = g / g.max()
    sv.rect(X0, Y0, 32 * c, 32 * c, "s-mu f-bg", 1)
    for i in range(32):
        for j in range(32):
            if gn[i, j] > 0.004:
                sv.add(f'<rect x="{X0 + j * c:.1f}" y="{Y0 + i * c:.1f}" width="{c}" height="{c}" class="f-ac" fill-opacity="{gn[i, j]:.3f}"/>')
    sv.add(f'<rect x="{X0 + 9 * c}" y="{Y0 + 9 * c}" width="{18 * c}" height="{18 * c}" class="s-bd" fill="none" stroke-width="1.5"/>')
    px, py = X0 + (cx + 0.5) * c, Y0 + (cy + 0.5) * c
    for r, dash in ((e["r50"], None), (e["r95"], "5 3")):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        sv.add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r * c:.1f}" class="s-fg" fill="none" stroke-width="1.5"{d}/>')
    ly = Y0 + 32 * c + 22
    sv.line(X0, ly - 4, X0 + 22, ly - 4, "s-bd", 1.5); sv.text(X0 + 28, ly, "이론 18×18", "f-mu", 11, anchor="start")
    sv.line(X0 + 108, ly - 4, X0 + 130, ly - 4, "s-fg", 1.5); sv.text(X0 + 136, ly, "R50", "f-mu", 11, anchor="start")
    sv.line(X0 + 108, ly + 14, X0 + 130, ly + 14, "s-fg", 1.5, dash="5 3"); sv.text(X0 + 136, ly + 18, "R95", "f-mu", 11, anchor="start")
    sv.text(X0 + 16 * c, Y0 - 12, "32×32 타일 위 기울기 크기", "f-mu", 12)
    # 누적 비율
    L, Rr, T, B = 360, 670, 34, 258
    rmax = 14
    sx = lambda r: L + (Rr - L) * r / rmax
    sy = lambda v: B - (B - T) * v
    u = (g > 0).astype(float)
    grid = np.linspace(0, rmax, 281)
    for arr, cls, dash in ((g, "s-ac", None), (u, "s-mu", "6 4")):
        tot = arr.sum()
        pts = [(sx(r), sy(arr[rr <= r].sum() / tot)) for r in grid]
        sv.path("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts), cls, 2 if dash is None else 1.6)
        if dash:
            sv.parts[-1] = sv.parts[-1].replace("/>", f' stroke-dasharray="{dash}"/>')
    for q in (0.5, 0.95):
        sv.line(L, sy(q), Rr, sy(q), "s-mu", 0.8, dash="2 3")
        sv.text(L - 6, sy(q) + 4, f"{q:g}", "f-mu", 11, anchor="end")
    sv.text(L - 6, sy(0) + 4, "0", "f-mu", 11, anchor="end")
    sv.text(L - 6, sy(1) + 4, "1", "f-mu", 11, anchor="end")
    sv.line(L, B, Rr, B, "s-mu", 1); sv.line(L, T, L, B, "s-mu", 1)
    for r in range(0, rmax + 1, 2):
        sv.line(sx(r), B, sx(r), B + 4, "s-mu", 1); sv.text(sx(r), B + 17, str(r), "f-mu", 11)
    sv.text((L + Rr) / 2, B + 36, "중심에서 반지름(화소)", "f-mu", 12)
    for r, cls, dy in ((e["r50"], "f-ac", -6), (r_at(u, rr, .5), "f-mu", 15), (e["r95"], "f-ac", -6), (r_at(u, rr, .95), "f-mu", 15)):
        q = 0.5 if r < 8 else 0.95
        sv.circle(sx(r), sy(q), 3, cls)
        sv.text(sx(r) + (-4 if cls == "f-ac" else 4), sy(q) + dy, f"{r:.1f}", cls, 11, anchor="end" if cls == "f-ac" else "start")
    sv.line(L + 150, T + 150, L + 174, T + 150, "s-ac", 2); sv.text(L + 180, T + 154, "실제 ERF", "f-mu", 11, anchor="start")
    sv.line(L + 150, T + 170, L + 174, T + 170, "s-mu", 1.6, dash="6 4"); sv.text(L + 180, T + 174, "TRF에 고르게 퍼졌다면", "f-mu", 11, anchor="start")
    sv.text(L + 4, T - 12, "누적 기울기 비율", "f-mu", 12, anchor="start")
    sv.save(os.path.join(OUT, "56-erf.svg"))


def fig_rfm():
    e = R["erf"]
    ex = [("선박 200×34", -2.3853), ("타일 전체 32×32", -0.8848), ("소형선박 24×8", -0.3034), ("차량 9×4", 0.6399)]
    assert all(abs(e["rfm_examples"][k] - v) < 1e-4 for k, v in zip(e["rfm_examples"], [x[1] for x in [ex[3], ex[2], ex[1], ex[0]]]))
    sv = Svg(700, 180, "객체 크기별 RFM 값과 판정 구간. 결핍은 −0.5 아래, 과다는 +1.0 위, 그 사이는 원문 간 경계가 엇갈리는 회색 지대")
    L, Rr = 50, 670
    lo, hi = -2.75, 1.5
    sx = lambda v: L + (Rr - L) * (v - lo) / (hi - lo)
    T, B = 42, 120
    sv.rect(sx(lo), T, sx(-0.5) - sx(lo), B - T, "f-bds", 0)
    sv.rect(sx(-0.5), T, sx(1.0) - sx(-0.5), B - T, "f-sf", 0)
    sv.rect(sx(1.0), T, sx(hi) - sx(1.0), B - T, "f-bds", 0)
    sv.text((sx(lo) + sx(-0.5)) / 2, T - 10, "수용영역 결핍", "f-bd", 12, weight="600")
    sv.text((sx(-0.5) + sx(1.0)) / 2, T - 10, "회색 지대", "f-mu", 12, weight="600")
    sv.text((sx(1.0) + sx(hi)) / 2, T - 10, "과다", "f-bd", 12, weight="600")
    sv.line(sx(0), T, sx(0), B, "s-mu", 1, dash="3 3")
    yax = B
    sv.line(L, yax, Rr, yax, "s-fg", 1.2)
    for v in (-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5):
        sv.line(sx(v), yax, sx(v), yax + 5, "s-fg", 1)
        sv.text(sx(v), yax + 19, (f"{v:+g}" if v else "0").replace("-", "−"), "f-mu", 11)
    sv.text((L + Rr) / 2, yax + 40, "RFM = ln(2·R95 / 객체 대각선)", "f-mu", 12)
    for i, (name, v) in enumerate(ex):
        y = 72 if i % 2 == 0 else 98
        sv.circle(sx(v), y, 5, "f-ac")
        left = name.startswith("타일")
        sv.text(sx(v) + (-9 if left else 9), y + 4, f"{name} ({v:+.2f})".replace("-", "−"), "f-fg", 11, anchor="end" if left else "start")
    sv.save(os.path.join(OUT, "56-rfm.svg"))


if __name__ == "__main__":
    s = spectrum()
    g = erf_map()
    check(s, g)
    fig_spectrum(s)
    fig_erf(g)
    fig_rfm()
