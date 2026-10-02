"""49강 그림과 본문 수치 검산

그림: 시드 다섯 개의 best·last 시험 정확도(배치 정규화 있음·없음, 평균±표준편차), 체크포인트 도식(모두 SVG)
학습 결과는 data/part7_runs.json의 seeds 실험(run_part7.py exp_seeds)과 data/part5_runs.json의 base(시드 0 검증 손실 곡선)에서 읽음.
검산: 시드별 표, 평균·표준편차, 웰치 t검정(4강)과 95% 신뢰구간, 검정력, 시드별 차이, 부동소수 덧셈 순서, 스레드 수에 따른 행렬곱 차이
    python3 49.py          # 검산 + 그림
    python3 49.py rerun    # 추가 확인: 같은 시드로 2에폭 두 번 학습해 가중치가 비트 단위로 같은지(약 35초)
"""
import json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from svglib import Svg

OUT = os.path.join(HERE, "..", "fig")
DATA = os.path.join(HERE, "..", "data")
S = json.load(open(os.path.join(DATA, "part7_runs.json"), encoding="utf-8"))["seeds"]
P5 = json.load(open(os.path.join(DATA, "part5_runs.json"), encoding="utf-8"))


# ================================================================ 검산
def welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean()
    h = stats.t.ppf(0.975, df) * se
    t = stats.ttest_ind(a, b, equal_var=False)
    return d, se, t.statistic, t.pvalue, df, d - h, d + h


def check():
    bn, nb = S["bn"]["runs"], S["no_bn"]["runs"]
    print("시드 | BN best에폭 last best shift | 없음 best에폭 last best shift")
    for r, q in zip(bn, nb):
        print(r["seed"], r["best_epoch"], r["test_last"], r["test_best"], r["shift_best"], "|",
              q["best_epoch"], q["test_last"], q["test_best"], q["shift_best"])
    for key in ("test_best", "test_last", "shift_best"):
        a = np.array([r[key] for r in bn]); b = np.array([r[key] for r in nb])
        print(f"{key}: BN {a.mean():.4f}±{a.std(ddof=1):.4f} ({a.min()}~{a.max()})  없음 {b.mean():.4f}±{b.std(ddof=1):.4f} ({b.min()}~{b.max()})")
        d, se, t, p, df, lo, hi = welch(a, b)
        print(f"   웰치: 차이 {d:.4f} 표준오차 {se:.4f} t {t:.2f} p {p:.4f} 자유도 {df:.1f} 95% CI {lo:.4f}~{hi:.4f}")
        print("   시드별 차이(같은 시드 번호끼리)", np.round(a - b, 4))
        pr = stats.ttest_rel(a, b); dd = a - b; hh = stats.t.ppf(0.975, 4) * dd.std(ddof=1) / np.sqrt(5)
        print(f"   대응표본: t {pr.statistic:.2f} p {pr.pvalue:.4f} 95% CI {dd.mean() - hh:.4f}~{dd.mean() + hh:.4f}")
    print("JSON 요약값", {k: S["bn"][k] for k in ("last_mean", "last_sd", "best_mean", "best_sd")},
          {k: S["no_bn"][k] for k in ("last_mean", "last_sd", "best_mean", "best_sd")})
    a = np.array([r["test_best"] for r in bn]); b = np.array([r["test_best"] for r in nb])
    sdp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    dz = (a.mean() - b.mean()) / sdp
    try:
        from statsmodels.stats.power import TTestIndPower
        print(f"효과크기 d {dz:.2f}, 참 효과가 이만할 때 n=5 검정력 {TTestIndPower().power(dz, 5, 0.05):.2f}")
    except ImportError:
        pass
    # 같은 시드면 BN 있음·없음 모델의 합성곱·선형 층 초기 가중치가 같은가(BN 층은 난수를 쓰지 않음)
    import torch
    sys.path.insert(0, DATA)
    from part5_common import TinyCNN
    torch.manual_seed(0); ma = TinyCNN(norm="bn").state_dict()
    torch.manual_seed(0); mb = TinyCNN(norm="none").state_dict()
    print("같은 시드 공통 층 초기 가중치 같음", all(torch.equal(ma[k], mb[k]) for k in mb))
    print("시험 900장에서 차이 0.0113 = 약 %.1f장" % (0.0113 * 900))
    print("같은 시드 두 번(3에폭 학습 손실)", S["same_seed_identical"], S["same_seed_losses"])
    b5 = P5["base"]
    print("시드 0 BN = 5부 기본 학습?", b5["best_epoch"], b5["best"]["test_acc"], b5["last"]["test_acc"], b5["best"]["shift_acc"])
    print("시드 0 BN last−best", [round(r["test_last"] - r["test_best"], 4) for r in bn])

    # 부동소수 덧셈은 순서에 따라 결과가 다름(float32)
    x = np.float32([1e8, 1.0, -1e8])
    print("(1e8+1)-1e8 =", (x[0] + x[1]) + x[2], " (1e8-1e8)+1 =", (x[0] + x[2]) + x[1])

    # CPU 스레드 수에 따른 행렬곱 차이
    import torch
    torch.manual_seed(0)
    A, B = torch.randn(512, 4096), torch.randn(4096, 512)
    n0 = torch.get_num_threads()
    torch.set_num_threads(1); r1 = A @ B
    torch.set_num_threads(2); r2 = A @ B
    torch.set_num_threads(n0)
    print("스레드 1 vs 2 행렬곱 같음?", torch.equal(r1, r2), "최대 차이 %.2e" % (r1 - r2).abs().max().item(),
          "값 표준편차 %.1f" % r1.std().item(), "(torch", torch.__version__, ")")


def rerun():
    import torch
    sys.path.insert(0, DATA)
    import part5_common as P5C
    from part5_common import TinyCNN
    from part7_common import fit
    D = P5C.data()
    ws, ls = [], []
    for _ in range(2):
        torch.manual_seed(0)
        m = TinyCNN()
        r = fit(m, D["x_train"], D["y_train"], D["x_val"], D["y_val"], None, epochs=2, seed=0)
        ws.append({k: v.clone() for k, v in m.state_dict().items()}); ls.append(r["hist"]["train_loss"])
    print("같은 시드 2에폭 학습 손실", ls, "가중치·러닝 통계 비트 단위 같음", all(torch.equal(ws[0][k], ws[1][k]) for k in ws[0]))


# ================================================================ 그림 1: 시드별 best·last
def fig_seeds():
    s = Svg(700, 310, "VD-CNN을 시드 0~4로 학습한 시험 정확도. 왼쪽 묶음은 best 체크포인트, 오른쪽 묶음은 last 체크포인트. "
                      "각 묶음에서 배치 정규화 있음과 없음, 점은 시드 하나, 막대는 평균±표준편차")
    L, Rr, T, B = 78, 680, 40, 240
    lo, hi = 0.905, 0.98
    sy = lambda v: B - (B - T) * (v - lo) / (hi - lo)
    s.line(L, B, Rr, B, "s-mu", 1.2)
    s.line(L, T, L, B, "s-mu", 1.2)
    for v in (0.91, 0.93, 0.95, 0.97):
        s.line(L - 4, sy(v), L, sy(v), "s-mu", 1.2)
        s.line(L, sy(v), Rr, sy(v), "s-mu", 0.5, "2 4")
        s.text(L - 7, sy(v) + 4, f"{v:.2f}", "f-mu", 11, anchor="end")
    s.text(L, T - 14, "시험 정확도", "f-mu", 12, anchor="start")
    s.line((L + Rr) / 2 + 10, T - 4, (L + Rr) / 2 + 10, B, "s-mu", 1.0)
    cols = [("test_best", "bn", 190, "BN 있음", "f-ac", "s-ac"), ("test_best", "no_bn", 320, "BN 없음", "f-bd", "s-bd"),
            ("test_last", "bn", 480, "BN 있음", "f-ac", "s-ac"), ("test_last", "no_bn", 610, "BN 없음", "f-bd", "s-bd")]
    for key, cfg, c, name, fcls, scls in cols:
        vals = np.array([r[key] for r in S[cfg]["runs"]])
        for i, v in enumerate(vals):
            s.circle(c - 34 + 9 * i, sy(v), 4.2, fcls)
        m, sd = vals.mean(), vals.std(ddof=1)
        x = c + 20
        s.line(x, sy(m - sd), x, sy(m + sd), scls, 2.0)
        s.line(x - 5, sy(m - sd), x + 5, sy(m - sd), scls, 2.0)
        s.line(x - 5, sy(m + sd), x + 5, sy(m + sd), scls, 2.0)
        s.line(x - 9, sy(m), x + 9, sy(m), scls, 3.0)
        s.text(x + 13, sy(m) + 4, f"{m:.3f}", "f-fg", 11, anchor="start")
        s.text(c - 6, B + 18, name, "f-fg", 12)
    s.text(255, B + 42, "best (검증 손실 최소 에폭)", "f-mu", 12, weight="600")
    s.text(545, B + 42, "last (30에폭 끝)", "f-mu", 12, weight="600")
    s.save(os.path.join(OUT, "49-seeds.svg"))


# ================================================================ 그림 2: 체크포인트 도식
def fig_ckpt():
    s = Svg(700, 300, "체크포인트 도식. 왼쪽은 시드 0 VD-CNN의 에폭별 검증 손실과 best(27에폭)·last(30에폭) 저장 시점, "
                      "가운데는 두 파일과 쓰임, 오른쪽은 파일에 담는 것")
    vl = P5["base"]["hist"]["val_loss"]
    L, Rr, T, B = 56, 300, 46, 230
    sx = lambda e: L + (Rr - L) * (e - 1) / 29
    sy = lambda v: B - (B - T) * v / 0.45
    s.line(L, B, Rr, B, "s-mu", 1.2)
    s.line(L, T, L, B, "s-mu", 1.2)
    for e in (1, 10, 20, 30):
        s.line(sx(e), B, sx(e), B + 4, "s-mu", 1.2)
        s.text(sx(e), B + 17, str(e), "f-mu", 11)
    for v in (0, 0.2, 0.4):
        s.line(L - 4, sy(v), L, sy(v), "s-mu", 1.2)
        s.text(L - 7, sy(v) + 4, f"{v:.1f}", "f-mu", 11, anchor="end")
    s.text((L + Rr) / 2, B + 36, "에폭", "f-mu", 12)
    s.text(L, T - 14, "검증 손실 (시드 0)", "f-mu", 12, anchor="start")
    d = "M " + " L ".join(f"{sx(i + 1):.1f} {sy(v):.1f}" for i, v in enumerate(vl))
    s.path(d, "s-fg", 1.6)
    pb, pl = (sx(27), sy(vl[26])), (sx(30), sy(vl[29]))
    s.circle(*pb, 4.8, "f-ac")
    s.circle(*pl, 4.8, "f-bd")
    # 파일 상자
    fx, fw, fh = 350, 112, 50
    fl, fb = (fx, 62), (fx, 166)
    s.rect(fb[0], fb[1], fw, fh, "s-ac f-acs", 1.6, rx=4)
    s.text(fb[0] + fw / 2, fb[1] + 21, "best.pt (27)", "f-fg", 12, weight="600")
    s.text(fb[0] + fw / 2, fb[1] + 39, "평가·배포", "f-mu", 11)
    s.rect(fl[0], fl[1], fw, fh, "s-bd f-bds", 1.6, rx=4)
    s.text(fl[0] + fw / 2, fl[1] + 21, "last.pt (30)", "f-fg", 12, weight="600")
    s.text(fl[0] + fw / 2, fl[1] + 39, "학습 재개", "f-mu", 11)
    s.line(pb[0] + 5, pb[1] - 4, fb[0] - 2, fb[1] + fh / 2, "s-ac", 1.3, "4 3")
    s.line(pl[0] + 5, pl[1], fl[0] - 2, fl[1] + fh / 2, "s-bd", 1.3, "4 3")
    # 담는 것 목록
    bx, by, bw, bh = 498, 30, 190, 240
    s.rect(bx, by, bw, bh, "s-mu f-sf", 1.2, rx=4)
    s.text(bx + bw / 2, by + 22, "체크포인트에 담는 것", "f-fg", 12, weight="600")
    items = ["가중치(러닝 통계 포함)", "옵티마이저 상태", "스케줄러 상태", "에폭·반복 번호", "난수 생성기 상태",
             "설정·시드", "코드 커밋·자료 버전", "검증 지표"]
    for i, it in enumerate(items):
        s.text(bx + 14, by + 50 + i * 24, "· " + it, "f-fg", 11.5, anchor="start")
    for (x, y) in (fb, fl):
        s.line(x + fw + 2, y + fh / 2, bx - 2, y + fh / 2, "s-mu", 1.2)
    s.save(os.path.join(OUT, "49-ckpt.svg"))


if __name__ == "__main__":
    if "rerun" in sys.argv:
        rerun()
    else:
        check()
        fig_seeds()
        fig_ckpt()
