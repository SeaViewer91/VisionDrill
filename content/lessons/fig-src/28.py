"""28강 그림과 본문 수치 확인: 옵티마이저별 학습 곡선, 층별 기울기 크기, 기울기 클리핑 유무

학습 결과는 data/part5_runs.json(run_part5.py의 opt·grad·clip 실험)에서 읽음. 새 학습은 하지 않음.
본문의 손계산(교차 엔트로피·MSE 기울기, 역전파 예, 에폭당 반복 수)은 아래 check()에서 다시 계산해 출력함.
python3 28.py --deep 을 주면 20층 망(학습 없이 첫 묶음 한 번)의 층별 기울기 축소 비율도 확인함(수십 초).
"""
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
R = json.load(open(os.path.join(HERE, "..", "data", "part5_runs.json"), encoding="utf-8"))


# ================================================================ 본문 수치 확인
def check():
    print("ln 6 =", round(math.log(6), 4))

    def ce_mse(p, t=0):
        p = np.array(p, float)
        y = np.zeros(6)
        y[t] = 1
        J = np.diag(p) - np.outer(p, p)          # 소프트맥스의 야코비안
        return (-math.log(p[t]), (p - y)[t],     # 교차 엔트로피 손실과 정답 로짓 기울기
                ((p - y) ** 2).sum(), (J @ (2 * (p - y)))[t])   # 제곱 오차 합과 정답 로짓 기울기
    for p in ([0.01, 0.95, 0.01, 0.01, 0.01, 0.01], [0.3, 0.4, 0.075, 0.075, 0.075, 0.075]):
        print("p_true", p[0], "CE %.3f dCE %.3f  SE %.3f dSE %.3f" % ce_mse(p))

    # 역전파 손계산: ŷ = w2·ReLU(w1·x), L = (ŷ − y)²
    x, w1, w2, y = 0.5, 2.0, 1.5, 1.0
    z = w1 * x
    h = max(z, 0.0)
    yh = w2 * h
    L = (yh - y) ** 2
    dy = 2 * (yh - y)
    dw2, dh = dy * h, dy * w2
    dz = dh * (1.0 if z > 0 else 0.0)
    dw1 = dz * x
    print("forward z h yhat L", z, h, yh, L, " backward dyhat dw2 dh dz dw1", dy, dw2, dh, dz, dw1)
    f = lambda a, b: (b * max(a * x, 0) - y) ** 2
    e = 1e-6
    print("  수치 미분", round((f(w1 + e, w2) - f(w1 - e, w2)) / 2 / e, 6), round((f(w1, w2 + e) - f(w1, w2 - e)) / 2 / e, 6))
    a, b = w1 - 0.1 * dw1, w2 - 0.1 * dw2
    print("  한 걸음(학습률 0.1) 뒤 w1 w2 yhat L", a, b, b * max(a * x, 0), round(f(a, b), 4))

    n, bs = 4200, 64
    print("에폭당 반복", math.ceil(n / bs), "마지막 배치", n - (math.ceil(n / bs) - 1) * bs, "30에폭 갱신", 30 * math.ceil(n / bs))
    print("AdamW 감쇠만으로 1,980걸음 뒤 배율", round((1 - 1e-3 * 0.05) ** 1980, 3))
    print("시그모이드 0.25^10, 0.25^19", 0.25 ** 10, 0.25 ** 19)
    print("기본 초기화 분산 1/(3·144) vs He 2/144, 비", 1 / 432, 2 / 144, (1 / 432) / (2 / 144))
    print("층당 축소 예상: ReLU 1/sqrt6 = %.3f, 시그모이드 sqrt(1/3)·0.25 = %.3f" % (1 / math.sqrt(6), math.sqrt(1 / 3) * 0.25))
    g = R["grad"]
    for k, v in g.items():
        print("grad", k, "입력쪽", v[0], "출력쪽", v[-1], "층당 비 %.2f" % ((v[-1] / v[0]) ** (1 / 19)), "범위", min(v), max(v))
    print("0.408^19 =", 0.408 ** 19, " relu 측정 비", v and g["relu"][0] / g["relu"][-1])

    for k, v in R["opt"].items():
        h = v["hist"]
        tl = np.array(h["train_loss"])
        va = np.array(h["val_acc"])
        print(k, "첫 에폭 %.3f 끝 %.3f 0.25 아래 첫 에폭 %d" % (tl[0], tl[-1], np.argmax(tl < 0.25) + 1),
              "검증정확도 10~30에폭 %.3f~%.3f" % (va[9:].min(), va[9:].max()),
              "last %.3f best %.3f(%d)" % (v["last"]["test_acc"], v["best"]["test_acc"], v["best_epoch"]))
    for k, v in R["clip"].items():
        h = v["hist"]
        print(k, "loss", h["train_loss"], "gn", h["grad_norm"], "last", v["last"]["test_acc"], "best", v["best"]["test_acc"], v["best_epoch"])


def deep_check():
    """20층 망에서 층을 하나 거칠 때 활성값 기울기가 줄어드는 비율 (학습 없음, 첫 묶음 256개 한 번)"""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.join(HERE, "..", "data"))
    from part5_common import data, DeepNet
    D = data()
    x, y = D["x_train"][:256], D["y_train"][:256]
    for act in ("sigmoid", "relu"):
        torch.manual_seed(0)
        m = DeepNet(20, False, act)
        hs = []

        def fwd(v):
            h = m.act(m.stem(v))
            h.retain_grad()
            hs.append(h)
            for c in m.convs:
                h = m.act(c(h))
                h.retain_grad()
                hs.append(h)
            return m.fc(h.mean(dim=(2, 3)))
        F.cross_entropy(fwd(x), y).backward()
        gg = [h.grad.norm().item() for h in hs]
        print(act, "가중치 분산 %.5f" % m.convs[5].weight.var().item(),
              "층당 활성값 기울기 축소 %.2f" % ((gg[-1] / gg[0]) ** (1 / 19)),
              "활성값 RMS(입력쪽→출력쪽)", ["%.3f" % h.detach().pow(2).mean().sqrt().item() for h in hs[::4]])


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


# ---------------------------------------------------------------- 그림 1: 옵티마이저별 학습 손실 + Adam의 학습·검증 정확도
def fig_optim():
    O = R["opt"]
    s = Svg(700, 300, "옵티마이저 여섯 가지의 에폭별 학습 손실과, Adam 학습률 0.001의 학습·검증 정확도")
    L1, R1, T1, B1 = 56, 380, 34, 238
    sx = lambda e: L1 + (R1 - L1) * (e - 1) / 29
    sy = lambda v: B1 - (B1 - T1) * v / 1.2
    axes(s, L1, R1, T1, B1, (1, 10, 20, 30), sx, (0, 0.4, 0.8, 1.2), sy, lambda v: f"{v:.1f}", "에폭", "학습 손실")
    SER = [("sgd_0.01", "s-bd", None, "f-bd", "SGD 0.01"),
           ("momentum_0.01", "s-ok", None, "f-ok", "모멘텀 0.01"),
           ("adam_1e-3", "s-ac", None, "f-ac", "Adam 0.001"),
           ("adamw_1e-3_wd0.05", "s-ac", "5 3", "f-ac", "AdamW 0.001"),
           ("adam_0.1", "s-mu", None, "f-mu", "Adam 0.1"),
           ("sgd_0.1", "s-fg", "2 3", "f-fg", "SGD 0.1")]
    for key, cls, dash, tcls, name in SER:
        tl = O[key]["hist"]["train_loss"]
        poly(s, [(sx(e + 1), sy(v)) for e, v in enumerate(tl)], cls, 1.8, dash)
    LX, LY = sx(10.5), sy(1.12)          # 10에폭 뒤 0.5 위쪽은 곡선이 지나지 않음
    for i, (key, cls, dash, tcls, name) in enumerate(SER):
        col, row = i % 2, i // 2
        xx, yy = LX + col * 112, LY + row * 19
        poly(s, [(xx, yy - 4), (xx + 22, yy - 4)], cls, 2.2, dash)
        s.text(xx + 27, yy, name, tcls, 11, anchor="start")

    # 오른쪽: Adam 0.001의 학습·검증 정확도
    L2, R2, T2, B2 = 452, 680, 34, 238
    h = O["adam_1e-3"]["hist"]
    sx2 = lambda e: L2 + (R2 - L2) * (e - 1) / 29
    sy2 = lambda v: B2 - (B2 - T2) * (v - 0.75) / 0.25
    axes(s, L2, R2, T2, B2, (1, 10, 20, 30), sx2, (0.75, 0.85, 0.95), sy2, lambda v: f"{v:.2f}", "에폭", "정확도 (Adam 0.001)")
    poly(s, [(sx2(e + 1), sy2(v)) for e, v in enumerate(h["train_acc"])], "s-fg", 1.8)
    poly(s, [(sx2(e + 1), sy2(v)) for e, v in enumerate(h["val_acc"])], "s-ac", 1.8, "4 3")
    s.text(sx2(21), sy2(0.985), "학습", "f-fg", 11)
    s.text(sx2(20), sy2(0.885), "검증", "f-ac", 11)
    s.save(os.path.join(OUT, "28-optim.svg"))


# ---------------------------------------------------------------- 그림 2: 층별 기울기 크기(로그 눈금)
def fig_grad():
    G = R["grad"]
    s = Svg(680, 300, "20층 합성곱 망의 첫 묶음에서 잰 층별 가중치 기울기 크기. 시그모이드, ReLU, 잔차 연결을 넣은 ReLU")
    L, Rr, T, B = 70, 640, 30, 236
    sx = lambda k: L + (Rr - L) * (k - 1) / 19
    sy = lambda v: T + (B - T) * (0 - math.log10(v)) / 18
    s.line(L, B, Rr, B, "s-mu", 1.2)
    s.line(L, T, L, B, "s-mu", 1.2)
    for k in (1, 5, 10, 15, 20):
        s.line(sx(k), B, sx(k), B + 4, "s-mu", 1.2)
        s.text(sx(k), B + 17, f"{k}", "f-mu", 11)
    for p in range(0, -19, -3):
        s.line(L - 4, sy(10.0 ** p), L, sy(10.0 ** p), "s-mu", 1.2)
        s.line(L, sy(10.0 ** p), Rr, sy(10.0 ** p), "s-mu", 0.5, "2 4")
        s.text(L - 7, sy(10.0 ** p) + 4, "1" if p == 0 else f"1e{p}", "f-mu", 11, anchor="end")
    s.text(L, B + 36, "← 입력 쪽", "f-mu", 12, anchor="start")
    s.text(Rr, B + 36, "출력 쪽 →", "f-mu", 12, anchor="end")
    s.text((L + Rr) / 2, B + 36, "층 번호", "f-mu", 12)
    s.text(L, T - 12, "가중치 기울기 크기 (로그 눈금)", "f-mu", 12, anchor="start")
    for key, cls, tcls, name, lx, ly in [
            ("sigmoid", "s-bd", "f-bd", "시그모이드", 6.0, 1e-16),
            ("relu", "s-ac", "f-ac", "ReLU", 3.0, 3e-11),
            ("relu_residual", "s-ok", "f-ok", "ReLU + 잔차 연결", 8.0, 1e-3)]:
        v = G[key]
        poly(s, [(sx(k + 1), sy(g)) for k, g in enumerate(v)], cls, 2.0)
        for k, g in enumerate(v):
            s.circle(sx(k + 1), sy(g), 2.6, tcls)
        s.text(sx(lx), sy(ly), name, tcls, 12, anchor="start", weight="600")
    s.save(os.path.join(OUT, "28-grad.svg"))


# ---------------------------------------------------------------- 그림 3: 클리핑 유무
def fig_clip():
    C = R["clip"]
    s = Svg(700, 290, "정규화 없는 VD-CNN을 모멘텀 SGD 학습률 0.1로 학습할 때 클리핑 유무에 따른 학습 손실과 기울기 크기")
    L1, R1, T1, B1 = 56, 330, 34, 228
    sx = lambda e: L1 + (R1 - L1) * (e - 1) / 14
    sy = lambda v: B1 - (B1 - T1) * v / 2.2
    axes(s, L1, R1, T1, B1, (1, 5, 10, 15), sx, (0, 0.5, 1.0, 1.5, 2.0), sy, lambda v: f"{v:.1f}", "에폭", "학습 손실")
    poly(s, [(sx(e + 1), sy(v)) for e, v in enumerate(C["noclip"]["hist"]["train_loss"])], "s-bd", 2.0)
    poly(s, [(sx(e + 1), sy(v)) for e, v in enumerate(C["clip1"]["hist"]["train_loss"])], "s-ac", 2.0, "5 3")
    s.text(sx(11), sy(1.95), "클리핑 없음", "f-bd", 12, weight="600")
    s.text(sx(10.5), sy(0.36), "최대 노름 1로 클리핑", "f-ac", 12, weight="600")

    L2, R2, T2, B2 = 420, 680, 34, 228
    sx2 = lambda e: L2 + (R2 - L2) * (e - 1) / 14
    sy2 = lambda v: B2 - (B2 - T2) * v / 12
    axes(s, L2, R2, T2, B2, (1, 5, 10, 15), sx2, (0, 4, 8, 12), sy2, lambda v: f"{v:.0f}", "에폭",
         "기울기 크기 (자르기 전, 에폭 평균)")
    s.line(L2, sy2(1), R2, sy2(1), "s-mu", 1.2, "2 3")
    s.text(L2 + 6, sy2(1) + 15, "최대 노름 1", "f-mu", 11, anchor="start")
    poly(s, [(sx2(e + 1), sy2(v)) for e, v in enumerate(C["noclip"]["hist"]["grad_norm"])], "s-bd", 2.0)
    poly(s, [(sx2(e + 1), sy2(v)) for e, v in enumerate(C["clip1"]["hist"]["grad_norm"])], "s-ac", 2.0, "5 3")
    pk = int(np.argmax(C["noclip"]["hist"]["grad_norm"]))
    pv = C["noclip"]["hist"]["grad_norm"][pk]
    s.text(sx2(pk + 1) + 8, sy2(pv) + 4, f"{pv:.1f}", "f-bd", 11, anchor="start")
    s.save(os.path.join(OUT, "28-clip.svg"))


if __name__ == "__main__":
    check()
    if "--deep" in sys.argv:
        deep_check()
    fig_optim()
    fig_grad()
    fig_clip()
