"""55강 그림과 검산.

본문 수치는 data/part8_runs.json의 calib(run_part8.exp_calib)에서 읽음.
    python3 55.py          # 검산 출력 + 그림 3개
    python3 55.py extra    # (약 20초, CPU 1스레드) 저장된 가중치로 추론만 다시 해서 본문의 보조 수치를 계산
                           #  - VD-CNN(best) 로짓으로 JSON의 ECE·MCE·T를 다시 계산해 같은지 확인
                           #  - ECE의 칸 수(10·15·20)와 같은 개수 칸(15칸)에 따른 차이
                           #  - 연무·계절 시험의 라벨로 맞췄다면 나왔을 온도(실무에서는 라벨이 없어 불가능, 비교용)
                           #  - 드롭아웃 모델을 추론 모드(드롭아웃 끔)로 한 번 돌렸을 때의 ECE (MC 평균과 비교)
그림: 55-reliability.svg(신뢰도 도표), 55-temperature.svg(온도 전후 평균 확신도·정확도), 55-uncertainty.svg(불확실성 분해·오류 탐지 AUROC)
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from svglib import Svg

OUT = os.path.join(HERE, "..", "fig")
DATA = os.path.join(HERE, "..", "data")
R = json.load(open(os.path.join(DATA, "part8_runs.json"), encoding="utf-8"))
C = R["calib"]


# ================================================================ 검산
def ece_from_bins(rows, n):
    e = sum(c / n * abs(a - f) for _, _, c, f, a in rows)
    m = max(abs(a - f) for _, _, c, f, a in rows)
    return e, m


def check():
    for split in ("test", "shift"):
        d = C[split]
        n = sum(r[2] for r in d["bins"])
        e, m = ece_from_bins(d["bins"], n)
        eT, mT = ece_from_bins(d["bins_T"], n)
        print(split, "N", n, "acc", d["acc"], "mean_conf", d["mean_conf"], "conf-acc", round(d["mean_conf"] - d["acc"], 4))
        print("   ECE(칸에서 재계산) %.4f JSON %.4f | MCE %.4f JSON %.4f" % (e, d["ece"], m, d["mce"]))
        print("   온도 뒤 ECE %.4f JSON %.4f | MCE %.4f JSON %.4f" % (eT, d["ece_T"], mT, d["mce_T"]))
        print("   NLL %.4f → %.4f" % (d["nll"], d["nll_T"]))
        gaps = sorted(((abs(a - f), c, lo, hi) for lo, hi, c, f, a in d["bins"]), reverse=True)[:3]
        print("   큰 칸 차이(차이, 개수, 구간)", [(round(g, 4), c, lo, hi) for g, c, lo, hi in gaps])
        gapsT = sorted(((abs(a - f), c, lo, hi) for lo, hi, c, f, a in d["bins_T"]), reverse=True)[:2]
        print("   온도 뒤 큰 칸", [(round(g, 4), c, lo, hi) for g, c, lo, hi in gapsT])
        top = d["bins"][-1]
        print("   맨 위 칸", top, "비중", round(top[2] / n, 4), "기여", round(top[2] / n * abs(top[4] - top[3]), 4))
        print("   0.9 이상 칸(0.9333~1) 정확도", top[4])
    print("T", C["temperature"], "1/T", round(1 / C["temperature"], 4))
    print("드롭아웃 모델", C["dropout_model"])
    U = C["mc_dropout"]
    for s in ("test", "shift"):
        u = U[s]
        print(s, u, "우연적 비중", round(u["aleatoric"] / u["total"], 4), "인지적 비중", round(u["epistemic"] / u["total"], 4),
              "틀림/맞음 인지적 비", round(u["epistemic_wrong"] / u["epistemic_correct"], 2))
    print("이동 대 시험 구분 AUROC(인지적)", U["auroc_shift_vs_test_by_epistemic"])
    print("ln 6 =", round(math.log(6), 4))
    for k in ("total", "aleatoric", "epistemic"):
        print("이동/시험 배수", k, round(U["shift"][k] / U["test"][k], 2))
    # 예: 로짓 차이 2짜리 두 클래스에서 T에 따른 확률
    for T in (0.837, 1.0, 1.5):
        p = 1 / (1 + math.exp(-2 / T))
        print("두 클래스 로짓 차 2, T=%.3f → %.3f" % (T, p))


def extra():
    import numpy as np
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, DATA)
    import part5_common as P5C
    from part5_common import TinyCNN
    from sklearn.metrics import roc_auc_score
    D = P5C.data()

    def logits(m, x):
        m.eval()
        with torch.no_grad():
            return torch.cat([m(x[i:i + 300]) for i in range(0, len(x), 300)])

    def ece(p, y, bins=15, mass=False):
        conf, pred = p.max(1); acc = (pred == y).float()
        if mass:
            order = torch.argsort(conf); chunks = torch.tensor_split(order, bins)
            groups = [c for c in chunks if len(c)]
        else:
            edges = torch.linspace(0, 1, bins + 1)
            groups = [torch.where((conf > edges[i]) & (conf <= edges[i + 1]))[0] for i in range(bins)]
            groups = [g for g in groups if len(g)]
        e = sum(len(g) / len(y) * abs(acc[g].mean().item() - conf[g].mean().item()) for g in groups)
        mc = max(abs(acc[g].mean().item() - conf[g].mean().item()) for g in groups)
        return round(e, 4), round(mc, 4)

    def fit_T(L, y):
        t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)
        def closure():
            opt.zero_grad(); loss = F.cross_entropy(L / t.exp(), y); loss.backward(); return loss
        opt.step(closure)
        return float(t.exp().detach())

    m = TinyCNN(); m.load_state_dict(torch.load(os.path.join(DATA, ".part5_base.pt"))); m.eval()
    Lv = logits(m, D["x_val"]); T = fit_T(Lv, D["y_val"])
    print("검증으로 맞춘 T %.4f (JSON %.4f)" % (T, C["temperature"]))
    pv = Lv.softmax(1)
    print("검증 정확도 %.4f 평균 확신도 %.4f" % ((pv.argmax(1) == D["y_val"]).float().mean(), pv.max(1).values.mean()))
    for split in ("test", "shift"):
        L = logits(m, D[f"x_{split}"]); y = D[f"y_{split}"]
        p = L.softmax(1); pT = (L / T).softmax(1)
        print(split, "ECE·MCE 15칸", ece(p, y), "JSON", C[split]["ece"], C[split]["mce"])
        print("   10칸", ece(p, y, 10), "20칸", ece(p, y, 20), "같은 개수 15칸", ece(p, y, 15, True))
        print("   온도 뒤 15칸", ece(pT, y), "같은 개수 15칸", ece(pT, y, 15, True), "평균 확신도 %.4f → %.4f" % (p.max(1).values.mean(), pT.max(1).values.mean()))
        wrong = (p.argmax(1) != y).numpy()
        print("   틀린 수", int(wrong.sum()), "틀린 것 평균 확신도 %.4f" % p.max(1).values[torch.tensor(wrong)].mean(),
              "0.9 넘게 확신한 틀림", int(((p.max(1).values > 0.9).numpy() & wrong).sum()))
        print("   오류 탐지 AUROC(최대 확률) 온도 전 %.4f 뒤 %.4f" % (roc_auc_score(wrong, -p.max(1).values.numpy()), roc_auc_score(wrong, -pT.max(1).values.numpy())))
        To = fit_T(L, y)
        print("   이 자료의 라벨로 맞춘 T %.4f → ECE %s, NLL %.4f" % (To, ece((L / To).softmax(1), y), F.cross_entropy(L / To, y)))
    md = TinyCNN(dropout=0.3); md.load_state_dict(torch.load(os.path.join(DATA, ".part8_dropout.pt"))); md.eval()
    for split in ("test", "shift"):
        L = logits(md, D[f"x_{split}"]); y = D[f"y_{split}"]
        p = L.softmax(1)
        print("드롭아웃 모델 추론 모드", split, "acc %.4f" % (p.argmax(1) == y).float().mean(), "ECE·MCE", ece(p, y),
              "평균 확신도 %.4f" % p.max(1).values.mean(), "| MC 평균 ECE", C["mc_dropout"][split]["ece_mc"])


# ================================================================ 공통
def axes(s, X0, X1, YT, YB, ticks, xlabel=None, ylabel=None, yticks=True):
    sx = lambda v: X0 + (X1 - X0) * v
    sy = lambda v: YB - (YB - YT) * v
    s.rect(X0, YT, X1 - X0, YB - YT, "s-mu f-bg", 1.2)
    for v in ticks:
        s.line(sx(v), YB, sx(v), YB + 4, "s-mu", 1.2)
        s.text(sx(v), YB + 17, f"{v:g}", "f-mu", 11)
        if yticks:
            s.line(X0 - 4, sy(v), X0, sy(v), "s-mu", 1.2)
            s.text(X0 - 7, sy(v) + 4, f"{v:g}", "f-mu", 11, anchor="end")
    if xlabel:
        s.text((X0 + X1) / 2, YB + 36, xlabel, "f-mu", 12)
    if ylabel:
        s.text(X0 - 38, (YT + YB) / 2, ylabel, "f-mu", 12)
        s.parts[-1] = s.parts[-1].replace("<text ", f'<text transform="rotate(-90 {X0 - 38} {(YT + YB) / 2:.1f})" ')
    return sx, sy


# ================================================================ 그림 1: 신뢰도 도표
def fig_reliability():
    s = Svg(700, 380, "VD-CNN의 신뢰도 도표. 왼쪽 같은 분포 시험, 오른쪽 연무·계절 시험. 막대는 확신도 칸별 실제 정확도, 대각선은 완벽한 보정")
    panels = [("test", "같은 분포 시험", 70), ("shift", "연무·계절 시험", 400)]
    W, YT, YB = 250, 66, 306
    for key, title, X0 in panels:
        X1 = X0 + W
        sx, sy = axes(s, X0, X1, YT, YB, (0, 0.5, 1), "확신도(최대 소프트맥스 확률)", "정확도" if key == "test" else None)
        d = C[key]
        s.text((X0 + X1) / 2, YT - 40, title, "f-fg", 13, weight="600")
        s.text((X0 + X1) / 2, YT - 22, "ECE %.3f · MCE %.3f" % (d["ece"], d["mce"]), "f-mu", 11)
        s.line(sx(0), sy(0), sx(1), sy(1), "s-mu", 1.3, "5 4")
        for lo, hi, n, conf, acc in d["bins"]:
            x0, x1 = sx(lo) + 1, sx(hi) - 1
            # 막대 = 실제 정확도
            s.rect(x0, sy(acc), x1 - x0, sy(0) - sy(acc), "s-ac f-acs", 1.1)
            # 칸 평균 확신도 = 주황 가로선
            s.line(x0, sy(conf), x1, sy(conf), "s-bd", 2.2)
            # 표본 수
            ty = min(sy(acc), sy(conf)) - 4
            s.text((x0 + x1) / 2, ty, str(n), "f-mu", 8.5)
    # 범례
    ly = 368
    s.rect(150, ly - 10, 16, 12, "s-ac f-acs", 1.1)
    s.text(172, ly, "칸의 실제 정확도", "f-mu", 11, anchor="start")
    s.line(300, ly - 4, 320, ly - 4, "s-bd", 2.2)
    s.text(326, ly, "칸의 평균 확신도", "f-mu", 11, anchor="start")
    s.text(460, ly, "숫자 = 칸의 타일 수", "f-mu", 11, anchor="start")
    s.save(os.path.join(OUT, "55-reliability.svg"))


# ================================================================ 그림 2: 온도 전후 평균 확신도 vs 정확도
TEMP_CONF = {"test": (0.9639, 0.9720), "shift": (0.9112, 0.9270)}   # extra()로 계산한 온도 전·후 평균 확신도


def fig_temperature():
    s = Svg(700, 300, "온도 스케일링 전후의 평균 확신도 − 정확도. 0보다 작으면 덜 확신, 크면 과신. 같은 분포 시험에서는 0에 가까워지고, 연무·계절 시험에서는 과신이 더 커짐")
    L, Rr = 250, 650
    lo, hi = -0.05, 0.25
    sx = lambda v: L + (Rr - L) * (v - lo) / (hi - lo)
    YT, YB = 44, 226
    s.rect(L, YT, Rr - L, YB - YT, "s-mu f-bg", 1.2)
    for v in (-0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25):
        s.line(sx(v), YB, sx(v), YB + 4, "s-mu", 1.2)
        s.text(sx(v), YB + 17, f"{v:+.2f}" if v else "0", "f-mu", 11)
    s.line(sx(0), YT, sx(0), YB, "s-fg", 1.4)
    s.text(sx(0) - 6, YT - 8, "← 덜 확신", "f-mu", 11, anchor="end")
    s.text(sx(0) + 6, YT - 8, "과신 →", "f-mu", 11, anchor="start")
    rows = [("test", "같은 분포 시험"), ("shift", "연무·계절 시험")]
    bh = 20
    for i, (key, name) in enumerate(rows):
        yc = YT + 46 + i * 90
        acc = C[key]["acc"]; c0, c1 = TEMP_CONF[key]
        s.text(L - 14, yc - 2, name, "f-fg", 13, anchor="end", weight="600")
        s.text(L - 14, yc + 16, "정확도 %.3f · ECE %.3f → %.3f" % (acc, C[key]["ece"], C[key]["ece_T"]), "f-mu", 11, anchor="end")
        for j, (c, cls, tcls, lab) in enumerate([(c0, "s-mu f-sf", "f-mu", "전"), (c1, "s-bd f-bds", "f-bd", "후")]):
            g = c - acc
            y = yc - bh - 1 + j * (bh + 2)
            x0, x1 = sorted((sx(0), sx(g)))
            s.rect(x0, y, max(x1 - x0, 1), bh, cls, 1.1)
            txt = "%s %+.3f (확신도 %.3f)" % (lab, g, c)
            if g >= 0.1:
                s.text(x1 - 8, y + 14, txt, tcls, 11, anchor="end")
            elif g >= 0:
                s.text(x1 + 6, y + 14, txt, tcls, 11, anchor="start")
            else:
                s.text(sx(0) + 6, y + 14, txt, tcls, 11, anchor="start")
    s.text((L + Rr) / 2, YB + 38, "평균 확신도 − 정확도", "f-mu", 12)
    ly = 290
    s.rect(250, ly - 10, 16, 12, "s-mu f-sf", 1.1); s.text(272, ly, "온도 스케일링 전", "f-mu", 11, anchor="start")
    s.rect(420, ly - 10, 16, 12, "s-bd f-bds", 1.1); s.text(442, ly, "T = %.2f로 나눈 뒤" % C["temperature"], "f-mu", 11, anchor="start")
    s.save(os.path.join(OUT, "55-temperature.svg"))


# ================================================================ 그림 3: 불확실성 분해와 오류 탐지 AUROC
def fig_uncertainty():
    U = C["mc_dropout"]
    s = Svg(700, 320, "MC 드롭아웃 30회로 나눈 평균 불확실성(왼쪽)과, 각 점수로 틀린 타일을 골라낼 때의 AUROC(오른쪽)")
    # 왼쪽: 누적 막대 (우연적 + 인지적)
    X0, X1, YT, YB = 80, 290, 40, 250
    hi = 0.4
    sy = lambda v: YB - (YB - YT) * v / hi
    s.rect(X0, YT, X1 - X0, YB - YT, "s-mu f-bg", 1.2)
    for v in (0, 0.1, 0.2, 0.3, 0.4):
        s.line(X0 - 4, sy(v), X0, sy(v), "s-mu", 1.2)
        s.text(X0 - 7, sy(v) + 4, f"{v:g}", "f-mu", 11, anchor="end")
    s.text(X0 - 40, (YT + YB) / 2, "평균 엔트로피(nat)", "f-mu", 12)
    s.parts[-1] = s.parts[-1].replace("<text ", f'<text transform="rotate(-90 {X0 - 40} {(YT + YB) / 2:.1f})" ')
    s.text((X0 + X1) / 2, YT - 14, "불확실성 분해", "f-fg", 13, weight="600")
    bw = 56
    for i, (key, name) in enumerate([("test", "같은 분포"), ("shift", "연무·계절")]):
        cx = X0 + 55 + i * 100
        a, e = U[key]["aleatoric"], U[key]["epistemic"]
        s.rect(cx - bw / 2, sy(a), bw, sy(0) - sy(a), "s-ac f-acs", 1.1)
        s.rect(cx - bw / 2, sy(a + e), bw, sy(a) - sy(a + e), "s-bd f-bds", 1.1)
        s.text(cx, sy(a) + 16, "%.3f" % a, "f-ac", 11)
        s.text(cx, sy(a + e) - 6, "+%.3f" % e, "f-bd", 11)
        s.text(cx, YB + 17, name, "f-fg", 12)
    # 오른쪽: AUROC 묶음 막대
    X0, X1 = 400, 660
    lo, hi2 = 0.5, 1.0
    sy2 = lambda v: YB - (YB - YT) * (v - lo) / (hi2 - lo)
    s.rect(X0, YT, X1 - X0, YB - YT, "s-mu f-bg", 1.2)
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s.line(X0 - 4, sy2(v), X0, sy2(v), "s-mu", 1.2)
        s.text(X0 - 7, sy2(v) + 4, f"{v:.1f}", "f-mu", 11, anchor="end")
    s.text((X0 + X1) / 2, YT - 14, "틀린 타일 골라내기 AUROC", "f-fg", 13, weight="600")
    keys = [("auroc_error_by_maxprob", "s-mu f-sf", "f-mu"), ("auroc_error_by_total", "s-ac f-acs", "f-ac"), ("auroc_error_by_epistemic", "s-bd f-bds", "f-bd")]
    gw = 26
    for i, (split, name) in enumerate([("test", "같은 분포"), ("shift", "연무·계절")]):
        cx = X0 + 68 + i * 126
        for j, (k, cls, tcls) in enumerate(keys):
            v = U[split][k]
            x = cx + (j - 1.5) * (gw + 4)
            s.rect(x, sy2(v), gw, sy2(lo) - sy2(v), cls, 1.1)
            s.text(x + gw / 2, sy2(v) - 5, "%.2f" % v, tcls, 10)
        s.text(cx - 2, YB + 17, name, "f-fg", 12)
    ly = 300
    items = [("s-ac f-acs", "우연적 / 총 엔트로피"), ("s-bd f-bds", "인지적"), ("s-mu f-sf", "1 − 최대 확률")]
    xs = [150, 330, 450]
    for (cls, lab), x in zip(items, xs):
        s.rect(x, ly - 10, 16, 12, cls, 1.1)
        s.text(x + 22, ly, lab, "f-mu", 11, anchor="start")
    s.save(os.path.join(OUT, "55-uncertainty.svg"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extra":
        extra()
    else:
        check()
        fig_reliability()
        fig_temperature()
        fig_uncertainty()
