"""46강 그림과 본문 수치: 학습률 범위 테스트 곡선, 스케줄 모양, 스케줄별 검증 정확도

학습 결과는 data/part7_runs.json(run_part7.py의 lrfind·sched·warmup·bscale 실험)에서 읽음. 새 학습은 하지 않음.
스케줄 모양은 data/part7_common.py의 lr_at(실험에서 쓴 함수)으로 걸음마다 다시 계산함.
본문의 손계산(2차 함수에서 학습률에 따른 수렴·발산, 걸음 수, 걸음당 배율)은 check()에서 출력함.
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg
from part7_common import lr_at

OUT = os.path.join(HERE, "..", "fig")
R = json.load(open(os.path.join(DATA, "part7_runs.json"), encoding="utf-8"))
SPE, EP = 66, 30                     # 배치 64면 에폭당 66걸음, 30에폭
TOTAL = SPE * EP
SCHED = [  # 이름, 실험 키, lr_at 인자
    ("고정", "const_0.05", dict(base=0.05, sched="const")),
    ("계단", "step_0.05", dict(base=0.05, sched="step", step_at=(20 * SPE, 25 * SPE))),
    ("코사인", "cos_0.05", dict(base=0.05, sched="cos")),
    ("원사이클", "onecycle_max0.1", dict(base=0.05, sched="onecycle", max_lr=0.1)),
    ("워밍업+코사인", "warm2_cos_0.05", dict(base=0.05, sched="cos", warmup=2 * SPE)),
]


def curve(kw):
    return [lr_at(i, TOTAL, **kw) for i in range(TOTAL)]


# ================================================================ 본문 수치 확인
def check():
    # 1절: L = a θ²/2 에서 θ ← θ(1 − η a)
    a = 4.0
    for eta in (0.01, 0.1, 0.4, 0.6):
        f = 1 - eta * a
        print(f"η={eta}: 걸음당 배율 {f:+.2f}, 10걸음 {f ** 10:+.4g}, 50걸음 {f ** 50:+.4g}")
    print("발산 경계 2/a =", 2 / a, " 진동 시작 1/a =", 1 / a)

    lf = R["lrfind"]
    lrs, sm = np.array(lf["lrs"]), lf["smoothed_loss"]
    print("범위 테스트: 걸음 수", len(lrs), "걸음당 배율 %.4f" % ((10 / 1e-5) ** (1 / 199)),
          "10배에 걸리는 걸음 %.1f" % (1 / math.log10((10 / 1e-5) ** (1 / 199))))
    print("  시작 손실", sm[0], "가장 가파름", lf["lr_steepest"], "최소", lf["lr_at_min_loss"], lf["min_loss"],
          "최소/10 %.4f" % (lf["lr_at_min_loss"] / 10), "발산", lf["diverged_at"])
    L = np.array([v if v is not None else np.nan for v in sm])
    for q in (0.0059, 0.0235, 0.05, 0.1, 2.674, 3.783):
        i = int(np.argmin(abs(np.log(lrs) - math.log(q))))
        print(f"  lr≈{q}: 걸음 {i + 1}, 손실 {L[i]}")
    flat = (lrs > 0.04) & (lrs < 2.7)
    print("  0.04~2.7 구간 손실 범위 %.3f~%.3f" % (np.nanmin(L[flat]), np.nanmax(L[flat])))
    print("  관례 비교: 0.05 = 최소의 1/%.1f, 0.1 = 최소의 1/%.1f" % (0.2354 / 0.05, 0.2354 / 0.1))

    S = R["sched"]
    for name, key, kw in SCHED:
        v = S[key]
        va = v["hist"]["val_acc"]
        c = curve(kw)
        print(f"{name}: last {v['last']['test']} best {v['best']['test']}({v['best_epoch']}) "
              f"검증 21~30에폭 {min(va[20:]):.3f}~{max(va[20:]):.3f}, 26~30 {min(va[25:]):.3f}~{max(va[25:]):.3f} "
              f"| lr 시작 {c[0]:.4g} 최고 {max(c):.4g}(걸음 {int(np.argmax(c))}, 에폭 {np.argmax(c) / SPE:.1f}) 끝 {c[-1]:.3g}")
    same = S["const_0.05"]["hist"]["val_acc"][:20] == S["step_0.05"]["hist"]["val_acc"][:20]
    print("고정·계단 20에폭까지 검증 정확도 같음:", same)
    oc = S["onecycle_max0.1"]["hist"]["val_acc"]
    print("원사이클 1~10에폭 검증", oc[:10])

    W = R["warmup"]
    for k in ("nowarm", "warm3"):
        v = W[k]
        print("warmup", k, "last", v["last"], "best", v["best"], v["best_epoch"], "1에폭 학습 손실", v["hist"]["train_loss"][0],
              "검증 손실 최소", min(v["hist"]["val_loss"]))
    print("  0.007 × 900 =", 0.007 * 900)

    B = R["bscale"]
    for k in ("bs32_lr0.025", "bs256_lr0.025", "bs256_lr0.2", "bs256_lr0.2_warm2"):
        v = B[k]
        h = v["hist"]
        print("bscale", k, "걸음", v["steps"], "last", v["last"]["test"], "best", v["best"]["test"], v["best_epoch"],
              "검증 손실 1~3에폭", h["val_loss"][:3], "최대 %.2f" % max(h["val_loss"]))
    print("배치 32 에폭당 걸음", math.ceil(4200 / 32), "배치 256", math.ceil(4200 / 256), "선형 규칙 0.05×32/64 =", 0.05 * 32 / 64,
          "0.05×256/64 =", 0.05 * 256 / 64)

    # 코드 블록 확인: PyTorch 스케줄러 조합이 실험의 워밍업+코사인과 같은 모양인지
    import torch
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.05, momentum=0.9)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=132)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL - 132)
    sch = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[132])
    got = []
    for _ in range(TOTAL):
        got.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    ref = curve(SCHED[4][2])
    print("PyTorch 조합 vs 실험 lr: 최대 차이 %.4g, 걸음 0·66·132·1000·1979" % max(abs(x - y) for x, y in zip(got, ref)),
          [round(got[i], 5) for i in (0, 66, 132, 1000, 1979)], [round(ref[i], 5) for i in (0, 66, 132, 1000, 1979)])


# ================================================================ 그림 도구
def poly(s, pts, cls, w, dash=None):
    d = "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts)
    s.path(d, cls, w)
    if dash:
        s.parts[-1] = s.parts[-1].replace("/>", f' stroke-dasharray="{dash}"/>')


# ---------------------------------------------------------------- 그림 1: 범위 테스트 곡선
def fig_lrfind():
    lf = R["lrfind"]
    lrs, sm = lf["lrs"], lf["smoothed_loss"]
    s = Svg(700, 290, "학습률 범위 테스트. 학습률을 0.00001에서 10까지 지수로 키우며 기록한 이동 평균 손실과, 가장 가파른 곳·최소의 10분의 1·손실 최소·발산 지점")
    L, Rr, T, B = 64, 620, 30, 226
    Y0, Y1 = 0.3, 2.0
    sx = lambda v: L + (Rr - L) * (math.log10(v) + 5) / 6
    sy = lambda v: B - (B - T) * (v - Y0) / (Y1 - Y0)
    s.line(L, B, Rr, B, "s-mu", 1.2)
    s.line(L, T, L, B, "s-mu", 1.2)
    for p, lab in zip(range(-5, 2), ("1e-5", "1e-4", "0.001", "0.01", "0.1", "1", "10")):
        s.line(sx(10.0 ** p), B, sx(10.0 ** p), B + 4, "s-mu", 1.2)
        s.text(sx(10.0 ** p), B + 17, lab, "f-mu", 11)
    for v in (0.5, 1.0, 1.5, 2.0):
        s.line(L - 4, sy(v), L, sy(v), "s-mu", 1.2)
        s.text(L - 7, sy(v) + 4, f"{v:.1f}", "f-mu", 11, anchor="end")
    s.text((L + Rr) / 2, B + 36, "학습률 (로그 눈금)", "f-mu", 12)
    s.text(L, T - 12, "학습 손실 (이동 평균)", "f-mu", 12, anchor="start")
    # 곡선: 위 끝(2.0)을 넘는 곳에서 자름
    pts = []
    for i, (lr, v) in enumerate(zip(lrs, sm)):
        if v is None:
            break
        if v > Y1:
            lr0, v0 = lrs[i - 1], sm[i - 1]
            u = (Y1 - v0) / (v - v0)
            pts.append((sx(10 ** (math.log10(lr0) + u * (math.log10(lr) - math.log10(lr0)))), sy(Y1)))
            break
        pts.append((sx(lr), sy(v)))
    poly(s, pts, "s-ac", 2.0)
    # 표시점
    def at(q):
        i = int(np.argmin([abs(math.log(x) - math.log(q)) for x in lrs[:-1]]))
        return sx(lrs[i]), sy(sm[i])
    x1, y1 = at(lf["lr_steepest"])
    s.circle(x1, y1, 4.5, "f-bd")
    s.text(x1 - 9, y1 + 22, "가장 가파름 0.0059", "f-bd", 11, anchor="end", weight="600")
    x2, y2 = at(lf["lr_at_min_loss"] / 10)
    s.circle(x2, y2, 4.5, "f-ok")
    s.text(x2, y2 + 24, "최소의 1/10 0.024", "f-ok", 11, weight="600")
    x3, y3 = at(lf["lr_at_min_loss"])
    s.circle(x3, y3, 4.5, "f-fg")
    s.text(x3, y3 - 44, "손실 최소 0.235", "f-fg", 11, weight="600")
    s.line(x3, y3 - 38, x3, y3 - 8, "s-mu", 1.0)
    xd = sx(lf["diverged_at"])
    s.line(xd, T, xd, B, "s-bd", 1.2, "4 3")
    s.text(xd + 5, T + 14, "발산", "f-bd", 11, anchor="start", weight="600")
    s.text(xd + 5, T + 29, "5.35", "f-bd", 11, anchor="start", weight="600")
    s.save(os.path.join(OUT, "46-lrfind.svg"))


# ---------------------------------------------------------------- 그림 2: 스케줄 모양(작은 패널 5개)
def fig_shapes():
    s = Svg(700, 230, "학습률 스케줄 다섯 가지의 모양. 고정, 계단식 감쇠, 코사인 어닐링, 원사이클, 워밍업 뒤 코사인. 가로는 에폭 0~30, 세로는 학습률 0~0.1")
    L0, W, G, T, B = 58, 116, 10, 34, 178
    sy = lambda v: B - (B - T) * v / 0.1
    S = R["sched"]
    for k, (name, key, kw) in enumerate(SCHED):
        L = L0 + k * (W + G)
        Rr = L + W
        sx = lambda e: L + W * e / 30
        for v in (0.05, 0.1):
            s.line(L, sy(v), Rr, sy(v), "s-mu", 0.5, "2 4")
        s.line(L, B, Rr, B, "s-mu", 1.2)
        s.line(L, T, L, B, "s-mu", 1.2)
        for e, anc in ((0, "start"), (30, "end")):
            s.line(sx(e), B, sx(e), B + 4, "s-mu", 1.2)
            s.text(sx(e), B + 16, f"{e}", "f-mu", 10, anchor=anc)
        if k == 0:
            for v in (0, 0.05, 0.1):
                s.line(L - 4, sy(v), L, sy(v), "s-mu", 1.2)
                s.text(L - 7, sy(v) + 4, f"{v:g}", "f-mu", 11, anchor="end")
        c = curve(kw)
        poly(s, [(sx(i / SPE), sy(v)) for i, v in enumerate(c)][::2] + [(sx(30), sy(c[-1]))], "s-ac", 2.0)
        s.text(L + W / 2, T - 12, name, "f-fg", 12, weight="600")
        s.text(L + W / 2, B + 36, f"last {S[key]['last']['test']:.3f}", "f-bd" if key == "const_0.05" else "f-mu", 11,
               weight="600" if key == "const_0.05" else None)
    s.text(L0 - 50, B + 16, "에폭", "f-mu", 10, anchor="start")
    s.save(os.path.join(OUT, "46-shapes.svg"))


# ---------------------------------------------------------------- 그림 3: 검증 정확도(고정 vs 계단, 고정 vs 코사인)
def fig_val():
    S = R["sched"]
    s = Svg(700, 280, "에폭별 검증 정확도. 왼쪽은 고정과 계단식 감쇠, 오른쪽은 고정과 코사인 어닐링")
    T, B, Y0, Y1 = 34, 222, 0.70, 1.0
    sy = lambda v: B - (B - T) * (v - Y0) / (Y1 - Y0)
    const = S["const_0.05"]["hist"]["val_acc"]
    for k, (L, Rr) in enumerate(((56, 330), (410, 684))):
        sx = lambda e, L=L, Rr=Rr: L + (Rr - L) * (e - 1) / 29
        s.line(L, B, Rr, B, "s-mu", 1.2)
        s.line(L, T, L, B, "s-mu", 1.2)
        for e in (1, 10, 20, 30):
            s.line(sx(e), B, sx(e), B + 4, "s-mu", 1.2)
            s.text(sx(e), B + 17, f"{e}", "f-mu", 11)
        for v in (0.7, 0.8, 0.9, 1.0):
            s.line(L - 4, sy(v), L, sy(v), "s-mu", 1.2)
            s.text(L - 7, sy(v) + 4, f"{v:.1f}", "f-mu", 11, anchor="end")
        s.text((L + Rr) / 2, B + 36, "에폭", "f-mu", 12)
        s.text(L, T - 12, "검증 정확도", "f-mu", 12, anchor="start")
        poly(s, [(sx(e + 1), sy(v)) for e, v in enumerate(const)], "s-bd", 1.6)
        if k == 0:
            st = S["step_0.05"]["hist"]["val_acc"]
            for e in (20, 25):
                s.line(sx(e + 0.5), T + 4, sx(e + 0.5), B, "s-mu", 1.0, "2 3")
            s.text(sx(20.5) - 4, T + 14, "×0.1", "f-mu", 10, anchor="end")
            s.text(sx(25.5) + 4, T + 14, "×0.1", "f-mu", 10, anchor="start")
            poly(s, [(sx(e + 1), sy(v)) for e, v in enumerate(st)][19:], "s-ac", 2.2)
            s.text(sx(23), sy(0.985), "계단", "f-ac", 12, weight="600")
            s.text(sx(25), sy(0.868), "고정", "f-bd", 12, weight="600")
            s.text(sx(10.5), sy(0.74), "20에폭까지 둘이 같음", "f-mu", 11)
        else:
            co = S["cos_0.05"]["hist"]["val_acc"]
            poly(s, [(sx(e + 1), sy(v)) for e, v in enumerate(co)], "s-ok", 2.2, "5 2")
            s.text(sx(27.5), sy(0.99), "코사인", "f-ok", 12, weight="600")
            s.text(sx(25), sy(0.868), "고정", "f-bd", 12, weight="600")
    s.save(os.path.join(OUT, "46-val.svg"))


if __name__ == "__main__":
    check()
    fig_lrfind()
    fig_shapes()
    fig_val()
