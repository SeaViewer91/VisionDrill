"""60강 그림과 검산: 층별 SQNR(SVG), conv2 입력의 채널별 범위와 per-tensor 칸(SVG), 루프라인(SVG)

- 본문 수치 대부분은 data/part8_runs.json의 hw(계산: data/run_part8.py의 exp_hw·quant_run)에서 가져옴
- JSON에 없는 것은 여기서 계산해 출력함(약 30초, CPU 1스레드)
  · conv2 입력의 채널별 최댓값(보정 자료)과 per-tensor 축척, 보통 채널(중앙값)이 쓰는 칸 수 — 기본·이상치·SmoothQuant
  · 단일 층 양자화(그 층만 int8, 나머지 fp32)의 층 출력 SQNR과 정확도 — 이상치 모델(원문 06장의 단일 계층 민감도 방식)
  · 루프라인: 가상 장치에서 층별 이론 하한 시간 max(연산/최대 연산, 바이트/대역폭)
  · 오퍼레이터 융합 결손도(FDR): conv2 블록(합성곱→배치 정규화→ReLU)을 합치지 않을 때의 트래픽(int8, 1바이트로 셈)
    python3 60.py          # 검산 + 그림
    python3 60.py fig      # 그림만(모델 계산 없이 JSON과 아래 상수로)
"""
import copy, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "fig")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg

R = json.load(open(os.path.join(DATA, "part8_runs.json"), encoding="utf-8"))
H = R["hw"]
NAMES = ["conv1", "conv2", "conv3", "fc"]
CONDS = [("base", "기본"), ("outlier", "이상치 채널"), ("smoothquant", "SmoothQuant")]


# ================================================================ 검산(JSON)
def check_json():
    for k, nm in CONDS:
        v = H[k]
        print(nm, "SQNR", v["sqnr_db"], "PAR", v["par_max"], "채널 간 범위 비", v["channel_range_ratio"], "int8 정확도", v["acc_int8"],
              "fp32 로짓 차이", v.get("fp32_max_logit_diff"))
    print("fp32 정확도", H["base"]["acc_fp32"], "→ 맞힌 타일", round(H["base"]["acc_fp32"] * 900), "이상치 int8", round(H["outlier"]["acc_int8"] * 900))
    print("conv2 SQNR 하락", round(H["base"]["sqnr_db"]["conv2"] - H["outlier"]["sqnr_db"]["conv2"], 2), "dB")
    for k in ("conv2", "conv3", "fc"):
        d = H["base"]["sqnr_db"][k] - H["outlier"]["sqnr_db"][k]
        print(f"  {k}: 기본 {H['base']['sqnr_db'][k]:.1f} → 이상치 {H['outlier']['sqnr_db'][k]:.1f} (−{d:.1f} dB, 잡음 전력 {10 ** (d / 10):.0f}배)")
    rf = H["roofline"]; dev = rf["device"]
    peak, bw = dev["peak_int8_ops"], dev["bandwidth_Bps"]
    print("능선점", peak / bw, "연산/바이트")
    shapes = [(4, 16, 32), (16, 32, 16), (32, 64, 8)]
    for b in ("batch1", "batch64"):
        for row, (ci, co, hw) in zip(rf["layers"][b], shapes):
            nb = int(b[5:])
            ops = 2 * nb * co * hw * hw * ci * 9
            act = nb * (ci * hw * hw + co * hw * hw); wt = co * ci * 9
            t_c, t_m = ops / peak * 1e6, row["bytes"] / bw * 1e6
            att = min(peak, row["ai"] * bw)
            print(f"  {b} {row['layer']}: 연산 {ops:,} 바이트 {row['bytes']:,}(활성 {act:,} 가중치 {wt:,}) 산술 강도 {row['ai']:.1f} "
                  f"→ {'메모리' if row['ai'] < peak / bw else '연산'} 한계, 도달 성능 {att / 1e12:.2f} TOPS, 하한 시간 연산 {t_c:.2f} µs / 메모리 {t_m:.2f} µs")
    # 오퍼레이터 융합 결손도: conv2 블록, 배치 1, int8(1바이트)
    ci, co, hw = 16, 32, 16
    x_in, y, w = ci * hw * hw, co * hw * hw, co * ci * 9
    fused = x_in + w + y                       # 합성곱+BN+ReLU 한 커널: 입력·가중치 읽고 출력 한 번 씀
    unfused = (x_in + w + y) + (y + y) + (y + y)  # BN이 읽고 쓰고, ReLU가 읽고 씀
    print(f"FDR conv2 블록: 합침 {fused:,} B, 안 합침 {unfused:,} B → FDR = {(unfused - fused) / fused:.3f} (트래픽 {unfused / fused:.2f}배)")
    # 원문 예: 입력·출력 크기가 같고 가중치를 무시하면 합침 2T, 안 합침 6T → FDR = 2.0 = 200% 증가
    print("원문 예(입력=출력=T, 가중치 무시): 합침 2T, 안 합침 6T → FDR", (6 - 2) / 2, "= 트래픽 3배(200% 증가)")


# ================================================================ 검산(모델)
def check_model():
    import torch
    torch.set_num_threads(1)
    import run_part8 as R8
    D = R8.D
    xcal, xte, yte = D["x_train"][:500], D["x_test"], D["y_test"]
    m = R8.base_model()
    # exp_hw와 같은 방식으로 이상치·SmoothQuant 모델을 만듦
    jj = H["outlier_channel"]
    mo = copy.deepcopy(m)
    with torch.no_grad():
        mo.features[1].weight[jj] *= 30.0; mo.features[1].bias[jj] *= 30.0
        mo.features[4].weight[:, jj] /= 30.0

    def conv2_in(model, x):
        a = []
        h = model.features[4].register_forward_pre_hook(lambda mod, inp: a.append(inp[0].detach()))
        with torch.no_grad():
            model(x)
        h.remove()
        return torch.cat(a)

    ms = copy.deepcopy(mo)
    Xmax = conv2_in(ms, xcal).abs().amax(dim=(0, 2, 3)).clamp_min(1e-5)
    Wmax = ms.features[4].weight.abs().amax(dim=(0, 2, 3)).clamp_min(1e-5)
    sj = Xmax ** 0.5 / Wmax ** 0.5
    with torch.no_grad():
        ms.features[1].weight /= sj; ms.features[1].bias /= sj
        ms.features[4].weight *= sj[None, :, None, None]
    chans = {}
    for k, model in (("base", m), ("outlier", mo), ("smoothquant", ms)):
        a = conv2_in(model, xcal)
        cm = a.abs().amax(dim=(0, 2, 3))
        S, Z = R8.qparams_asym(a)
        med = float(cm.median())
        chans[k] = [round(float(v), 3) for v in cm]
        print(f"{k}: conv2 입력 최솟값 {float(a.min()):.3f}, 축척 S = {S:.4f}, 영점 {Z}, 채널 최댓값 중앙값 {med:.3f} → 보통 채널이 쓰는 칸 {med / S:.1f}개, "
              f"가장 넓은 채널 {float(cm.max()):.3f}({int(cm.argmax())}번)")
    print("채널별 최댓값(그림용)", json.dumps(chans))
    print("SmoothQuant 뒤 conv2 가중치(출력 채널별) 범위 비 max/median:",
          float((lambda wm: wm.max() / wm.median())(ms.features[4].weight.detach().abs().amax(dim=(1, 2, 3)))))
    # 단일 층 양자화: 이상치 모델에서 층 하나만 int8
    Ls32 = R8.layers_of(mo)
    ins = {}
    hs = [L.register_forward_pre_hook(lambda mod, inp, i=i: ins.setdefault(i, []).append(inp[0].detach())) for i, L in enumerate(Ls32)]
    with torch.no_grad():
        mo(xcal)
    for h in hs:
        h.remove()
    qp = {i: R8.qparams_asym(torch.cat(v)) for i, v in ins.items()}

    def outs(model, x):
        st = {}
        hs = [L.register_forward_hook(lambda mod, inp, o, i=i: st.__setitem__(i, o.detach())) for i, L in enumerate(R8.layers_of(model))]
        with torch.no_grad():
            model(x)
        for h in hs:
            h.remove()
        return st
    o32 = outs(mo, xte)
    for k in range(4):
        mk = copy.deepcopy(mo); L = R8.layers_of(mk)[k]
        L.weight.data = R8.fq_w_perchannel(L.weight.data)
        S, Z = qp[k]
        L.register_forward_pre_hook(lambda mod, inp, S=S, Z=Z: (R8.fq_asym(inp[0], S, Z),))
        ok = outs(mk, xte)
        acc = R8.evaluate(mk, xte, yte)[1]
        print(f"이상치 모델, {NAMES[k]}만 int8: 그 층 출력 SQNR {R8.sqnr(o32[k], ok[k]):.1f} dB, 정확도 {acc:.4f}")


# ================================================================ 그림
# conv2 입력 채널별 최댓값(보정 자료 500장) — check_model() 출력에서 옮김
CH = {"base": [1.628, 4.655, 8.366, 2.293, 1.529, 7.458, 9.677, 1.722, 1.347, 5.79, 8.139, 5.628, 3.349, 8.959, 1.555, 2.169],
      "outlier": [1.628, 139.64, 8.366, 2.293, 1.529, 7.458, 9.677, 1.722, 1.347, 5.79, 8.139, 5.628, 3.349, 8.959, 1.555, 2.169]}


def fig_sqnr():
    W, Hh = 660, 310
    s = Svg(W, Hh, "층별 SQNR: 기본, 이상치 채널, SmoothQuant")
    x0, x1, y0, y1 = 60, 620, 40, 250          # 그림 영역(y0 위, y1 아래)
    vmax = 50

    def Y(v):
        return y1 - (y1 - y0) * v / vmax
    for v in range(0, 51, 10):
        s.line(x0, Y(v), x1, Y(v), "s-mu", 0.5)
        s.text(x0 - 8, Y(v) + 4, str(v), "f-mu", 11, "end")
    s.text(18, (y0 + y1) / 2 - 10, "SQNR", "f-mu", 11, "middle")
    s.text(18, (y0 + y1) / 2 + 6, "(dB)", "f-mu", 11, "middle")
    # 기준 범위 15~25 dB(원문 간 차이, 잠정치)
    s.rect(x0, Y(25), x1 - x0, Y(15) - Y(25), "f-bds", 0)
    s.line(x0, Y(25), x1, Y(25), "s-bd", 1, "4 3")
    s.line(x0, Y(15), x1, Y(15), "s-bd", 1, "4 3")
    s.text(x1 + 6, Y(25) + 4, "25", "f-bd", 11, "start")
    s.text(x1 + 6, Y(15) + 4, "15", "f-bd", 11, "start")
    gw = (x1 - x0) / 4
    bw = 34
    cls = ["s-ac f-acs", "s-bd f-bd", "s-ok f-sf"]
    for i, nm in enumerate(NAMES):
        cx = x0 + gw * (i + 0.5)
        for j, (k, _) in enumerate(CONDS):
            v = H[k]["sqnr_db"][nm]
            bx = cx + (j - 1) * (bw + 6) - bw / 2
            s.rect(bx, Y(v), bw, y1 - Y(v), cls[j], 1.2)
            s.text(bx + bw / 2, Y(v) - 4, f"{v:.1f}", "f-fg", 10)
        s.text(cx, y1 + 18, nm, "f-fg", 12)
    s.line(x0, y1, x1, y1, "s-fg", 1)
    # 범례
    lx = 120
    for j, (_, lab) in enumerate(CONDS):
        bx = lx + j * 160
        s.rect(bx, 284, 14, 12, cls[j], 1.2)
        s.text(bx + 20, 294, lab, "f-fg", 12, "start")
    s.save(os.path.join(OUT, "60-sqnr.svg"))


def fig_channels(chans):
    """conv2 입력 16채널의 최댓값과 per-tensor int8 칸(255칸)의 굵기"""
    W, Hh = 660, 260
    s = Svg(W, Hh, "conv2 입력 채널별 최댓값: 기본과 이상치 채널")
    panels = [("base", "기본"), ("outlier", "이상치 채널(1번 ×30)")]
    pw = 270
    for p, (k, lab) in enumerate(panels):
        cm = chans[k]
        x0 = 50 + p * (pw + 50); x1 = x0 + pw; y0, y1 = 36, 200
        vmax = max(cm) * 1.08
        s.text((x0 + x1) / 2, 22, lab, "f-fg", 13)
        s.line(x0, y1, x1, y1, "s-fg", 1)
        s.line(x0, y0, x0, y1, "s-fg", 1)
        s.text(x0 - 6, y1 + 4, "0", "f-mu", 10, "end")
        n = len(cm); bw = pw / n
        for i, v in enumerate(cm):
            h = (y1 - y0) * v / vmax
            c = "s-bd f-bd" if (k == "outlier" and i == H["outlier_channel"]) else "s-ac f-acs"
            s.rect(x0 + i * bw + 2, y1 - h, bw - 4, h, c, 1)
        med = sorted(cm)[(len(cm) - 1) // 2]   # torch.median과 같은 아래쪽 중앙값
        S = max(cm) / 255
        im = int(np.argmax(cm))
        s.text(x0 + im * bw + bw / 2, y1 - (y1 - y0) * cm[im] / vmax - 6, f"{cm[im]:.1f}", "f-fg", 11)
        s.text(x0 + pw / 2, y1 + 18, "채널 0~15", "f-mu", 11)
        s.text(x0 + pw / 2, y1 + 38, f"보통 채널이 쓰는 칸 ≈ {med / S:.0f} / 255", "f-fg", 12)
    s.save(os.path.join(OUT, "60-channels.svg"))


def fig_roofline():
    rf = H["roofline"]; dev = rf["device"]
    peak, bw = dev["peak_int8_ops"], dev["bandwidth_Bps"]
    ridge = peak / bw
    W, Hh = 640, 360
    s = Svg(W, Hh, "가상 장치의 루프라인과 층별 산술 강도")
    x0, x1, y0, y1 = 80, 610, 30, 290
    lx0, lx1 = 1, 3                          # 산술 강도 10~1000 (로그)
    ly0, ly1 = -1, 1                          # 성능 0.1~10 TOPS (로그)

    def X(ai):
        return x0 + (x1 - x0) * (np.log10(ai) - lx0) / (lx1 - lx0)

    def Y(tops):
        return y1 - (y1 - y0) * (np.log10(tops) - ly0) / (ly1 - ly0)
    for e in (1, 2, 3):
        s.line(X(10 ** e), y0, X(10 ** e), y1, "s-mu", 0.5)
        s.text(X(10 ** e), y1 + 16, f"{10 ** e:,}", "f-mu", 11)
    for e, lab in ((-1, "0.1"), (0, "1"), (1, "10")):
        s.line(x0, Y(10 ** e), x1, Y(10 ** e), "s-mu", 0.5)
        s.text(x0 - 8, Y(10 ** e) + 4, lab, "f-mu", 11, "end")
    s.line(x0, y1, x1, y1, "s-fg", 1); s.line(x0, y0, x0, y1, "s-fg", 1)
    s.text((x0 + x1) / 2, y1 + 36, "산술 강도 (연산/바이트, 로그)", "f-mu", 12)
    s.text(20, (y0 + y1) / 2 - 8, "TOPS", "f-mu", 11)
    s.text(20, (y0 + y1) / 2 + 8, "(로그)", "f-mu", 11)
    # 지붕선
    a0 = 10.0
    s.path(f"M{X(a0):.1f},{Y(a0 * bw / 1e12):.1f} L{X(ridge):.1f},{Y(peak / 1e12):.1f} L{X(1000):.1f},{Y(peak / 1e12):.1f}", "s-fg", 2)
    s.line(X(ridge), Y(peak / 1e12), X(ridge), y1, "s-mu", 1, "4 3")
    s.text(X(ridge) + 6, y1 - 8, f"능선점 {ridge:.0f}", "f-mu", 11, "start")
    s.text(x1 - 4, Y(peak / 1e12) + 20, "최대 연산 4 TOPS", "f-fg", 11, "end")
    s.text(X(25), Y(25 * bw / 1e12) + 22, "대역폭 25.6 GB/s", "f-fg", 11, "start")   # 경사선 아래(오른쪽으로 갈수록 선이 위로 멀어짐)
    # 층 점: 배치 1(속 빈 원), 배치 64(채운 원), 화살표
    b1 = {r["layer"]: r["ai"] for r in rf["layers"]["batch1"]}
    b64 = {r["layer"]: r["ai"] for r in rf["layers"]["batch64"]}
    perf = lambda ai: min(peak, ai * bw) / 1e12
    for nm in ("conv1", "conv2", "conv3"):
        a, b = b1[nm], b64[nm]
        if b / a > 1.1:
            s.line(X(a) + 6, Y(perf(a)), X(b) - 6, Y(perf(b)), "s-mu", 1)
        s.add(f'<circle cx="{X(a):.1f}" cy="{Y(perf(a)):.1f}" r="5" class="s-bd f-bg" stroke-width="1.8"/>')
        s.circle(X(b), Y(perf(b)), 5, "f-ac")
    # 이름표: 배치 1은 점의 왼쪽 위(지붕선 위 빈 곳), 배치 64는 점 위
    for nm in ("conv1", "conv2", "conv3"):
        a = b1[nm]
        s.text(X(a) - 9, Y(perf(a)) - 9, nm, "f-fg", 11, "end")
    for nm in ("conv2", "conv3"):
        b = b64[nm]
        s.text(X(b), Y(perf(b)) - 12, nm, "f-fg", 11)
    # 범례
    s.add(f'<circle cx="{x0 + 30}" cy="{y0 + 18}" r="5" class="s-bd f-bg" stroke-width="1.8"/>')
    s.text(x0 + 42, y0 + 22, "배치 1", "f-fg", 12, "start")
    s.circle(x0 + 120, y0 + 18, 5, "f-ac")
    s.text(x0 + 132, y0 + 22, "배치 64", "f-fg", 12, "start")
    s.save(os.path.join(OUT, "60-roofline.svg"))


if __name__ == "__main__":
    check_json()
    if "fig" not in sys.argv:
        check_model()
    fig_sqnr()
    fig_channels(CH)
    fig_roofline()
