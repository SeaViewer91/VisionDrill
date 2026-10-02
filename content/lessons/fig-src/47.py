"""47강 그림과 본문 수치 검산

그림: 학습 메모리 구성 막대(ResNet-50, 512×512), fp32·fp16·bf16 비트 배치 도식, 기울기 누적 도식 (모두 SVG)
수치는 data/part7_runs.json(run_part7.py의 memory·accum·bf16·loader 실험)에서 읽음. 새 학습은 하지 않음.
검산: 메모리 어림(가중치·기울기·Adam 상태·활성값, 배치·입력 크기별), 유효 배치, 에폭당 갱신 수,
      수 형식의 범위(비트 배치에서 직접 계산), 캐스팅·손실 스케일링 예, 데이터로더 시간 비
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from svglib import Svg

OUT = os.path.join(HERE, "..", "fig")
R = json.load(open(os.path.join(HERE, "..", "data", "part7_runs.json"), encoding="utf-8"))
MiB, GiB = 2 ** 20, 2 ** 30


# ================================================================ 검산
def check():
    M = R["memory"]
    r5, r2 = M["resnet50_512"], M["resnet50_224"]
    p = r5["params"]
    print("ResNet-50 파라미터", p, "가중치 %.1f MiB, 기울기 %.1f, Adam %.1f (파라미터당 4·4·8바이트)" % (p * 4 / MiB, p * 4 / MiB, p * 8 / MiB))
    static = p * 16 / MiB
    print("고정 몫 합 %.1f MiB = %.2f GiB" % (static, static / 1024))
    a = r5["act_per_image_MB_fp32"]
    for b in (1, 4, 8, 16):
        print("512 배치 %2d: 활성값 %.2f GiB, 합 %.2f GiB, 절반 정밀도 활성값 %.2f → 합 %.2f"
              % (b, a * b / 1024, (a * b + static) / 1024, a * b / 2048, (a * b / 2 + static) / 1024))
    print("활성값 / 고정 몫 (배치 16) %.1f배, 배치 1 %.2f배" % (a * 16 / static, a / static))
    print("224 대비 512 활성값 비 %.3f, 화소 비 %.3f" % (a / r2["act_per_image_MB_fp32"], (512 / 224) ** 2))
    print("입력 384로 줄이면 화소 비 %.4f → 배치 16 활성값 %.2f GiB" % ((384 / 512) ** 2, a * 16 * (384 / 512) ** 2 / 1024))
    print("VD-CNN", M["vdcnn_32"])
    print("유효 배치 16×4×2 =", 16 * 4 * 2)

    A = R["accum"]
    print("기울기 차이 BN", A["grad_diff_bn"], "GN", A["grad_diff_gn"])
    print("에폭당 갱신: 배치 64", math.ceil(4200 / 64), " 16×4", math.ceil(4200 / (16 * 4)))
    for k in ("bn_bs64", "bn_bs16x4", "gn_bs64", "gn_bs16x4"):
        h = A[k]["hist"]
        vl = h["val_loss"]
        print(k, "학습손실10 %.4f" % h["train_loss"][-1], "검증손실 최소 %.4f (%d)" % (min(vl), int(np.argmax(-np.array(vl))) + 1),
              "best", A[k]["best"]["test"], "last", A[k]["last"]["test"], "초", A[k]["sec_total"])
    g64, g16 = A["gn_bs64"]["hist"]["train_loss"], A["gn_bs16x4"]["hist"]["train_loss"]
    print("GN 학습 손실 최대 차이 %.4f" % max(abs(x - y) for x, y in zip(g64, g16)))
    print("배치 64 기록이 bf16 실험 fp32 앞 10에폭과 같은지", A["bn_bs64"]["hist"]["train_loss"] == R["bf16"]["fp32"]["hist"]["train_loss"][:10])

    B = R["bf16"]
    for k in ("fp32", "bf16"):
        print(k, "best", B[k]["best"], B[k]["best_epoch"], "last", B[k]["last"], "학습손실30", B[k]["hist"]["train_loss"][-1], "초", B[k]["sec_total"])
    d = max(abs(x - y) for x, y in zip(B["fp32"]["hist"]["train_loss"], B["bf16"]["hist"]["train_loss"]))
    print("학습 손실 에폭별 최대 차이 %.4f" % d)
    print("시간 비 %.3f" % (B["bf16"]["sec_total"] / B["fp32"]["sec_total"]))
    # 비트 배치에서 직접 계산: (지수 비트, 가수 비트)
    for name, e, m in (("fp32", 8, 23), ("fp16", 5, 10), ("bf16", 8, 7)):
        bias = 2 ** (e - 1) - 1
        mx = (2 - 2 ** -m) * 2.0 ** bias
        print("%s 최대 %.6g 최소정규 %.4g 최소비정규 %.4g eps %.4g" % (name, mx, 2.0 ** (1 - bias), 2.0 ** (1 - bias - m), 2.0 ** -m))
    print("JSON finfo", B["finfo"], "fp16 최소 비정규", B["fp16_min_subnormal"])
    print("캐스팅 예", B["cast_examples"])
    print("1e-8×1024 = %.4g (fp16 최소 정규 %.3g 보다 작음 → 비정규수)" % (1e-8 * 1024, 2 ** -14))
    print("1e-8×65536 = %.4g" % (1e-8 * 65536))
    print("bf16 70000 → 70144 상대 오차 %.4f, bf16 eps/2 = %.4f" % ((70144 - 70000) / 70000, 2 ** -8))

    L = R["loader"]
    print("로더", L, "0→2 작업자 %.2f배" % (L["workers0"] / L["workers2"]))


# ================================================================ 그림 도구
def arrow(s, x1, y1, x2, y2, cls="s-mu", w=1.4):
    s.line(x1, y1, x2, y2, cls, w)
    ang = math.atan2(y2 - y1, x2 - x1)
    a1, a2 = ang + math.radians(150), ang - math.radians(150)
    d = f"M {x2 + 7 * math.cos(a1):.1f} {y2 + 7 * math.sin(a1):.1f} L {x2:.1f} {y2:.1f} L {x2 + 7 * math.cos(a2):.1f} {y2 + 7 * math.sin(a2):.1f}"
    s.path(d, cls, w)


def xaxis(s, L, Rr, y, ticks, sx, label):
    s.line(L, y, Rr, y, "s-mu", 1.2)
    for v in ticks:
        s.line(sx(v), y, sx(v), y + 4, "s-mu", 1.2)
        s.text(sx(v), y + 17, f"{v:g}", "f-mu", 11)
    s.text((L + Rr) / 2, y + 35, label, "f-mu", 12)


# ---------------------------------------------------------------- 그림 1: 학습 메모리 구성
def fig_memory():
    M = R["memory"]["resnet50_512"]
    s = Svg(720, 270, "ResNet-50, 512×512 입력의 학습 메모리 어림. 왼쪽은 영상 한 장일 때 구성 요소별 크기(MB), "
                      "오른쪽은 배치 크기별 합계(GB)로 가중치·기울기·Adam 상태와 활성값을 쌓은 막대")
    top, bh, gap = 46, 26, 14
    # 왼쪽: 구성 요소(MB)
    L, Rr = 120, 320
    sx = lambda v: L + (Rr - L) * v / 700
    s.text(L - 100, 22, "영상 한 장 (MB)", "f-fg", 13, anchor="start", weight="600")
    rows = [("가중치", M["weights_MB"], "s-ac f-acs"), ("기울기", M["grads_MB"], "s-ac f-acs"),
            ("Adam 상태", M["adam_states_MB"], "s-ac f-acs"), ("활성값", M["act_per_image_MB_fp32"], "s-bd f-bds")]
    for i, (name, v, cls) in enumerate(rows):
        y = top + i * (bh + gap)
        s.rect(L, y, sx(v) - L, bh, cls, 1.2)
        s.text(L - 8, y + bh / 2 + 5, name, "f-fg", 12, anchor="end")
        s.text(sx(v) + 6, y + bh / 2 + 5, f"{v:g}", "f-fg", 12, anchor="start", weight="600")
    yb = top + 4 * (bh + gap) - gap + 8
    s.line(L, top - 8, L, yb, "s-mu", 1.2)
    xaxis(s, L, Rr, yb, (0, 200, 400, 600), sx, "MB")
    # 오른쪽: 배치별 합계(GB)
    L2, R2 = 470, 660
    sx2 = lambda v: L2 + (R2 - L2) * v / 12
    s.text(L2 - 90, 22, "배치별 합계 (GB)", "f-fg", 13, anchor="start", weight="600")
    static = M["params"] * 16 / GiB
    a = M["act_per_image_MB_fp32"] / 1024
    rows2 = [("배치 1", a), ("배치 4", 4 * a), ("배치 16", 16 * a), ("16, 절반", 8 * a)]
    for i, (name, act) in enumerate(rows2):
        y = top + i * (bh + gap)
        s.rect(L2, y, sx2(static) - L2, bh, "s-ac f-acs", 1.2)
        s.rect(sx2(static), y, sx2(static + act) - sx2(static), bh, "s-bd f-bds", 1.2)
        s.text(L2 - 8, y + bh / 2 + 5, name, "f-fg", 12, anchor="end")
        s.text(sx2(static + act) + 6, y + bh / 2 + 5, f"{static + act:.1f}", "f-fg", 12, anchor="start", weight="600")
    s.line(L2, top - 8, L2, yb, "s-mu", 1.2)
    xaxis(s, L2, R2, yb, (0, 4, 8, 12), sx2, "GB")
    s.save(os.path.join(OUT, "47-memory.svg"))


# ---------------------------------------------------------------- 그림 2: 수 형식의 비트 배치
def fig_float():
    s = Svg(700, 250, "fp32, fp16, bf16의 비트 배치. 칸 하나가 1비트이고 부호·지수·가수 비트 수와 범위·정밀도를 함께 적음")
    cw, x0, bh = 16, 96, 26
    # 범례
    leg = [("부호", "s-bd f-bds"), ("지수", "s-ac f-acs"), ("가수", "s-mu f-sf")]
    for i, (name, cls) in enumerate(leg):
        lx = x0 + i * 90
        s.rect(lx, 14, 14, 14, cls, 1.2)
        s.text(lx + 20, 26, name, "f-fg", 12, anchor="start")
    rows = [("fp32", 8, 23, "최대 3.4e38 · 정밀도(eps) 1.2e-7"),
            ("fp16", 5, 10, "최대 65,504 · 최소 정규 6.1e-5 · eps 9.8e-4"),
            ("bf16", 8, 7, "최대 3.4e38 · eps 7.8e-3")]
    y0, step = 50, 66
    for r, (name, e, m, info) in enumerate(rows):
        y = y0 + r * step
        s.text(x0 - 12, y + bh / 2 + 5, name, "f-fg", 13, anchor="end", weight="600")
        segs = [(1, "s-bd f-bds", ""), (e, "s-ac f-acs", str(e)), (m, "s-mu f-sf", str(m))]
        x = x0
        for n, cls, lab in segs:
            for k in range(n):
                s.rect(x + k * cw, y, cw, bh, cls, 0.8)
            s.rect(x, y, n * cw, bh, cls.split()[0], 1.8)
            s.parts[-1] = s.parts[-1].replace("/>", ' fill="none"/>')
            if lab:
                s.text(x + n * cw / 2, y + bh / 2 + 5, lab, "f-fg", 12, weight="600")
            x += n * cw
        s.text(x0, y + bh + 18, info, "f-mu", 11.5, anchor="start")
    s.save(os.path.join(OUT, "47-float.svg"))


# ---------------------------------------------------------------- 그림 3: 기울기 누적
def fig_accum():
    s = Svg(700, 228, "위는 배치 64를 한 번에 넣어 기울기를 구하고 한 번 갱신, 아래는 16장씩 네 번 역전파해 기울기를 더한 뒤 한 번 갱신")
    bh = 36
    # 위: 큰 배치
    yt = 40
    s.text(20, yt + bh / 2 + 5, "배치 64", "f-fg", 13, anchor="start", weight="600")
    s.rect(110, yt, 316, bh, "s-ac f-acs", 1.4)
    s.text(268, yt + bh / 2 + 5, "64장 → 손실 → 역전파", "f-fg", 12)
    s.text(268, yt + bh + 18, "배치 정규화 통계: 64장으로", "f-bd", 11.5)
    # 아래: 누적
    yb = 140
    s.text(20, yb + bh / 2 + 5, "16 × 4", "f-fg", 13, anchor="start", weight="600")
    for i in range(4):
        x = 110 + i * 82
        s.rect(x, yb, 70, bh, "s-ac f-acs", 1.4)
        s.text(x + 35, yb + bh / 2 + 5, f"16장 ({i + 1})", "f-fg", 12)
    s.text(268, yb + bh + 18, "배치 정규화 통계: 16장씩 따로", "f-bd", 11.5)
    s.text(268, yb + bh + 36, "손실은 4로 나눠 역전파", "f-mu", 11.5)
    # 오른쪽: .grad와 갱신
    gx, gw = 486, 84
    ux, uw = 606, 78
    for y, lab in ((yt, ".grad"), (yb, ".grad에 더함")):
        s.rect(gx, y, gw, bh, "s-mu f-sf", 1.4)
        s.text(gx + gw / 2, y + bh / 2 + 5, lab, "f-fg", 12)
        s.rect(ux, y, uw, bh, "s-ok f-bg", 1.6)
        s.text(ux + uw / 2, y + bh / 2 + 5, "갱신 1번", "f-ok", 12, weight="600")
        arrow(s, gx + gw, y + bh / 2, ux - 3, y + bh / 2)
    arrow(s, 428, yt + bh / 2, gx - 3, yt + bh / 2)
    # 아래: 네 덩어리의 기울기가 모두 같은 .grad로 모임(위쪽 레일)
    rail = yb - 18
    cxs = [110 + i * 82 + 35 for i in range(4)]
    for cx in cxs:
        s.line(cx, yb, cx, rail, "s-mu", 1.4)
    s.line(cxs[0], rail, gx + gw / 2, rail, "s-mu", 1.4)
    arrow(s, gx + gw / 2, rail, gx + gw / 2, yb - 3)
    s.save(os.path.join(OUT, "47-accum.svg"))


if __name__ == "__main__":
    check()
    fig_memory()
    fig_float()
    fig_accum()
