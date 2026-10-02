"""57강 그림과 본문 수치 검산

그림: 같은 시가지 타일에 10대 교란을 1·3·5단계로 넣은 예시(PNG, 글자 없음), 교란별 RDI 막대(SVG)
실험 결과는 data/part8_runs.json의 robust(run_part8.py exp_robust)에서 읽음. 새 학습은 하지 않음.
교란 구현은 run_part8.py의 corrupt를 그대로 씀.
추가 확인(추론만, 학습 없음)
- LR 단계별 정확도(JSON과 같은지)와 1단계의 시가지 오분류
- DEF 1단계부터의 하락: 기울이기 크기와 반 화소 이동(보간만)으로 비교
- OCC·TC에서 예측이 어느 클래스로 쏠리는지
검산: RDI·CE·mCE, 증강 모델과의 차이, 물리량 예(SCR, 오프나디르 GSD, 건물 기울어짐, EPS, Koschmieder, 다크 채널)
"""
import json, math, os, sys
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg

OUT = os.path.join(HERE, "..", "fig")
R = json.load(open(os.path.join(DATA, "part8_runs.json"), encoding="utf-8"))["robust"]
NUIS = ["BC", "IV", "CM", "TC", "OCC", "LR", "DEF", "OV", "MB", "ATM"]
KO = {"BC": "배경 클러터", "IV": "조도", "CM": "위장", "TC": "열·반사", "OCC": "가림", "LR": "저해상도",
      "DEF": "사각 왜곡", "OV": "경계 절단", "MB": "블러", "ATM": "대기 산란"}


# ================================================================ 검산 (JSON 값)
def check_json():
    acc = R["acc"]
    for name in ("vdcnn", "vdcnn_aug", "flat_mlp"):
        print(name, "clean", acc[name]["clean"])
    for name in ("vdcnn", "vdcnn_aug"):
        c = acc[name]["clean"]
        rdi = {k: (c - np.mean(acc[name][k])) / c for k in NUIS}
        ce = {k: sum(1 - a for a in acc[name][k]) / sum(1 - a for a in acc["flat_mlp"][k]) for k in NUIS}
        mce = 100 * np.mean(list(ce.values()))
        for k in NUIS:
            assert abs(rdi[k] - R[name]["rdi"][k]) < 1e-3 and abs(ce[k] - R[name]["ce"][k]) < 1e-3
        print(name, "mCE %.2f (JSON %.3f)" % (mce, R[name]["mce"]), "max", R[name]["worst"], R[name]["max_rdi"],
              "robust<0.15", [k for k in NUIS if rdi[k] < 0.15], "critical>=0.35", [k for k in NUIS if rdi[k] >= 0.35],
              ">=0.40", [k for k in NUIS if rdi[k] >= 0.40])
        print("  RDI", {k: round(rdi[k], 3) for k in NUIS})
        print("  CE ", {k: round(ce[k], 3) for k in NUIS})
    v, a = R["vdcnn"], R["vdcnn_aug"]
    print("증강 효과 RDI·CE (vdcnn → aug):")
    for k in NUIS:
        print("  %-4s RDI %.3f→%.3f  CE %.3f→%.3f  %s" % (k, v["rdi"][k], a["rdi"][k], v["ce"][k], a["ce"][k],
                                                       "나아짐" if a["ce"][k] < v["ce"][k] else "나빠짐"))
    print("mCE 차이 %.2f, CE 증가 합 LR+MB+BC %.3f" % (a["mce"] - v["mce"], sum(a["ce"][k] - v["ce"][k] for k in ("LR", "MB", "BC"))))
    for k in ("TC", "OCC"):
        e1 = sum(1 - x for x in acc["vdcnn"][k]); e0 = sum(1 - x for x in acc["flat_mlp"][k])
        print(k, "오차 합 VD-CNN %.4f, MLP %.4f, CE %.3f, 5단계 정확도 VD %.4f MLP %.4f" % (e1, e0, e1 / e0, acc["vdcnn"][k][4], acc["flat_mlp"][k][4]))
    print("MLP 교란별 5단계 정확도", {k: acc["flat_mlp"][k][4] for k in NUIS})


# ================================================================ 추가 확인 (추론만)
def check_infer():
    torch.set_num_threads(1)
    import run_part8 as R8
    from part5_common import TinyCNN, evaluate
    from part7_common import to_tensor
    import make_part5 as P5
    raw = P5.load(); xs = raw["x_test"]; ys = R8.D["y_test"]
    m = R8.base_model()
    aug = TinyCNN(); aug.load_state_dict(torch.load(os.path.join(DATA, ".part8_aug.pt")))

    for s_ in range(1, 6):
        print("LR %d 정확도 VD-CNN %.4f 증강 %.4f" % (s_, evaluate(m, to_tensor(R8.corrupt(xs, "LR", s_)), ys)[1],
                                                 evaluate(aug, to_tensor(R8.corrupt(xs, "LR", s_)), ys)[1]))
    # LR 1단계: 시가지 타일
    p = evaluate(m, to_tensor(R8.corrupt(xs, "LR", 1)), ys)[2].numpy(); y = ys.numpy()
    print("LR 1단계 시가지 정답률 %.3f, 시가지를 나지로 %.3f" % ((p[y == 2] == 2).mean(), (p[y == 2] == 4).mean()))
    # DEF: 기울이기 크기별, 반 화소 이동
    rng = np.random.default_rng(1)
    for sh in (0.02, 0.1):
        x = xs.copy()
        for j in range(len(x)):
            M = np.array([[1, sh * rng.choice([-1, 1])], [0, 1 + sh / 2]]); off = np.array([16, 16]) - M @ np.array([16, 16])
            for bnd in range(4):
                x[j, bnd] = ndi.affine_transform(xs[j, bnd], M, offset=off, order=1, mode="reflect")
        print("DEF 기울이기 %.2f 정확도 %.4f" % (sh, evaluate(m, to_tensor(np.clip(x, 0, 1)), ys)[1]))
    print("반 화소 이동(쌍선형 보간만) %.4f" % evaluate(m, to_tensor(ndi.shift(xs, (0, 0, 0.5, 0.5), order=1, mode="reflect")), ys)[1])
    for k in ("OCC", "TC"):
        for s in (1, 5):
            p = evaluate(m, to_tensor(R8.corrupt(xs, k, s)), ys)[2].numpy()
            print(k, s, "예측 분포(산림 농경지 시가지 수계 나지 갯벌)", np.bincount(p, minlength=6))
    # 다크 채널(4밴드 최솟값, 3×3 창 최솟값): 클린 vs ATM 5단계(t = 0.5)
    def dark(x):
        return ndi.minimum_filter(x.min(1), size=(1, 3, 3))
    d0, d5 = np.median(dark(xs)), np.median(dark(R8.corrupt(xs, "ATM", 5)))
    print("다크 채널 중앙값 클린 %.4f → ATM5 %.4f" % (d0, d5))


# ================================================================ 물리량 예
def check_phys():
    # SCR: 선박(표적)과 바다(배경)
    print("SCR 잔잔한 바다 %.2f, 흰 물결 %.2f" % (abs(0.14 - 0.04) / 0.008, abs(0.14 - 0.05) / 0.08))
    for th in (15, 30, 45):
        c = math.cos(math.radians(th))
        print("오프나디르 %d°: 횡방향 GSD 배율 %.3f (0.5 m → %.2f m), 종방향 %.3f (→ %.2f m)" % (th, 1 / c**2, 0.5 / c**2, 1 / c, 0.5 / c))
    print("30 m 건물 30° 기울어짐 %.1f m = %.0f 화소(0.5 m)" % (30 * math.tan(math.radians(30)), 30 * math.tan(math.radians(30)) / 0.5))
    area = 6 * 2.5
    for g in (0.5, 3.0):
        e = area / g**2
        print("소형선박 %.0f m² GSD %.1f m: EPS %.2f 화소² (한 변 %.1f)" % (area, g, e, math.sqrt(e)))
    print("COCO 소형 1024 화소² 한 변", math.sqrt(1024), "EPS 16 한 변", math.sqrt(16))
    for t in (0.9, 0.5, 0.4):
        print("투과율 %.1f ↔ βd %.3f" % (t, -math.log(t)))
    # 산림 청색(0.03)이 t=0.5, A=0.12에서
    print("산림 청색 0.03 → t=0.5: %.3f" % (0.03 * 0.5 + 0.12 * 0.5))
    print("복원 J=(I-A)/t+A: %.3f" % ((0.075 - 0.12) / 0.5 + 0.12))


# ================================================================ 그림 1: 교란 예시 (PNG)
def fig_examples():
    import run_part8 as R8
    import make_part5 as P5
    D = P5.load()
    x, y, msk = D["x_test"], D["y_test"], D["m_test"]
    idx = [i for i in np.where(y == 2)[0] if (msk[i] == 2).mean() > 0.95]
    t = x[idx[0]:idx[0] + 1]
    rows = []
    for s in (1, 3, 5):
        rows.append([t[0] if s == 1 else None] + [R8.corrupt(t, k, s, seed=2)[0] for k in NUIS])
    fc0 = t[0][[3, 2, 1]]
    lo = np.percentile(fc0, 1, axis=(1, 2)); hi = np.percentile(fc0, 99, axis=(1, 2))
    SC, G = 3, 6
    N = 32 * SC
    W = 11 * N + 10 * G + G          # 클린 열 뒤에 간격을 하나 더 둠
    H = 3 * N + 2 * G
    canvas = np.zeros((H, W, 4), np.uint8)
    for r, row in enumerate(rows):
        for c, p in enumerate(row):
            if p is None:
                continue
            fc = p[[3, 2, 1]]
            v = np.clip((fc - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1) ** 0.8
            rgb = np.round(v * 255).astype(np.uint8).transpose(1, 2, 0).repeat(SC, 0).repeat(SC, 1)
            x0 = c * (N + G) + (G if c > 0 else 0); y0 = r * (N + G)
            canvas[y0:y0 + N, x0:x0 + N, :3] = rgb
            canvas[y0:y0 + N, x0:x0 + N, 3] = 255
    img = Image.fromarray(canvas, "RGBA").quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    img.save(os.path.join(OUT, "57-examples.png"), optimize=True)
    print("예시 타일", idx[0], "크기", W, H, os.path.getsize(os.path.join(OUT, "57-examples.png")), "bytes")


# ================================================================ 그림 2: 교란별 RDI (SVG)
def fig_rdi():
    s = Svg(720, 470, "교란 10가지의 RDI. VD-CNN과 기하+광학 증강 VD-CNN. 0.15 아래는 강건, 0.35~0.40 위는 심각")
    L, Rr = 150, 690
    xmax = 0.6
    sx = lambda v: L + (Rr - L) * v / xmax
    top, bh, bg, gg = 52, 12, 2, 12
    gh = 2 * bh + bg
    yb = top + 10 * (gh + gg) - gg + 8
    # 기준 구간: 심각(0.35~0.40) 띠, 강건 경계 0.15
    s.rect(sx(0.35), top - 10, sx(0.40) - sx(0.35), yb - top + 10, "f-bds", 0)
    s.line(sx(0.15), top - 10, sx(0.15), yb, "s-ok", 1.4, dash="5 4")
    s.text(sx(0.15) + 4, top - 14, "강건 경계", "f-ok", 11, anchor="start")
    s.text(sx(0.375), top - 14, "심각 경계", "f-bd", 11)
    # 범례
    s.rect(L, 12, 14, 10, "s-ac f-ac", 1.0); s.text(L + 20, 21, "VD-CNN", "f-fg", 12, anchor="start")
    s.rect(L + 110, 12, 14, 10, "s-mu f-mu", 1.0); s.text(L + 130, 21, "증강 VD-CNN", "f-fg", 12, anchor="start")
    for gi, k in enumerate(NUIS):
        y0 = top + gi * (gh + gg)
        s.text(L - 10, y0 + gh / 2 + 5, f"{k} {KO[k]}", "f-fg", 12, anchor="end")
        for si, (name, cls) in enumerate((("vdcnn", "s-ac f-ac"), ("vdcnn_aug", "s-mu f-mu"))):
            v = R[name]["rdi"][k]
            y = y0 + si * (bh + bg)
            s.rect(L, y, sx(v) - L, bh, cls, 0.8)
            if si == 0:
                s.text(sx(v) + 5, y + bh - 2, f"{v:.3f}", "f-ac", 10, anchor="start", weight="600")
    s.line(L, yb, Rr, yb, "s-mu", 1.2)
    s.line(L, top - 10, L, yb, "s-mu", 1.2)
    for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        s.line(sx(v), yb, sx(v), yb + 4, "s-mu", 1.2)
        s.text(sx(v), yb + 17, f"{v:g}", "f-mu", 11)
    s.text((L + Rr) / 2, yb + 36, "RDI (클린 대비 하락률)", "f-mu", 12)
    s.save(os.path.join(OUT, "57-rdi.svg"))


if __name__ == "__main__":
    check_json()
    check_phys()
    if "--no-infer" not in sys.argv:
        check_infer()
    fig_examples()
    fig_rdi()
