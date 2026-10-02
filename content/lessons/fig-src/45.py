"""45강 그림과 본문 수치 검산

그림: 같은 타일에 증강을 건 예시(PNG, 글자 없음), 증강별 세 시험 정확도(SVG)
학습 결과는 data/part7_runs.json의 aug 실험(run_part7.py exp_aug)에서 읽음. 새 학습은 하지 않음.
증강 구현은 data/part7_common.py의 aug_*. 그림의 광학·밴드 섞기 예시는 같은 식을 고정값으로 반사율에 적용한 것.
검산: 표의 값, 연무·계절 전후 NDVI, 밴드 섞기·밴드별 왜곡 뒤 NDVI, 상자 회전, 크기 증강과 6부 선박 크기, 49강 시드 범위
"""
import json, math, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from svglib import Svg
from make_part5 import MEAN, load

OUT = os.path.join(HERE, "..", "fig")
R = json.load(open(os.path.join(HERE, "..", "data", "part7_runs.json"), encoding="utf-8"))
A = R["aug"]
KEYS = ["none", "geo", "photo", "geo_photo", "band_shuffle", "geo_mixup0.2"]
NAMES = {"none": "없음", "geo": "기하", "photo": "광학", "geo_photo": "기하+광학", "band_shuffle": "밴드 섞기",
         "geo_mixup0.2": "기하+믹스업"}
HAZE = np.array([0.035, 0.022, 0.010, 0.004])   # part7_common.aug_photo의 산란광 최댓값 = make_part5 test_shift의 값


def ndvi(s):
    return (s[3] - s[2]) / (s[3] + s[2])


# ================================================================ 검산
def check():
    print("설정 | 시험 best/last | 회전 시험 best/last | 연무 best/last | best 에폭 | 검증 손실 최소 | 마지막 10에폭 검증 정확도 범위")
    for k in KEYS:
        v, h = A[k], A[k]["hist"]
        print(f"{NAMES[k]:8s} {v['best']['test']:.4f}/{v['last']['test']:.4f}  {v['best']['test_rot90']:.4f}/"
              f"{v['last']['test_rot90']:.4f}  {v['best']['shift']:.4f}/{v['last']['shift']:.4f}  ep{v['best_epoch']}"
              f"  vlmin {min(h['val_loss']):.4f}  va10 {min(h['val_acc'][-10:]):.4f}~{max(h['val_acc'][-10:]):.4f}"
              f"  tl_last {h['train_loss'][-1]}")
    print("시험 타일 900장이면 한 장 =", round(1 / 900, 4))
    s = R["seeds"]["bn"]["runs"]
    print("49강 시드 5개(학습 4,200장) 연무 best 범위", min(r["shift_best"] for r in s), max(r["shift_best"] for r in s),
          " 시험 best 범위", min(r["test_best"] for r in s), max(r["test_best"] for r in s))

    # NDVI: 클래스 평균 반사율(B2 B3 B4 B8)
    f, a, w = MEAN[0].astype(float), MEAN[1].astype(float), MEAN[3].astype(float)
    print("NDVI 산림 %.3f 농경지 %.3f 수계 %.3f" % (ndvi(f), ndvi(a), ndvi(w)))
    fs = f.copy(); fs[3] *= 0.78; fs = fs + HAZE                     # test_shift: 식생 근적외 0.78배 + 산란광
    print("연무·계절 뒤 산림 NDVI %.3f, 청색 %.3f→%.3f (%.2f배)" % (ndvi(fs), f[0], fs[0], fs[0] / f[0]))
    fp = f.copy(); fp[3] *= 0.85; fp = fp + HAZE                     # 광학 증강 최대 산란광 + 근적외 0.85배
    print("광학 증강(산란광 최대, 근적외 0.85) 뒤 산림 NDVI %.3f" % ndvi(fp))
    sw = f[[0, 1, 3, 2]]                                              # 적색↔근적외 맞바꿈
    print("적색↔근적외 바꾼 산림 NDVI %.3f" % ndvi(sw))
    g15 = f * np.array([1, 1, 1.15, 0.85])
    print("이득 최악(적 1.15, 근적외 0.85) 산림 NDVI %.3f" % ndvi(g15))
    gw = g15.copy(); gw[3] *= 0.85; gw = gw + HAZE                   # 이득 최악 + 계절 흉내 + 최대 산란광
    print("이득 최악 + 근적외 0.85배 + 최대 산란광 산림 NDVI %.3f" % ndvi(gw))
    g50 = f * np.array([1, 1, 1.5, 0.6])
    print("적 1.5배, 근적외 0.6배 산림 NDVI %.3f" % ndvi(g50))
    print("지역 이득 exp(N(0,0.12)) 의 ±1σ 배율 %.3f~%.3f" % (math.exp(-0.12), math.exp(0.12)))

    # 상자 좌표: 좌우 뒤집기와 90° 회전(반시계, torch.rot90 k=1, 정사각 W)
    W = 32
    a_ = np.zeros((W, W)); a_[3:7, 20:30] = 1                       # 행 3~6, 열 20~29 객체
    def box(m):
        r, c = np.where(m); return (c.min(), r.min(), c.max() + 1, r.max() + 1)  # (x1,y1,x2,y2) 연속 좌표
    x1, y1, x2, y2 = box(a_)
    print("원래 상자", (x1, y1, x2, y2), "뒤집기", box(a_[:, ::-1]), "식", (W - x2, y1, W - x1, y2),
          "회전", box(np.rot90(a_)), "식", (y1, W - x2, y2, W - x1))
    # 임의 각 회전: 40×10 객체를 30° 돌리면 감싸는 상자
    L_, S_, t = 40, 10, math.radians(30)
    bw, bh = L_ * math.cos(t) + S_ * math.sin(t), L_ * math.sin(t) + S_ * math.cos(t)
    print("30° 회전 감싸는 상자 %.1f×%.1f 넓이 %.0f / 객체 %d = %.2f배" % (bw, bh, bw * bh, L_ * S_, bw * bh / (L_ * S_)))
    # 크기 증강(Ultralytics 8.4 기본 scale 0.5 → 0.5~1.5배)과 6부 클래스 길이
    print("0.5 m 영상, 4.5 m 차 9화소 → %.1f~%.1f화소 = %.2f~%.2f m" % (9 * 0.5, 9 * 1.5, 4.5 * 0.5, 4.5 * 1.5))
    print("선박 40 m ×0.5 = %.0f m, 소형선박 20 m ×1.5 = %.0f m" % (40 * 0.5, 20 * 1.5))
    # 믹스업 Beta(0.2, 0.2)에서 λ 분포
    lam = np.random.default_rng(0).beta(0.2, 0.2, 200000)
    print("Beta(0.2,0.2): λ<0.1 또는 >0.9 비율 %.3f, 0.3~0.7 비율 %.3f" % (((lam < 0.1) | (lam > 0.9)).mean(),
                                                                       ((lam > 0.3) & (lam < 0.7)).mean()))


# ================================================================ 그림 1: 증강 예시 (PNG)
def pick(D, c):
    x, y, m = D["x_train"], D["y_train"], D["m_train"]
    idx = np.where(y == c)[0]
    frac = np.array([(m[i] != c).mean() for i in idx])
    blob = np.array([ndi.gaussian_filter(x[i][0], 3).max() - np.median(x[i][0]) for i in idx])
    pc = np.where(frac == 0)[0][:60]
    return int(idx[pc[blob[pc].argmin()]])


def fig_examples():
    D = load()
    ia, iu = pick(D, 1), pick(D, 2)                       # 농경지(방향 있는 필지), 시가지
    t, u = D["x_train"][ia].astype(float), D["x_train"][iu].astype(float)
    rng = np.random.default_rng(3)
    photo = t.copy(); photo[3] *= 0.85
    photo = photo + HAZE[:, None, None] + rng.normal(0, 0.004, t.shape)
    panels = [t,
              t[:, :, ::-1],                               # 좌우 뒤집기
              np.rot90(t, 1, (1, 2)),                      # 90° 반시계(torch.rot90 k=1과 같은 방향)
              photo,                                       # 산란광 최대 + 근적외 0.85배 + 잡음
              t[[0, 1, 3, 2]],                             # 밴드 섞기: 적색↔근적외
              0.6 * t + 0.4 * u]                           # 믹스업 λ=0.6 (농경지 0.6 + 시가지 0.4)
    ref = np.stack([t, u])[:, [3, 2, 1]]                  # 표시 범위는 원본 두 타일로 한 번만 정함
    lo = np.percentile(ref, 1, axis=(0, 2, 3)); hi = np.percentile(ref, 99, axis=(0, 2, 3))
    SC, G, N = 5, 8, 32 * 5
    canvas = np.zeros((N, 6 * N + 5 * G, 4), np.uint8)
    for k, p in enumerate(panels):
        fc = p[[3, 2, 1]]                                  # 폴스컬러: B8, B4, B3 → R, G, B
        l_, h_ = lo, hi
        if k == 4:   # 밴드 섞기: 근적외 값이 가시광 자리에 들어가 공통 범위로는 한 색으로 포화됨 → 이 패널만 따로 늘림
            l_, h_ = np.percentile(fc, 1, axis=(1, 2)), np.percentile(fc, 99, axis=(1, 2))
        v = np.clip((fc - l_[:, None, None]) / (h_ - l_)[:, None, None], 0, 1) ** 0.8
        rgb = np.round(v * 255).astype(np.uint8).transpose(1, 2, 0).repeat(SC, 0).repeat(SC, 1)
        c0 = k * (N + G)
        canvas[:, c0:c0 + N, :3] = rgb
        canvas[:, c0:c0 + N, 3] = 255
    img = Image.fromarray(canvas, "RGBA").quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    img.save(os.path.join(OUT, "45-examples.png"), optimize=True)
    print("예시 타일", ia, iu, "농경지 타일 NDVI 중앙값 원본/광학", round(float(np.median(ndvi(t))), 3), round(float(np.median(ndvi(photo))), 3))


# ================================================================ 그림 2: 증강별 세 시험 정확도 (SVG)
def fig_bars():
    s = Svg(720, 400, "증강 여섯 가지의 시험, 90° 돌린 시험, 연무·계절 시험 정확도. 막대는 best 모델, 세로 눈금은 last 모델")
    L, Rr = 150, 650
    sx = lambda v: L + (Rr - L) * v
    series = [("test", "시험", "f-ac"), ("test_rot90", "90° 돌린 시험", "f-mu"), ("shift", "연무·계절 시험", "f-bd")]
    # 범례
    lx = L
    for _, name, cls in series:
        s.rect(lx, 14, 14, 10, cls.replace("f-", "s-") + " " + cls, 1.0)
        s.text(lx + 20, 23, name, "f-fg", 12, anchor="start")
        lx += 34 + 13 * len(name)
    s.line(lx + 6, 11, lx + 6, 27, "s-fg", 2.2)
    s.text(lx + 14, 23, "last", "f-fg", 12, anchor="start")
    top, bh, bg, gg = 48, 10, 2, 16
    gh = 3 * bh + 2 * bg
    for gi, k in enumerate(KEYS):
        y0 = top + gi * (gh + gg)
        s.text(L - 12, y0 + gh / 2 + 5, NAMES[k], "f-fg", 13, anchor="end")
        for si, (key, _, cls) in enumerate(series):
            y = y0 + si * (bh + bg)
            b, l = A[k]["best"][key], A[k]["last"][key]
            s.rect(L, y, sx(b) - L, bh, cls.replace("f-", "s-") + " " + cls, 0.8)
            s.line(sx(l), y - 2, sx(l), y + bh + 2, "s-fg", 2.2)
            if key == "shift":
                s.text(max(sx(b), sx(l)) + 7, y + bh - 1, f"{b:.3f}", "f-bd", 11, anchor="start", weight="600")
    yb = top + 6 * (gh + gg) - gg + 8
    s.line(L, yb, Rr, yb, "s-mu", 1.2)
    s.line(L, top - 6, L, yb, "s-mu", 1.2)
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        s.line(sx(v), yb, sx(v), yb + 4, "s-mu", 1.2)
        s.text(sx(v), yb + 17, f"{v:g}", "f-mu", 11)
    s.text((L + Rr) / 2, yb + 38, "정확도", "f-mu", 12)
    s.save(os.path.join(OUT, "45-results.svg"))


if __name__ == "__main__":
    check()
    fig_examples()
    fig_bars()
