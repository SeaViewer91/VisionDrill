"""54강 그림과 본문 수치: 슬라이스 정확도와 BH 판정(SVG), 신뢰 학습 vs 단순 불일치(SVG), 분포 거리 W1·MMD²(SVG)

본문 수치 대부분은 data/part8_runs.json의 slice(계산: data/run_part8.py의 exp_slice)에서 가져옴.
JSON에 worst 8개만 있으므로 슬라이스 21개 전체는 같은 정의로 여기서 다시 계산함(학습 없음, VD-CNN 추론만, 몇 초).
JSON에 없는 것(BH q = 0.05·본페로니 기각 수, W1의 상대 크기, 탐지 슬라이스 격차, 6부 정답의 클래스×타일 종류 PMI·지니 계수,
BAS·EIAH 손 계산 예)도 여기서 계산해 출력함
"""
import json, math, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg

OUT = os.path.join(HERE, "..", "fig")
R = json.load(open(os.path.join(DATA, "part8_runs.json"), encoding="utf-8"))
SL = R["slice"]
CLS5 = ["산림", "농경지", "시가지", "수계", "나지", "갯벌"]


def bh(p, q):
    p = np.asarray(p); M = len(p); o = np.argsort(p)
    ok = np.where(p[o] <= (np.arange(1, M + 1) / M) * q)[0]
    return set(o[: ok.max() + 1].tolist()) if len(ok) else set()


# ---------------------------------------------------------------- 1. 슬라이스 21개 다시 계산 (run_part8.exp_slice와 같은 정의)
import torch
import part5_common as P5C
import make_part5 as P5
from part5_common import TinyCNN

D = P5C.data()
raw = P5.load()
m = TinyCNN(); m.load_state_dict(torch.load(os.path.join(DATA, ".part5_base.pt"))); m.eval()
ALL = {}
for split in ("test", "shift"):
    x, y = D[f"x_{split}"], D[f"y_{split}"]
    with torch.no_grad():
        pred = torch.cat([m(x[i:i + 300]) for i in range(0, len(x), 300)]).argmax(1)
    ok = (pred == y).numpy().astype(float); pg = ok.mean()
    xr = raw[f"x_{split}"]; mm = raw[f"m_{split}"]; reg = raw[f"r_{split}"]
    bright = xr[:, :3].mean(axis=(1, 2, 3))
    mixed = np.array([len(np.unique(a)) > 1 for a in mm])
    mx = xr[:, :3].max(axis=(1, 2, 3)); cloud = mx > np.percentile(mx, 90)
    sl = {f"클래스={CLS5[c]}": (y.numpy() == c) for c in range(6)}
    for r in np.unique(reg):
        sl[f"지역={int(r)}"] = reg == r
    sl["경계 타일"] = mixed; sl["단일 클래스 타일"] = ~mixed
    t1, t2 = np.percentile(bright, [33.3, 66.7])
    sl["밝기 하위 1/3"] = bright < t1; sl["밝기 상위 1/3"] = bright >= t2
    sl["밝은 점 상위 10%(구름 의심)"] = cloud
    for c in range(6):
        sl[f"클래스={CLS5[c]} & 경계 타일"] = (y.numpy() == c) & mixed
    rows = []
    for k, msk in sl.items():
        n = int(msk.sum())
        if n < 10:
            continue
        ps = ok[msk].mean(); z = (ps - pg) / math.sqrt(pg * (1 - pg) / n)
        rows.append(dict(slice=k, n=n, acc=ps, z=z, p=stats.norm.cdf(z)))
    pv = [r["p"] for r in rows]
    r10, r05 = bh(pv, 0.1), bh(pv, 0.05)
    bonf = sum(p < 0.05 / len(pv) for p in pv)
    for i, r in enumerate(rows):
        r["bh"] = i in r10
    ALL[split] = dict(glob=pg, rows=rows)
    srt = sorted(rows, key=lambda r: r["p"])
    print(f"[{split}] 전체 정확도 {pg:.4f} (JSON {SL[split]['global_acc']}) 슬라이스 {len(rows)} | p<0.05 {sum(p < 0.05 for p in pv)} "
          f"| BH q=0.1 {len(r10)} (JSON {SL[split]['n_reject_bh']}) | BH q=0.05 {len(r05)} | 본페로니 0.05/{len(pv)}={0.05 / len(pv):.5f} → {bonf}")
    for i, r in enumerate(srt[:8], 1):
        print(f"   {i:2d} {r['slice']:<20} n={r['n']:3d} acc={r['acc']:.4f} z={r['z']:.3f} p={r['p']:.3g} "
              f"BH기준(q=0.1) {i / len(pv) * 0.1:.4f} (q=0.05) {i / len(pv) * 0.05:.4f}")
    lo = min(rows, key=lambda r: r["acc"])
    print(f"   정확도 최솟값 슬라이스: {lo['slice']} n={lo['n']} acc={lo['acc']:.4f}; z 최솟값: {min(rows, key=lambda r: r['z'])['slice']}")
    lab = max(rows, key=lambda r: r["acc"])
    print(f"   정확도 최댓값 슬라이스: {lab['slice']} acc={lab['acc']:.4f} → 격차 {lab['acc'] - lo['acc']:.4f}")
# 농경지&경계 손 계산
pg = ALL["test"]["glob"]
print("Z 손 계산(농경지&경계, n=54, 46/54):", round(46 / 54, 4), round((46 / 54 - pg) / math.sqrt(pg * (1 - pg) / 54), 3),
      "| 표준오차", round(math.sqrt(pg * (1 - pg) / 54), 4), "| 틀린 수", round(54 * (1 - 0.8519)))

# ---------------------------------------------------------------- 2. 신뢰 학습
L = SL["label_noise"]; sd = L["simple_disagree"]
tp_cl = round(L["precision"] * L["n_flagged"]); tp_sd = round(sd["precision"] * sd["n"])
print(f"신뢰 학습: 표시 {L['n_flagged']} 중 진짜 {tp_cl} (오경보 {L['n_flagged'] - tp_cl}), 놓침 {L['n_flipped'] - tp_cl}, "
      f"재현율 {tp_cl / L['n_flipped']:.4f} | 문턱 {min(L['thresholds'])}~{max(L['thresholds'])}")
print(f"단순 불일치: 표시 {sd['n']} 중 진짜 {tp_sd} (오경보 {sd['n'] - tp_sd}), 놓침 {L['n_flipped'] - tp_sd}, 재현율 {tp_sd / L['n_flipped']:.4f}")
print("바뀐 비율", L["n_flipped"] / L["n"])

# ---------------------------------------------------------------- 3. 분포 거리
Dr = R["slice"]["drift"]
tm = raw["x_train"].mean(axis=(2, 3))
sdv = tm.std(0)
print("학습 타일 밴드 평균의 표준편차", sdv.round(4))
for s_ in ("val", "test", "shift"):
    w = np.array(Dr["w1_band_means"][s_])
    print(f"W1 {s_}", w, "표준편차 대비", (w / sdv).round(3), "| MMD²", Dr["mmd"][s_])
print("W1 shift/test 배수", (np.array(Dr["w1_band_means"]["shift"]) / np.array(Dr["w1_band_means"]["test"])).round(1))
print("MMD² shift/test", round(Dr["mmd"]["shift"]["mmd2"] / Dr["mmd"]["test"]["mmd2"], 1), "| 순열 100회 최소 p = 1/101 =", round(1 / 101, 4))

# ---------------------------------------------------------------- 4. 탐지 슬라이스 (6부 항만 자료)
ds = SL["detection_slices"]; base = R["tide"]["base_map50"]
v = {k: d["map50"] for k, d in ds.items()}
print("탐지 슬라이스", v, "전체", base, "| 최소", min(v.values()), "격차", round(max(v.values()) - min(v.values()), 4),
      "| 슬라이스 단순 평균", round(np.mean(list(v.values())), 4), "| 정답 수", {k: d["n_gt"] for k, d in ds.items()})

# ---------------------------------------------------------------- 5. PMI·지니 (6부 정답: 클래스 × 타일 종류)
from make_part6 import tiles
T = tiles()
kinds = ["harbor", "sea", "marina"]
C = np.zeros((3, 3))
for t in T:
    for o in t["objs"]:
        C[o["cls"], kinds.index(t["kind"])] += 1
print("정답 수(행=선박·소형선박·차량, 열=항만·외해·마리나)\n", C.astype(int))
P = C / C.sum(); pc = P.sum(1, keepdims=True); pb = P.sum(0, keepdims=True)
with np.errstate(divide="ignore"):
    pmi = np.log(P / (pc * pb))
print("PMI\n", pmi.round(3))
N = C.sum(1)


def gini(n):
    n = np.asarray(n, float); Cn = len(n)
    return np.abs(n[:, None] - n[None]).sum() / (2 * Cn * n.sum())


print("클래스별 정답 수", N, "지니", round(gini(N), 4), "| 최댓값 (C−1)/C: C=3", round(gini([1, 0, 0]), 4), "C=6", round(gini([1, 0, 0, 0, 0, 0]), 4),
      "C=20", round(19 / 20, 3))
print("5부 학습 클래스 수", np.bincount(D["y_train"].numpy()), "지니", gini(np.bincount(D["y_train"].numpy())))
print("예: 3클래스 900:90:10 지니", round(gini([900, 90, 10]), 4), "| 9000:900:100", round(gini([9000, 900, 100]), 4),
      "| 1000:0:0", round(gini([1000, 0, 0]), 4), "| 990:9:1", round(gini([990, 9, 1]), 4))
print("지니 불순도 1−Σp² 최댓값(C=3, 균등)", round(1 - 3 * (1 / 3) ** 2, 4))

# ---------------------------------------------------------------- 6. BAS 손 계산 (이 자습서에서 만든 예: MC 드롭아웃 20회의 네 변 표준편차, 화소)
for nm, sds in [("선박(뱃전 뚜렷)", [0.6, 0.7, 0.5, 0.6]), ("양식장(경계 흐림)", [3.1, 2.6, 4.0, 3.3])]:
    print(f"BAS {nm}: 표준편차 {sds} → 분산 {[round(s * s, 2) for s in sds]} → BAS {np.mean(np.square(sds)):.3f} 화소²")

# ---------------------------------------------------------------- 7. EIAH 손 계산 (이 자습서에서 만든 예)
cands = [("A 항만 타일(상자 60개)", 0.90, 30), ("B 외해 타일(누락 의심 1개)", 0.35, 2), ("C 마리나 타일(오표기 의심 3개)", 0.40, 5)]
for nm, dl, mins in cands:
    print(f"EIAH {nm}: ΔL {dl} / {mins}분 = {dl / (mins / 60):.2f} /시간")


# ---------------------------------------------------------------- 그림 1: 슬라이스 정확도와 BH 판정
SHORT = lambda k: k.replace("클래스=", "").replace("지역=", "지역 ").replace(" & 경계 타일", " 경계")
s = Svg(780, 300, "VD-CNN의 p값이 작은 슬라이스 7개의 정확도. 왼쪽은 시험(전체 0.974), 오른쪽은 연무·계절 시험(전체 0.708). "
                  "색칠한 막대는 BH(q = 0.1)로 기각된 취약 슬라이스, 회색은 기각되지 않은 것, 점선은 전체 정확도")
for pi, (split, title) in enumerate([("test", "시험"), ("shift", "연무·계절 시험")]):
    X0 = 20 + pi * 390
    LX, BX0, BX1 = X0 + 98, X0 + 104, X0 + 314
    rows = sorted(ALL[split]["rows"], key=lambda r: r["p"])[:7]
    g = ALL[split]["glob"]
    s.text(X0 + 4, 22, f"{title} (전체 {g:.3f})", "f-fg", 13, anchor="start", weight="600")
    Y0, RH = 44, 32
    xv = lambda v: BX0 + v * (BX1 - BX0)
    for t in (0, 0.5, 1.0):
        s.line(xv(t), Y0 - 4, xv(t), Y0 + RH * 7, "s-mu", 0.6)
        s.text(xv(t), Y0 + RH * 7 + 16, f"{t:g}", "f-mu", 11)
    for i, r in enumerate(rows):
        y = Y0 + i * RH
        s.text(LX, y + 15, SHORT(r["slice"]), "f-fg", 12, anchor="end")
        s.rect(BX0, y + 4, max(r["acc"], 0.004) * (BX1 - BX0), 16, "f-ac" if r["bh"] else "f-mu", 0)
        s.text(BX1 + 6, y + 11, f"{r['acc']:.3f}", "f-fg", 11, anchor="start")
        s.text(BX1 + 6, y + 24, f"n={r['n']}", "f-mu", 10, anchor="start")
    s.line(xv(g), Y0 - 4, xv(g), Y0 + RH * 7, "s-bd", 1.6, "4 3")
s.save(os.path.join(OUT, "54-slices.svg"))

# ---------------------------------------------------------------- 그림 2: 신뢰 학습 vs 단순 불일치
s = Svg(720, 200, "바꾼 라벨 424개를 찾을 때 표시한 후보 수. 초록은 실제로 바뀐 라벨, 주황은 오경보. "
                  "단순 불일치는 612개를 표시해 421개를 잡고 오경보 191개, 신뢰 학습은 357개를 표시해 332개를 잡고 오경보 25개")
BX0, BX1, MAXV = 150, 600, 650
xv = lambda v: BX0 + v / MAXV * (BX1 - BX0)
for t in (0, 200, 400, 600):
    s.line(xv(t), 30, xv(t), 150, "s-mu", 0.6)
    s.text(xv(t), 168, f"{t}", "f-mu", 11)
s.line(xv(424), 30, xv(424), 150, "s-fg", 1.2, "4 3")
s.text(xv(424), 22, "바꾼 라벨 424", "f-fg", 11)
for i, (nm, tp, n, pr, rc) in enumerate([("단순 불일치", tp_sd, sd["n"], sd["precision"], sd["recall"]),
                                         ("신뢰 학습", tp_cl, L["n_flagged"], L["precision"], L["recall"])]):
    y = 44 + i * 56
    s.text(BX0 - 10, y + 18, nm, "f-fg", 13, anchor="end")
    s.rect(xv(0), y, xv(tp) - xv(0), 26, "f-ok", 0)
    s.rect(xv(tp), y, xv(n) - xv(tp), 26, "f-bd", 0)
    s.text(xv(n) + 6, y + 12, f"정밀도 {pr:.3f}", "f-fg", 11, anchor="start")
    s.text(xv(n) + 6, y + 26, f"재현율 {rc:.3f}", "f-fg", 11, anchor="start")
for j, (cls, lab) in enumerate([("f-ok", "실제로 바뀐 라벨"), ("f-bd", "오경보")]):
    lx = BX0 + j * 150
    s.rect(lx, 182, 14, 12, cls, 0)
    s.text(lx + 20, 192, lab, "f-fg", 12, anchor="start")
s.save(os.path.join(OUT, "54-cl.svg"))

# ---------------------------------------------------------------- 그림 3: 밴드별 W1
s = Svg(720, 250, "학습 자료 대비 타일별 밴드 평균 반사율의 W1. 검증·시험은 0.002~0.014, 연무·계절 시험은 청·녹 밴드에서 0.033·0.029")
PX0, PX1, PY0, PY1 = 70, 700, 40, 200
VMAX = 0.04
yv = lambda v: PY1 - v / VMAX * (PY1 - PY0)
for t in (0, 0.01, 0.02, 0.03, 0.04):
    s.line(PX0, yv(t), PX1, yv(t), "s-mu", 0.6, "2 3" if t else None)
    s.text(PX0 - 8, yv(t) + 4, f"{t:g}", "f-mu", 11, anchor="end")
SPL = [("val", "검증", "f-mu"), ("test", "시험", "f-ac"), ("shift", "연무·계절", "f-bd")]
GW = (PX1 - PX0) / 4; BW = 40
for b in range(4):
    gx = PX0 + GW * (b + 0.5)
    for j, (k, _, cls) in enumerate(SPL):
        v = Dr["w1_band_means"][k][b]
        x = gx - 1.5 * BW - 4 + j * (BW + 4)
        s.rect(x, yv(v), BW, PY1 - yv(v), cls, 0)
        s.text(x + BW / 2, yv(v) - 5, f"{v:.4f}", "f-fg", 10)
    s.text(gx, PY1 + 20, Dr["w1_bands"][b] + ["(청)", "(녹)", "(적)", "(근적외)"][b], "f-fg", 12)
for j, (_, lab, cls) in enumerate(SPL):
    lx = PX0 + 10 + j * 110
    s.rect(lx, 12, 14, 12, cls, 0)
    s.text(lx + 20, 22, lab, "f-fg", 12, anchor="start")
s.text(PX1, 22, "W1 (반사율)", "f-mu", 11, anchor="end")
s.save(os.path.join(OUT, "54-drift.svg"))
print("그림 3개 저장")
