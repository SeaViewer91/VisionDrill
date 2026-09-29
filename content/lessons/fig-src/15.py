"""15강 그림: 무작위 분할 vs 공간 블록 분할 도식, 교차검증 방식별 오류율, NDVI 코렐로그램

수치는 3부 공통 자료(data/part3_landcover.csv)로 scikit-learn을 써서 직접 계산함
- 모델: 랜덤 포레스트(나무 200그루, random_state=0), 특징: 밴드 6개 + NDVI + NDWI
- 무작위 5겹: KFold(shuffle, 시드 0~4)
- 폴리곤 단위: GroupKFold(groups=patch_id, shuffle, 시드 0~4)
- 10 km 블록: 폴리곤 중심이 든 10 km 격자 칸을 블록으로 삼아 GroupKFold(shuffle, 시드 0~4)
- 새 폴리곤 시험: make_part3.py와 같은 방식으로 폴리곤을 새로 만들어(시드 101~103, 클래스별 폴리곤 수 5배)
  전체 자료로 학습한 모델을 시험함
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, StratifiedGroupKFold, cross_val_score

sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
d = pd.read_csv(os.path.join(HERE, "..", "data", "part3_landcover.csv"))
F = ["B2", "B3", "B4", "B8", "B11", "B12", "ndvi", "ndwi"]
X, y, xy = d[F].values, d.cls.values, d[["x_km", "y_km"]].values
rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
cen = d.groupby("patch_id")[["x_km", "y_km"]].transform("mean").values
blk10 = (cen[:, 0] // 10).astype(int) + 6 * (cen[:, 1] // 10).astype(int)
print("블록 경계에 걸친 폴리곤", int((d.groupby("patch_id").block.nunique() > 1).sum()),
      "옮긴 픽셀", int((blk10 != d.block.values).sum()), "블록 수", len(np.unique(blk10)))


def vtext(s, x, y, txt, size=11):
    s.text(x, y, txt, "f-mu", size)
    s.parts[-1] = s.parts[-1].replace("<text ", f'<text transform="rotate(-90 {x:.1f} {y:.1f})" ')


def poly(pts):
    return "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts)


# ---------------------------------------------------------------- 수치 계산
schemes = [
    ("무작위 5겹", lambda s: KFold(5, shuffle=True, random_state=s), None),
    ("폴리곤 단위", lambda s: GroupKFold(5, shuffle=True, random_state=s), d.patch_id.values),
    ("10 km 블록", lambda s: GroupKFold(5, shuffle=True, random_state=s), blk10),
]
acc = {}
for name, mk, g in schemes:
    acc[name] = np.array([cross_val_score(rf, X, y, cv=mk(s), groups=g).mean() for s in range(5)])
    print(name, acc[name].round(4), "평균", acc[name].mean().round(4))
    folds = cross_val_score(rf, X, y, cv=mk(0), groups=g)
    print("   시드 0 겹별", folds.round(3))

# 최근접 학습 픽셀까지 거리(시드 0 배정)
for name, mk, g in schemes:
    ds = []
    for tr, te in mk(0).split(X, y, g):
        ds.append(cKDTree(xy[tr]).query(xy[te])[0])
    print(name, "검증 픽셀 → 가장 가까운 학습 픽셀 거리 중앙값(km)", round(float(np.median(np.concatenate(ds))), 3))

# 층화: 겹마다 갯벌 픽셀 수
for name, cv, g in [("무작위", KFold(5, shuffle=True, random_state=0), None),
                    ("층화", StratifiedKFold(5, shuffle=True, random_state=0), None),
                    ("폴리곤(GroupKFold 기본)", GroupKFold(5), d.patch_id.values),
                    ("층화 폴리곤", StratifiedGroupKFold(5, shuffle=True, random_state=0), d.patch_id.values)]:
    print(name, "겹별 갯벌 픽셀", [int((y[te] == "갯벌").sum()) for _, te in cv.split(X, y, g)])

# 새 폴리곤으로 시험 (make_part3.py와 같은 생성 방식, 다른 시드)
CLASSES = {
    "산림": ([0.030, 0.050, 0.030, 0.330, 0.160, 0.070], 34),
    "농경지": ([0.050, 0.080, 0.065, 0.280, 0.220, 0.120], 30),
    "시가지": ([0.095, 0.105, 0.115, 0.210, 0.250, 0.215], 16),
    "수계": ([0.060, 0.050, 0.030, 0.020, 0.012, 0.008], 12),
    "나지": ([0.110, 0.130, 0.160, 0.240, 0.300, 0.260], 8),
    "갯벌": ([0.070, 0.075, 0.075, 0.100, 0.080, 0.055], 4),
}


def gen(seed, mult=5):
    rng = np.random.default_rng(seed)
    rows = []
    for cls, (mean, n) in CLASSES.items():
        mean = np.array(mean)
        for _ in range(n * mult):
            pm = mean * np.exp(rng.normal(0, 0.16, 6))
            if cls == "농경지" and rng.random() < 0.25:
                pm = 0.5 * pm + 0.5 * np.array(CLASSES["나지"][0])
            for _ in range(int(rng.integers(15, 45))):
                r = pm * np.exp(rng.normal(0, 0.07, 6))
                r = np.clip(r + rng.normal(0, 0.004, 6), 0.001, 0.8)
                rows.append([*r, (r[3] - r[2]) / (r[3] + r[2]), (r[1] - r[3]) / (r[1] + r[3]), cls])
    return pd.DataFrame(rows, columns=F + ["cls"])


full = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(X, y)
new_acc = []
for sd in (101, 102, 103):
    nd = gen(sd)
    new_acc.append((full.predict(nd[F].values) == nd.cls.values).mean())
    print("새 폴리곤 시드", sd, "픽셀", len(nd), "정확도", round(new_acc[-1], 4))
new_acc = np.array(new_acc)

# NDVI 코렐로그램 (거리 구간 가중치, 가중치 0·1)
v = d.ndvi.values
z = v - v.mean()
D = squareform(pdist(xy))
edges = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 12.5, 15]
cor = []
for a, b in zip(edges[:-1], edges[1:]):
    W = ((D > a) & (D <= b)).astype(float)
    np.fill_diagonal(W, 0)
    cor.append(((a + b) / 2, len(v) / W.sum() * (z @ W @ z) / (z @ z)))
    print(f"NDVI 모란 지수 {a}~{b} km: {cor[-1][1]:.3f}")
del D

# ---------------------------------------------------------------- 1. 분할 도식
s = Svg(680, 300, "무작위 픽셀 분할과 공간 블록 분할에서 학습·검증 픽셀이 지도 위에 놓이는 모습")
PW, PH, TOP = 300, 200, 40      # 지도 패널 크기
BS = 100                        # 블록 한 변(10 km)
rng = np.random.default_rng(5)
# 폴리곤: (블록 열, 블록 행, 블록 안 x, y) — 블록 경계에 걸치지 않게 배치
fields = [(0, 0, 14, 16), (0, 0, 52, 58), (0, 1, 30, 22), (1, 0, 18, 50), (1, 0, 60, 14),
          (1, 1, 40, 40), (2, 0, 34, 28), (2, 1, 12, 56), (2, 1, 58, 12), (0, 1, 60, 62)]
CELL, NX, NY = 7, 4, 3
val_blocks = {(1, 0), (0, 1)}
for k, title in enumerate(["무작위 분할 (픽셀 단위)", "공간 블록 분할 (10 km 블록)"]):
    ox = 20 + k * (PW + 40)
    s.text(ox + PW / 2, TOP - 14, title, "f-fg", 13, weight="600")
    if k == 1:
        for bx in range(3):
            for by in range(2):
                cls = "s-mu f-bds" if (bx, by) in val_blocks else "s-mu f-sf"
                s.rect(ox + bx * BS, TOP + by * BS, BS, BS, cls, 1)
    else:
        s.rect(ox, TOP, PW, PH, "s-mu f-sf", 1)
    for bx, by, fx, fy in fields:
        x0, y0 = ox + bx * BS + fx, TOP + by * BS + fy
        s.rect(x0 - 3, y0 - 3, NX * (CELL + 1) + 5, NY * (CELL + 1) + 5, "s-fg f-bg", 1.2, rx=3)
        for i in range(NX):
            for j in range(NY):
                if k == 0:
                    val = rng.random() < 0.2
                else:
                    val = (bx, by) in val_blocks
                s.rect(x0 + i * (CELL + 1), y0 + j * (CELL + 1), CELL, CELL, "f-bd" if val else "f-ac", 0)
# 범례
ly = TOP + PH + 32
for x, cls, lab in ((210, "f-ac", "학습 픽셀"), (330, "f-bd", "검증 픽셀"), (450, "s-fg f-bg", "현장 조사 폴리곤")):
    s.rect(x, ly - 10, 12, 12, cls, 1.2 if "s-" in cls else 0, rx=2 if "s-" in cls else 0)
    s.text(x + 18, ly, lab, "f-fg", 12, anchor="start")
s.save(os.path.join(OUT, "15-split-map.svg"))

# ---------------------------------------------------------------- 2. 교차검증 방식별 오류율
names = [n for n, _, _ in schemes]
err = {n: 1 - acc[n] for n in names}
s = Svg(680, 250, "교차검증 방식별 오류율(1 − 정확도)과 새 폴리곤 시험 오류율")
x0, TOP, PW, BH, GAP = 120, 30, 400, 34, 22
XMAX = 0.10
mx = lambda v: x0 + PW * v / XMAX
s.line(x0, TOP - 6, x0, TOP + 3 * (BH + GAP) - GAP + 6, "s-mu", 1.2)
axy = TOP + 3 * (BH + GAP) - GAP + 14
s.line(x0, axy, x0 + PW, axy, "s-mu", 1.2)
for t in (0, 0.02, 0.04, 0.06, 0.08, 0.10):
    s.line(mx(t), axy, mx(t), axy + 4, "s-mu", 1.2)
    s.text(mx(t), axy + 18, f"{t * 100:.0f}%", "f-mu", 11)
s.text(x0 + PW / 2, axy + 38, "교차검증 오류율 (1 − 정확도, 겹 배정 5가지의 평균과 범위)", "f-mu", 11)
ne = 1 - new_acc
s.rect(mx(ne.min()), TOP - 10, mx(ne.max()) - mx(ne.min()), 3 * (BH + GAP) - GAP + 20, "f-acs", 0)
for i, n in enumerate(names):
    yy = TOP + i * (BH + GAP)
    m = err[n].mean()
    s.rect(x0, yy, mx(m) - x0, BH, "f-bd" if i == 0 else "f-ac", 0)
    s.line(mx(err[n].min()), yy + BH / 2, mx(err[n].max()), yy + BH / 2, "s-fg", 1.5)
    for e in (err[n].min(), err[n].max()):
        s.line(mx(e), yy + BH / 2 - 6, mx(e), yy + BH / 2 + 6, "s-fg", 1.5)
    s.text(x0 - 10, yy + BH / 2 + 5, n, "f-fg", 13, anchor="end", weight="600")
    s.text(mx(err[n].max()) + 8, yy + BH / 2 + 5, f"{m * 100:.1f}% (정확도 {1 - m:.3f})", "f-fg", 12, anchor="start")
s.text(mx(ne.max()) + 6, TOP - 12, f"새 폴리곤 시험 {ne.min() * 100:.1f}~{ne.max() * 100:.1f}%", "f-ac", 11, anchor="start", weight="600")
s.save(os.path.join(OUT, "15-cv-error.svg"))

# ---------------------------------------------------------------- 3. NDVI 코렐로그램
s = Svg(680, 260, "거리 구간별 NDVI 모란 지수(코렐로그램)")
x0, y0, PW, PH, TOP = 70, 200, 580, 160, 40
XM = 15
YLO, YHI = -0.2, 1.0
mx = lambda v: x0 + PW * v / XM
my = lambda v: y0 - PH * (v - YLO) / (YHI - YLO)
s.line(x0, y0, x0 + PW, y0, "s-mu", 1.2)
s.line(x0, TOP, x0, y0, "s-mu", 1.2)
s.line(x0, my(0), x0 + PW, my(0), "s-mu", 1, "2 3")
for v_ in (-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0):
    s.line(x0 - 4, my(v_), x0, my(v_), "s-mu", 1.2)
    s.text(x0 - 8, my(v_) + 4, f"{v_:g}", "f-mu", 11, anchor="end")
for t in (0, 1, 2, 5, 10, 15):
    s.line(mx(t), y0, mx(t), y0 + 4, "s-mu", 1.2)
    s.text(mx(t), y0 + 17, f"{t}", "f-mu", 11)
s.text(x0 + PW / 2, y0 + 38, "두 픽셀 사이 거리 (km)", "f-mu", 11)
vtext(s, 20, TOP + PH / 2, "NDVI 모란 지수")
s.rect(mx(0), TOP, mx(1) - mx(0), PH, "f-bds", 0)
s.text(mx(1) + 6, TOP + 14, "같은 폴리곤 범위(약 1 km 안)", "f-bd", 11, anchor="start", weight="600")
s.line(mx(10), TOP, mx(10), y0, "s-ac", 1.5, "5 4")
s.text(mx(10) - 6, TOP + 14, "블록 한 변 10 km", "f-ac", 11, anchor="end", weight="600")
s.path(poly([(mx(a), my(b)) for a, b in cor]), "s-fg", 2)
for a, b in cor:
    s.circle(mx(a), my(b), 3.2, "f-fg")
s.save(os.path.join(OUT, "15-correlogram.svg"))
print("ok")
