"""19강 그림: 주성분 1·2 산점도(클래스별), 설명분산비율 막대(스크리 도표), 엘보·실루엣으로 k 고르기

자료: content/lessons/data/part3_landcover.csv (3부 공통 예제)
밴드 반사율 6개(B2, B3, B4, B8, B11, B12)를 표준화해 scikit-learn으로 계산함
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
d = pd.read_csv(os.path.join(HERE, "..", "data", "part3_landcover.csv"))
B = ["B2", "B3", "B4", "B8", "B11", "B12"]
Z = StandardScaler().fit_transform(d[B].values)
pca = PCA().fit(Z)
S = pca.transform(Z)
evr = pca.explained_variance_ratio_
# 주성분 부호는 도구마다 뒤집힐 수 있음. 1번은 밝을수록 +, 2번은 근적외가 클수록 +가 되게 맞춤
for j, band in ((0, "B4"), (1, "B8")):
    if pca.components_[j, B.index(band)] < 0:
        S[:, j] *= -1

# ---------------------------------------------------------------- 1. 주성분 산점도
STY = {  # 클래스: (점 모양 클래스, 글자 클래스)
    "산림": ("f-ok", "f-ok"), "농경지": ("f-mu", "f-mu"), "시가지": ("f-bd", "f-bd"),
    "나지": ("s-bd f-bg", "f-bd"), "수계": ("f-ac", "f-ac"), "갯벌": ("s-ac f-bg", "f-ac"),
}
# 글자 위치(주성분 좌표). 점구름 바깥쪽에 둠
LAB = {"산림": (-1.7, 2.35), "농경지": (0.9, 1.95), "시가지": (2.2, 1.45), "나지": (4.2, 0.75),
       "수계": (-2.45, -3.15), "갯벌": (0.15, -1.5)}
rng = np.random.default_rng(19)
s = Svg(680, 400, "표준화한 밴드 6개의 첫 두 주성분 산점도. 클래스마다 점구름이 따로 모이고, 가로축은 밝기, 세로축은 식생 방향임")
X0, X1, YT, YB = 80, 600, 20, 340
xlo, xhi, ylo, yhi = -3.5, 5.5, -3.5, 3.0
sx = lambda v: X0 + (X1 - X0) * (v - xlo) / (xhi - xlo)
sy = lambda v: YB - (YB - YT) * (v - ylo) / (yhi - ylo)
s.line(X0, YB, X1, YB, "s-mu", 1.2)
s.line(X0, YT, X0, YB, "s-mu", 1.2)
for v in (-2, 0, 2, 4):
    s.line(sx(v), YB, sx(v), YB + 4, "s-mu", 1.2)
    s.text(sx(v), YB + 18, f"{v}", "f-mu", 12)
for v in (-3, -2, -1, 0, 1, 2):
    s.line(X0 - 4, sy(v), X0, sy(v), "s-mu", 1.2)
    s.text(X0 - 8, sy(v) + 4, f"{v}", "f-mu", 12, anchor="end")
s.line(sx(0), YT, sx(0), YB, "s-mu", 0.8, "2 4")
s.line(X0, sy(0), X1, sy(0), "s-mu", 0.8, "2 4")
s.text((X0 + X1) / 2, YB + 38, f"주성분 1 (밝기, {evr[0] * 100:.1f}%)", "f-mu", 12)
s.text(X0 + 6, YT + 4, f"주성분 2 (식생, {evr[1] * 100:.1f}%)", "f-mu", 12, anchor="start")
for cls, (pcls, _) in STY.items():
    idx = np.flatnonzero(d.cls.values == cls)
    idx = rng.choice(idx, size=min(70, len(idx)), replace=False)
    for i in idx:
        s.circle(sx(S[i, 0]), sy(S[i, 1]), 2.6, pcls)
for cls, (x, y) in LAB.items():
    s.text(sx(x), sy(y), cls, STY[cls][1], 13, weight="700")
s.save(os.path.join(OUT, "19-pca-scatter.svg"))

# ---------------------------------------------------------------- 2. 스크리 도표
s = Svg(680, 300, "주성분 6개의 설명분산비율 막대와 누적 설명분산비율 꺾은선")
X0, X1, YT, YB = 70, 620, 30, 240
sy = lambda p: YB - (YB - YT) * p
bw, gap = 58, (X1 - X0) / 6
s.line(X0, YB, X1, YB, "s-mu", 1.2)
s.line(X0, YT, X0, YB, "s-mu", 1.2)
for p in (0, 0.25, 0.5, 0.75, 1.0):
    s.line(X0 - 4, sy(p), X0, sy(p), "s-mu", 1.2)
    s.text(X0 - 8, sy(p) + 4, f"{int(p * 100)}%", "f-mu", 12, anchor="end")
    if p > 0:
        s.line(X0, sy(p), X1, sy(p), "s-mu", 0.6, "2 4")
cum = np.cumsum(evr)
cx = [X0 + gap * (j + 0.5) for j in range(6)]
for j in range(6):
    s.rect(cx[j] - bw / 2, sy(evr[j]), bw, sy(0) - sy(evr[j]), "s-ac f-acs", 1.5)
    s.text(cx[j], YB + 18, f"PC{j + 1}", "f-mu", 12)
    # 막대 값: 큰 막대는 안쪽 위, 작은 막대는 위쪽
    if evr[j] > 0.15:
        s.text(cx[j], sy(evr[j]) + 18, f"{evr[j] * 100:.1f}%", "f-ac", 12, weight="600")
    else:
        s.text(cx[j], sy(evr[j]) - 6, f"{evr[j] * 100:.1f}%", "f-ac", 12, weight="600")
s.path("M " + " L ".join(f"{cx[j]:.1f} {sy(cum[j]):.1f}" for j in range(6)), "s-bd", 2)
for j in range(6):
    s.circle(cx[j], sy(cum[j]), 4, "f-bd")
for j in (1, 2):
    s.text(cx[j], sy(cum[j]) - 10, f"{cum[j] * 100:.1f}%", "f-bd", 12, weight="600")
s.text(cx[0] + 34, sy(cum[0]) + 4, "누적", "f-bd", 12, anchor="start", weight="600")
s.text((X0 + X1) / 2, YB + 40, "주성분 (설명하는 분산이 큰 순서)", "f-mu", 12)
s.save(os.path.join(OUT, "19-scree.svg"))

# ---------------------------------------------------------------- 3. 엘보와 실루엣
ks = list(range(2, 11))
inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=0).fit(Z)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(Z, km.labels_))

s = Svg(680, 290, "k를 2에서 10까지 바꾸며 잰 군집 내 제곱합(엘보 도표)과 평균 실루엣 계수")
PW, TOP, BOT = 250, 30, 220
panels = [
    ("군집 내 제곱합", inertia, 0, 9000, (0, 3000, 6000, 9000), lambda v: f"{v:,.0f}", 80),
    ("평균 실루엣 계수", sil, 0.25, 0.55, (0.3, 0.4, 0.5), lambda v: f"{v:.1f}", 400),
]
for title, vals, lo, hi, ticks, fmt, ox in panels:
    mx = lambda k: ox + PW * (k - 2) / 8
    my = lambda v: BOT - (BOT - TOP) * (v - lo) / (hi - lo)
    s.line(ox, BOT, ox + PW, BOT, "s-mu", 1.2)
    s.line(ox, TOP, ox, BOT, "s-mu", 1.2)
    for t in ticks:
        s.line(ox - 4, my(t), ox, my(t), "s-mu", 1.2)
        s.text(ox - 8, my(t) + 4, fmt(t), "f-mu", 11, anchor="end")
    for k in ks:
        s.line(mx(k), BOT, mx(k), BOT + 4, "s-mu", 1.2)
        s.text(mx(k), BOT + 17, f"{k}", "f-mu", 11)
    s.text(ox + PW / 2, BOT + 36, "군집 수 k", "f-mu", 12)
    s.text(ox + PW / 2, TOP - 12, title, "f-fg", 13, weight="600")
    for k in (3, 6):
        s.line(mx(k), TOP, mx(k), BOT, "s-bd" if k == 3 else "s-mu", 1, "4 4")
    s.path("M " + " L ".join(f"{mx(k):.1f} {my(v):.1f}" for k, v in zip(ks, vals)), "s-ac", 2)
    for k, v in zip(ks, vals):
        s.circle(mx(k), my(v), 3.5, "f-bd" if k == 3 else "f-ac")
# 주석: 오른쪽 위 빈 곳
s.text(80 + PW * (3 - 2) / 8 + 6, 60, "k = 3", "f-bd", 12, anchor="start", weight="600")
s.text(80 + PW * (6 - 2) / 8 + 6, 60, "k = 6 (실제 클래스 수)", "f-mu", 11, anchor="start")
s.save(os.path.join(OUT, "19-elbow.svg"))
print("evr", np.round(evr, 4), "cum", np.round(cum, 4))
print("inertia", np.round(inertia, 1))
print("sil", np.round(sil, 3))
