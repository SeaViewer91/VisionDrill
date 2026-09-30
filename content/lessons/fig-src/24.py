"""24강 그림과 본문 수치: 캐니 단계(PNG), ORB 매칭과 RANSAC(PNG), RANSAC 원리 도식(SVG)

자료: content/lessons/data/part4_scene.npz (4부 공통 예제)
- 영상 A: 근적외(B8) 반사율 0~0.45를 8비트 0~255로 옮긴 뒤 가운데 210×210 화소를 자름
- 영상 B: 장면 중심을 기준으로 반시계 8° 회전 + 오른쪽 10화소·위쪽 6화소 이동한 뒤 같은 자리를 자르고,
  밝기를 0.75배 + 25로 바꾸고 잡음(표준편차 3)을 더함 (시기가 다른 영상 흉내)
- 좌표는 영상 좌표(x = 열, 오른쪽이 +, y = 행, 아래가 +, 원점 왼쪽 위)
실행하면 본문에 쓴 수치를 출력함
"""
import os, sys
import numpy as np
import cv2
from scipy import ndimage as ndi
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
d = np.load(os.path.join(HERE, "..", "data", "part4_scene.npz"))
refl = d["refl"].astype(np.float32)
H = W = 300
full = np.clip(np.round(refl[3] / 0.45 * 255), 0, 255)          # 근적외 8비트 (float로 보관)


def to8(a):
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def label(img, s):
    """패널 왼쪽 위에 짧은 영문 표식"""
    cv2.rectangle(img, (4, 4), (24, 26), 255, -1)
    cv2.putText(img, s, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2, cv2.LINE_AA)
    return img


# ============================================================ 1. 소벨 수치 예
patch = np.array([[10, 80, 150]] * 3, float)
kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
gx = (patch * kx).sum()
print("[소벨] 3x3 예: Gx =", gx, " Gx/8 =", gx / 8)

# ============================================================ 2. 캐니 단계 (직접 구현)
SIG = 1.5
g = ndi.gaussian_filter(full, SIG)
gx = ndi.sobel(g, axis=1) / 8            # 화소당 밝기 변화(밝기/화소)
gy = ndi.sobel(g, axis=0) / 8
mag = np.hypot(gx, gy)
ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
# 비최대 억제: 기울기 방향(4갈래)의 앞뒤 이웃보다 작으면 지움
nms = np.zeros_like(mag)
q = np.round(ang / 45).astype(int) % 4
off = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}      # (행, 열) 이웃 방향
for k, (dr, dc) in off.items():
    a = np.roll(mag, (dr, dc), (0, 1))
    b = np.roll(mag, (-dr, -dc), (0, 1))
    keep = (q == k) & (mag >= a) & (mag >= b)
    nms[keep] = mag[keep]
nms[:2, :] = nms[-2:, :] = 0
nms[:, :2] = nms[:, -2:] = 0
LO, HI = 2.0, 5.0
low_only = nms >= LO
high_only = nms >= HI
lab, n = ndi.label(low_only, structure=np.ones((3, 3)))
keep_lab = np.unique(lab[high_only])
hyst = np.isin(lab, keep_lab[keep_lab > 0])
thin = nms > 0.5
print(f"[캐니] 기울기>0.5 화소 {int((mag > 0.5).sum())}, 비최대 억제 후 {int(thin.sum())}")
print(f"[캐니] 낮은 임계 {LO}만: {int(low_only.sum())}  높은 임계 {HI}만: {int(high_only.sum())}  히스테리시스: {int(hyst.sum())}")
nlo = ndi.label(low_only, structure=np.ones((3, 3)))[1]
nhi = ndi.label(high_only, structure=np.ones((3, 3)))[1]
nhy = ndi.label(hyst, structure=np.ones((3, 3)))[1]
print(f"[캐니] 연결요소 수: 낮은 {nlo}, 높은 {nhi}, 히스테리시스 {nhy}")
# 바다(해안에서 먼 서쪽) 잡음 기울기
sea = d["cls"] == 0
sea_in = ndi.binary_erosion(sea, iterations=6)
print(f"[캐니] 바다 안쪽 기울기 크기 99백분위 {np.percentile(mag[sea_in], 99):.2f}, 최대 {mag[sea_in].max():.2f}")
# scikit-image와 비교(검산)
from skimage.feature import canny
sk = canny(full / 255.0, sigma=SIG, low_threshold=LO * 8 / 255, high_threshold=HI * 8 / 255)
print(f"[캐니] scikit-image canny 화소 {int(sk.sum())}, 직접 구현과 일치 {(sk == hyst).mean() * 100:.1f}%")

GAP = 6
panels = [
    to8(full),
    to8(np.clip(mag / 12 * 255, 0, 255)),
    to8(np.clip(nms / 12 * 255, 0, 255) * (nms > 0.5)),
    to8(low_only * 255),
    to8(high_only * 255),
    to8(hyst * 255),
]
for p, s in zip(panels, "abcdef"):
    label(p, s)
canvas = np.full((2 * H + GAP, 3 * W + 2 * GAP), 255, np.uint8)
for i, p in enumerate(panels):
    r, c = divmod(i, 3)
    canvas[r * (H + GAP):r * (H + GAP) + H, c * (W + GAP):c * (W + GAP) + W] = p
Image.fromarray(canvas).save(os.path.join(OUT, "24-canny.png"), optimize=True)

# ============================================================ 3. 해리스: 구조 텐서 고윳값
g1 = ndi.gaussian_filter(full, 1.0)
ix = ndi.sobel(g1, axis=1) / 8
iy = ndi.sobel(g1, axis=0) / 8
WS = 1.5
Sxx = ndi.gaussian_filter(ix * ix, WS)
Syy = ndi.gaussian_filter(iy * iy, WS)
Sxy = ndi.gaussian_filter(ix * iy, WS)
tr = Sxx + Syy
det = Sxx * Syy - Sxy ** 2
disc = np.sqrt(np.maximum((tr / 2) ** 2 - det, 0))
l1, l2 = tr / 2 + disc, tr / 2 - disc
KH = 0.05
R = det - KH * tr ** 2
for name, (r, c) in [("평탄(바다)", (150, 20)), ("엣지(필지 경계)", (180, 180)), ("코너(필지 모서리)", (240, 150))]:
    print(f"[해리스] {name} 행{r} 열{c}: λ1={l1[r, c]:.1f} λ2={l2[r, c]:.1f} R={R[r, c]:.0f}")

# ============================================================ 4. 두 번째 영상과 ORB 매칭·RANSAC
ANG, TX, TY = 8.0, 10.0, -6.0
M = cv2.getRotationMatrix2D((150, 150), ANG, 1.0)
M[0, 2] += TX
M[1, 2] += TY
warped = cv2.warpAffine(full.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
MG = 45
rng = np.random.default_rng(24)
A8 = to8(full[MG:H - MG, MG:W - MG])
B8 = to8(warped[MG:H - MG, MG:W - MG] * 0.75 + 25 + rng.normal(0, 3, (H - 2 * MG, W - 2 * MG)))
T = np.eye(3); T[:2] = M
S = np.eye(3); S[:2, 2] = MG
Tc = np.linalg.inv(S) @ T @ S                         # 자른 영상 A 좌표 → B 좌표 (참 변환)
print("[정합] 참 변환(A→B):", np.round(Tc[:2], 4).tolist())

orb = cv2.ORB_create(nfeatures=500)
ka, da = orb.detectAndCompute(A8, None)
kb, db = orb.detectAndCompute(B8, None)
pa = np.array([k.pt for k in ka])
pb = np.array([k.pt for k in kb])
print(f"[ORB] 특징점 A {len(ka)}, B {len(kb)}, 기술자 {da.shape[1] * 8}비트")
knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)


def true_err(ms):
    p = (Tc @ np.c_[pa[[m.queryIdx for m in ms]], np.ones(len(ms))].T)[:2].T
    return np.hypot(*(p - pb[[m.trainIdx for m in ms]]).T)


nn = [x[0] for x in knn]
e_nn = true_err(nn)
print(f"[매칭] 최근접만: {len(nn)}쌍, 맞는 쌍(참 위치 3화소 안) {int((e_nn < 3).sum())} ({(e_nn < 3).mean() * 100:.1f}%)")
print("[매칭] 최근접 해밍 거리 중앙값: 맞는 쌍", np.median([m.distance for m, e in zip(nn, e_nn) if e < 3]),
      "틀린 쌍", np.median([m.distance for m, e in zip(nn, e_nn) if e >= 3]))
for rt in (0.7, 0.8, 0.9):
    gd = [x[0] for x in knn if len(x) == 2 and x[0].distance < rt * x[1].distance]
    e = true_err(gd)
    print(f"[매칭] 비율 검정 {rt}: {len(gd)}쌍, 맞는 쌍 {int((e < 3).sum())} ({(e < 3).mean() * 100:.1f}%)")
ratio = [x[0] for x in knn if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
e_ratio = true_err(ratio)

gyy, gxx = np.mgrid[0:210:10, 0:210:10]
Gp = np.stack([gxx.ravel(), gyy.ravel(), np.ones(gxx.size)])
Gt = (Tc @ Gp)[:2]


def evalM(Me, name):
    Me3 = np.eye(3); Me3[:2] = Me
    dd = np.hypot(*((Me3 @ Gp)[:2] - Gt))
    a = np.degrees(np.arctan2(-Me[1, 0], Me[0, 0]))
    sc = np.hypot(Me[0, 0], Me[1, 0])
    print(f"[정합] {name}: 회전 {a:.2f}°, 배율 {sc:.4f}, 격자점 {Gp.shape[1]}개 위치 오차 RMSE {np.sqrt((dd ** 2).mean()):.2f}화소 최대 {dd.max():.2f}")


def lsq_similarity(src, dst):
    A_, b_ = [], []
    for (x, y), (u, v) in zip(src, dst):
        A_ += [[x, -y, 1, 0], [y, x, 0, 1]]
        b_ += [u, v]
    a, b, c, e = np.linalg.lstsq(np.array(A_), np.array(b_), rcond=None)[0]
    return np.array([[a, -b, c], [b, a, e]])


results = {}
for name, ms, ecorr in (("최근접 전부", nn, e_nn < 3), ("비율 검정 0.8", ratio, e_ratio < 3)):
    src = np.float32([pa[m.queryIdx] for m in ms])
    dst = np.float32([pb[m.trainIdx] for m in ms])
    evalM(lsq_similarity(src, dst), name + " 최소제곱(RANSAC 없음)")
    cv2.setRNGSeed(0)
    Mr, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    inl = inl.ravel().astype(bool)
    print(f"[정합] {name} RANSAC 인라이어 {int(inl.sum())}/{len(ms)} ({inl.mean() * 100:.1f}%), 그중 실제로 맞는 쌍 {int((inl & ecorr).sum())}")
    evalM(Mr, name + " RANSAC")
    results[name] = (ms, inl)

# 필요한 반복 횟수
w_all = results["최근접 전부"][1].mean()
for w in (w_all, 0.5, 0.3):
    for s_ in (2, 3, 4):
        N = np.log(1 - 0.99) / np.log(1 - w ** s_)
        print(f"[반복] w={w:.3f} s={s_}: N={N:.2f} → {int(np.ceil(N))}")

# 광학(근적외) ↔ SAR: 같은 기하변환, 센서만 다름
for nm in ("sar_clean",):
    sdb = 10 * np.log10(np.maximum(d[nm].astype(np.float32), 1e-4))
    lo, hi = np.percentile(sdb, [1, 99])
    simg = np.clip((sdb - lo) / (hi - lo) * 255, 0, 255).astype(np.float32)
    Bs = to8(cv2.warpAffine(simg, M, (W, H), flags=cv2.INTER_LINEAR)[MG:H - MG, MG:W - MG])
    kbs, dbs = orb.detectAndCompute(Bs, None)
    pbs = np.array([k.pt for k in kbs])
    kn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, dbs, k=2)
    nn_s = [x[0] for x in kn]
    p = (Tc @ np.c_[pa[[m.queryIdx for m in nn_s]], np.ones(len(nn_s))].T)[:2].T
    es = np.hypot(*(p - pbs[[m.trainIdx for m in nn_s]]).T)
    gd = [x[0] for x in kn if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
    pg = (Tc @ np.c_[pa[[m.queryIdx for m in gd]], np.ones(len(gd))].T)[:2].T
    eg = np.hypot(*(pg - pbs[[m.trainIdx for m in gd]]).T)
    print(f"[광학-SAR] {nm}: SAR 특징점 {len(kbs)}, 최근접 {len(nn_s)}쌍 중 맞는 쌍 {int((es < 3).sum())}, 비율 검정 {len(gd)}쌍 중 맞는 쌍 {int((eg < 3).sum())}")
    src = np.float32([pa[m.queryIdx] for m in nn_s]); dst = np.float32([pbs[m.trainIdx] for m in nn_s])
    cv2.setRNGSeed(0)
    Ms, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    print(f"[광학-SAR] RANSAC 인라이어 {int(inl.sum())}/{len(nn_s)}")
    evalM(Ms, "광학-SAR RANSAC")

# ------------------------------------------------------------ 매칭 선 그림
SC = 2
ms, inl = results["비율 검정 0.8"]
Au = cv2.resize(A8, None, fx=SC, fy=SC, interpolation=cv2.INTER_NEAREST)
Bu = cv2.resize(B8, None, fx=SC, fy=SC, interpolation=cv2.INTER_NEAREST)
GAPM = 24
hh, ww = Au.shape
can = np.full((hh, 2 * ww + GAPM), 255, np.uint8)
can[:, :ww] = Au
can[:, ww + GAPM:] = Bu
can = cv2.cvtColor(can, cv2.COLOR_GRAY2RGB)
GREEN, RED = (40, 200, 90), (255, 70, 30)
# 인라이어를 먼저, 아웃라이어를 나중에(위에) 그림
for i in list(np.flatnonzero(inl)) + list(np.flatnonzero(~inl)):
    m = ms[i]
    p1 = tuple(int(round(v * SC + SC / 2)) for v in pa[m.queryIdx])
    p2 = tuple(int(round(v * SC + SC / 2)) for v in pb[m.trainIdx])
    p2 = (p2[0] + ww + GAPM, p2[1])
    col = GREEN if inl[i] else RED
    cv2.line(can, p1, p2, col, 2 if not inl[i] else 1, cv2.LINE_AA)
    cv2.circle(can, p1, 4, col, 2, cv2.LINE_AA)
    cv2.circle(can, p2, 4, col, 2, cv2.LINE_AA)
for x0, s in ((0, "a"), (ww + GAPM, "b")):
    cv2.rectangle(can, (x0 + 4, 4), (x0 + 24, 26), (255, 255, 255), -1)
    cv2.putText(can, s, (x0 + 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
Image.fromarray(can).quantize(colors=128, method=Image.Quantize.MEDIANCUT).save(
    os.path.join(OUT, "24-matches.png"), optimize=True)

# ============================================================ 5. RANSAC 원리 도식 (SVG)
rng = np.random.default_rng(7)
xs_in = np.linspace(0.6, 9.4, 14) + rng.normal(0, 0.12, 14)
ys_in = 0.55 * xs_in + 1.0 + rng.normal(0, 0.18, 14)
xs_out = np.array([1.2, 2.3, 3.6, 5.2, 6.4, 7.8, 8.9, 4.4])
ys_out = np.array([5.9, 0.2, 6.6, 7.4, 1.0, 1.6, 8.5, 0.9])
X = np.r_[xs_in, xs_out]
Y = np.r_[ys_in, ys_out]
TH = 0.6            # 허용 거리(인라이어 판정)


def count_inl(i, j):
    x1, y1, x2, y2 = X[i], Y[i], X[j], Y[j]
    a, b, c = y2 - y1, -(x2 - x1), (x2 - x1) * y1 - (y2 - y1) * x1
    dist = np.abs(a * X + b * Y + c) / np.hypot(a, b)
    return dist <= TH


bad = (14 + 1, 14 + 5)          # 아웃라이어 두 점을 뽑은 경우 (2.3,0.2)-(6.4,1.0)
good = (2, 11)                  # 인라이어 두 점
nb, ng = count_inl(*bad).sum(), count_inl(*good).sum()
print(f"[RANSAC 도식] 나쁜 표본 인라이어 {nb}, 좋은 표본 인라이어 {ng}, 전체 {len(X)}")

sv = Svg(680, 300, "RANSAC 원리: 두 점을 무작위로 뽑아 직선을 세우고 허용 거리 띠 안의 점 수를 셈. 인라이어가 가장 많은 후보를 고름")
PW, X0s, TOP, BOT = 280, (50, 370), 40, 250
for (i, j), ox, title in ((bad, X0s[0], f"후보 1: 띠 안 {nb}점"), (good, X0s[1], f"후보 2: 띠 안 {ng}점")):
    mx = lambda v: ox + PW * v / 10
    my = lambda v: BOT - (BOT - TOP) * v / 9
    sv.rect(ox, TOP, PW, BOT - TOP, "s-mu f-bg", 1)
    # 후보 직선과 띠
    x1, y1, x2, y2 = X[i], Y[i], X[j], Y[j]
    slope = (y2 - y1) / (x2 - x1)
    off = TH * np.hypot(1, slope)
    xa, xb = 0.0, 10.0
    ya, yb = y1 + slope * (xa - x1), y1 + slope * (xb - x1)
    # 띠(사각형 경로), 판 안쪽으로 자름
    def clipy(v):
        return min(max(v, 0), 9)
    poly = [(xa, ya + off), (xb, yb + off), (xb, yb - off), (xa, ya - off)]
    sv.path("M " + " L ".join(f"{mx(px):.1f} {my(clipy(py)):.1f}" for px, py in poly) + " Z", "f-acs", 0)
    sv.line(mx(xa), my(clipy(ya)), mx(xb), my(clipy(yb)), "s-ac", 2)
    inside = count_inl(i, j)
    for k in range(len(X)):
        if k in (i, j):
            continue
        sv.circle(mx(X[k]), my(Y[k]), 4.2, "f-ac" if inside[k] else "s-mu f-bg")
    for k in (i, j):
        sv.circle(mx(X[k]), my(Y[k]), 7, "f-bd")
    sv.text(ox + PW / 2, TOP - 14, title, "f-fg", 13, weight="600")
# 범례
LY = 278
sv.circle(60, LY - 4, 7, "f-bd"); sv.text(74, LY, "무작위로 뽑은 2점", "f-fg", 12, anchor="start")
sv.circle(230, LY - 4, 4.2, "f-ac"); sv.text(242, LY, "띠 안(인라이어)", "f-fg", 12, anchor="start")
sv.circle(372, LY - 4, 4.2, "s-mu f-bg"); sv.text(384, LY, "띠 밖", "f-fg", 12, anchor="start")
sv.rect(452, LY - 11, 22, 12, "f-acs", 0); sv.text(482, LY, "허용 거리 띠", "f-fg", 12, anchor="start")
sv.save(os.path.join(OUT, "24-ransac.svg"))

for f in ("24-canny.png", "24-matches.png"):
    p = os.path.join(OUT, f)
    im = Image.open(p)
    print(f, im.size, im.mode, f"{os.path.getsize(p) / 1024:.0f} KB")
