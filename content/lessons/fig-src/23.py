"""23강 그림: 오츠 이진화 히스토그램, 형태학 연산 원리(격자), 수계 마스크 처리 단계(PNG)

자료: content/lessons/data/part4_scene.npz (4부 공통 예제, 300×300 화소, 10 m)
- NDWI = (B3 - B8) / (B3 + B8)  (녹색·근적외, 맥피터스 방식)
- 참 수계 = cls 0(바다) + 1(하천)
- SAR 수계 마스크: 스페클 강도 → 3×3 평균 필터 → dB → 오츠 → 열림 3×3 → 닫힘 3×3
  → 50화소(0.5 ha) 미만 조각 제거 → 구멍 메우기. 가장자리는 가장 가까운 값으로 채운 뒤(edge 패딩) 연산
- 하천 분리: NDWI 마스크(구멍 메움)에서 11×11 열림으로 바다만 남기고 빼서 가장 큰 조각 → 골격화
- 선박 세기: SAR 1-look 강도 > 0.3 이면서 바다(구멍 메운 NDWI 수계) 안
본문 수치는 이 스크립트 끝의 print로 확인함
"""
import os, sys
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
d = np.load(os.path.join(HERE, "..", "data", "part4_scene.npz"))
refl = d["refl"].astype(np.float32)
cls = d["cls"]
sar = d["sar"].astype(np.float32)
truth = (cls == 0) | (cls == 1)
E8 = np.ones((3, 3), bool)
SQ3 = np.ones((3, 3), bool)


def pad_op(fn, w, se, p=15):
    """가장자리를 가장 가까운 값으로 채운 뒤 연산하고 잘라 냄 (바깥을 0으로 보는 기본값 회피)"""
    return fn(np.pad(w, p, mode="edge"), se)[p:-p, p:-p]


def score(w):
    tp = (w & truth).sum(); fp = (w & ~truth).sum(); fn = (~w & truth).sum()
    return dict(OA=round(float((w == truth).mean()), 4), IoU=round(float(tp / (tp + fp + fn)), 3), FP=int(fp), FN=int(fn),
                water8=ndi.label(w, structure=E8)[1], water4=ndi.label(w)[1], land8=ndi.label(~w, structure=E8)[1],
                flat=int(w[cls == 7].sum()), crop=int(w[cls == 3].sum()))


# ================================================================ 계산
g, n = refl[1], refl[3]
ndwi = (g - n) / (g + n)
t_nd = threshold_otsu(ndwi)
w_nd = ndwi > t_nd


def otsu_parts(x, t):
    a, b = x[x <= t], x[x > t]
    w0 = a.size / x.size
    return w0, 1 - w0, a.mean(), b.mean(), w0 * (1 - w0) * (a.mean() - b.mean()) ** 2, x.var()


half = ndwi[:, 150:]
t_half = threshold_otsu(half)
w_half = half > t_half
tr_half = truth[:, 150:]

# SAR 수계 마스크 단계
db_raw = 10 * np.log10(np.maximum(sar, 1e-6))
w_raw = db_raw < threshold_otsu(db_raw)
db = 10 * np.log10(ndi.uniform_filter(sar, 3))
t_sar = threshold_otsu(db)
s0 = db < t_sar
s1 = pad_op(ndi.binary_opening, s0, SQ3)
s2 = pad_op(ndi.binary_closing, s1, SQ3)
lab, nl = ndi.label(s2, structure=E8)
sz = np.bincount(lab.ravel()); keep = sz >= 50; keep[0] = False
s3 = keep[lab]
holes = ndi.binary_fill_holes(s3) & ~s3
s4 = s3 | holes
s2_nopad = ndi.binary_closing(s1, SQ3)          # 바깥을 0으로 보는 기본값

# 하천 분리와 골격화
w_fill = ndi.binary_fill_holes(w_nd)
sea_only = pad_op(ndi.binary_opening, w_fill, np.ones((11, 11), bool), 30)
rv = w_fill & ~sea_only
lab_r, _ = ndi.label(rv, structure=E8)
rv = lab_r == (np.argmax(np.bincount(lab_r.ravel())[1:]) + 1)
sk = skeletonize(rv)
dt = ndi.distance_transform_edt(rv)
# 골격 길이: 끝점에서 따라가며 가로·세로 1, 대각선 √2
S = set(zip(*np.nonzero(sk)))
nb = lambda p: [(p[0] + a, p[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b) and (p[0] + a, p[1] + b) in S]
ends = [p for p in S if len(nb(p)) == 1]
cur, seen, orth, diag = ends[0], {ends[0]}, 0, 0
while True:
    cand = sorted([q for q in nb(cur) if q not in seen], key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
    if not cand:
        break
    q = cand[0]
    if abs(q[0] - cur[0]) + abs(q[1] - cur[1]) == 1: orth += 1
    else: diag += 1
    seen.add(q); cur = q
xx = np.linspace(0, 299, 30000); yy = 120 + 25 * np.sin(xx / 45)
coast = 70 + 12 * np.sin(yy / 37) + 6 * np.sin(yy / 11)
m = xx >= coast
true_len = np.sum(np.hypot(np.diff(xx[m]), np.diff(yy[m])))

# 선박 세기
ship_b = (sar > 0.3) & w_fill

# ================================================================ 1. 오츠 히스토그램 (SVG)
s = Svg(680, 250, "NDWI 히스토그램과 오츠 임계값. 장면 전체는 물과 뭍의 두 봉우리 사이에 임계값이 놓이지만, 물이 2.7%뿐인 오른쪽 절반에서는 뭍 안의 산림·농경지 대부분과 시가지 등 사이에 놓임")
bins = np.arange(-0.86, 0.62, 0.02)


def between_curve(x, ts):
    x = np.sort(x.ravel()); N = x.size; cs = np.cumsum(x); tot = cs[-1]
    out = []
    for t in ts:
        k = np.searchsorted(x, t, side="right")
        if k == 0 or k == N:
            out.append(0.0); continue
        w0 = k / N; m0 = cs[k - 1] / k; m1 = (tot - cs[k - 1]) / (N - k)
        out.append(w0 * (1 - w0) * (m0 - m1) ** 2)
    return np.array(out)


def hist_panel(x, t, X0, title, labels):
    W, YT, YB = 290, 40, 200
    xlo, xhi = -0.86, 0.60
    sx = lambda v: X0 + W * (v - xlo) / (xhi - xlo)
    h, _ = np.histogram(x, bins)
    hy = h / h.max()
    for i, v in enumerate(hy):
        if h[i] == 0:
            continue
        top = YB - (YB - YT) * v
        s.rect(sx(bins[i]), top, sx(bins[i + 1]) - sx(bins[i]), YB - top, "f-acs s-ac", 0.6)
    ts = np.linspace(xlo, xhi, 300)
    bc = between_curve(x, ts); bc = bc / bc.max()
    pts = " L ".join(f"{sx(a):.1f} {YB - (YB - YT) * b:.1f}" for a, b in zip(ts, bc))
    s.path("M " + pts, "s-fg", 1.6)
    s.line(sx(t), YT - 6, sx(t), YB, "s-bd", 2, "5 3")
    s.line(X0, YB, X0 + W, YB, "s-mu", 1.2)
    for v in (-0.8, -0.4, 0.0, 0.4):
        s.line(sx(v), YB, sx(v), YB + 4, "s-mu", 1.2)
        s.text(sx(v), YB + 17, f"{v:.1f}".replace("-", "−"), "f-mu", 11)
    s.text(X0 + W / 2, YB + 36, "NDWI", "f-mu", 12)
    s.text(X0 + W / 2, 18, title, "f-fg", 13, weight="600")
    tl = f"{t:.2f}".replace("-", "−")
    s.text(sx(t) + 5, YT + 2, f"임계값 {tl}", "f-bd", 12, anchor="start", weight="600")
    for (lx, ly, txt, cl) in labels:
        s.text(sx(lx), ly, txt, cl, 12)


hist_panel(ndwi, t_nd, 30, "장면 전체 (물 26%)", [(-0.62, 72, "뭍", "f-mu"), (0.40, 110, "물", "f-ac")])
hist_panel(half, t_half, 370, "오른쪽 절반 (물 2.7%)", [(-0.74, 33, "산림", "f-mu"), (-0.62, 150, "농경지", "f-mu"), (-0.26, 150, "시가지 등", "f-mu"), (0.42, 186, "하천", "f-ac")])
s.save(os.path.join(OUT, "23-otsu.svg"))

# ================================================================ 2. 형태학 연산 원리 (SVG)
R, C = 10, 13
A = np.zeros((R, C), bool)
A[2:8, 1:8] = True          # 본체 6×7
A[4, 4] = False             # 1화소 구멍
A[2, 3] = False             # 위쪽 가장자리 홈
A[5, 8:11] = True           # 1화소 폭 돌기
A[8, 11] = True             # 외딴 잡음 화소
ops = [("원본", A),
       ("침식", ndi.binary_erosion(A, SQ3)),
       ("팽창", ndi.binary_dilation(A, SQ3)),
       ("열림 (침식→팽창)", ndi.binary_opening(A, SQ3)),
       ("닫힘 (팽창→침식)", ndi.binary_closing(np.pad(A, 3), SQ3)[3:-3, 3:-3])]

s = Svg(680, 292, "3×3 구조요소로 한 침식, 팽창, 열림, 닫힘 결과와 4-연결, 8-연결 이웃 및 같은 모양의 연결요소 개수 비교")
CS = 9.5
GX, GY = 12, 34
PW = C * CS
GAP = (680 - 2 * GX - 5 * PW) / 4
for k, (name, B) in enumerate(ops):
    x0 = GX + k * (PW + GAP)
    s.text(x0 + PW / 2, GY - 12, name, "f-fg", 12, weight="600")
    for i in range(R):
        for j in range(C):
            a, b = A[i, j], B[i, j]
            if k == 0:
                c = "s-mu f-ac" if a else "s-mu f-bg"
            elif a and b:
                c = "s-mu f-ac"
            elif b and not a:
                c = "s-mu f-bd"
            elif a and not b:
                c = "s-mu f-acs"
            else:
                c = "s-mu f-bg"
            s.rect(x0 + j * CS, GY + i * CS, CS, CS, c, 0.4)
# 범례
ly = GY + R * CS + 22
for i, (c, txt) in enumerate((("s-mu f-ac", "남은 화소"), ("s-mu f-bd", "새로 생긴 화소"), ("s-mu f-acs", "사라진 화소"))):
    lx = 150 + i * 150
    s.rect(lx, ly - 10, 12, 12, c, 0.6)
    s.text(lx + 18, ly, txt, "f-mu", 12, anchor="start")

# 아래 줄: 4-연결 / 8-연결 이웃과 연결요소
Y2 = ly + 34
CS2 = 16


def nbr(x0, kind, title):
    s.text(x0 + 1.5 * CS2, Y2 - 8, title, "f-fg", 12, weight="600")
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                c = "s-mu f-fg"
            elif kind == 8 or (i == 1 or j == 1):
                c = "s-mu f-acs"
            else:
                c = "s-mu f-bg"
            s.rect(x0 + j * CS2, Y2 + 8 + i * CS2, CS2, CS2, c, 0.6)


nbr(40, 4, "4-연결 이웃")
nbr(150, 8, "8-연결 이웃")
Bshape = np.zeros((6, 9), bool)
for (i, j) in [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 7), (5, 8)]:
    Bshape[i, j] = True
l4, n4 = ndi.label(Bshape)
l8, n8 = ndi.label(Bshape, structure=E8)
for k, (lab_, nn, title) in enumerate(((l4, n4, f"4-연결로 세면 {n4}개"), (l8, n8, f"8-연결로 세면 {n8}개"))):
    x0 = 290 + k * 200
    s.text(x0 + 4.5 * CS2, Y2 - 8, title, "f-fg", 12, weight="600")
    for i in range(6):
        for j in range(9):
            v = lab_[i, j]
            c = ("s-mu f-ac" if v % 2 else "s-mu f-bd") if v else "s-mu f-bg"
            s.rect(x0 + j * CS2, Y2 + i * CS2 - 4, CS2, CS2, c, 0.6)
            if v and nn > 1:
                s.text(x0 + j * CS2 + CS2 / 2, Y2 + i * CS2 + 8, str(v), "f-bg", 10, weight="600")
s.save(os.path.join(OUT, "23-morph.svg"))

# ================================================================ 3. 수계 마스크 처리 단계 (PNG, 글자 없음)
def mask_img(w, water=40, land=225):
    return np.where(w, water, land).astype(np.uint8)


dbv = np.clip((db + 25) / 25 * 255, 0, 255).astype(np.uint8)       # −25~0 dB를 0~255로
err = np.full(truth.shape, 225, np.uint8)
err[s4 & ~truth] = 20            # 없는 물을 물로 (오탐)
err[~s4 & truth] = 130           # 있는 물을 놓침 (미탐)
riv = mask_img(w_fill, water=150)
riv[sk] = 0
panels = [dbv, mask_img(s0), mask_img(s1), mask_img(s4), err, riv]
Hh, Ww, G = 300, 300, 10
canvas = np.full((2 * Hh + G, 3 * Ww + 2 * G), 255, np.uint8)
for k, p in enumerate(panels):
    r_, c_ = divmod(k, 3)
    canvas[r_ * (Hh + G):r_ * (Hh + G) + Hh, c_ * (Ww + G):c_ * (Ww + G) + Ww] = p
Image.fromarray(canvas, "L").save(os.path.join(OUT, "23-water-steps.png"), optimize=True)

# ================================================================ 본문 수치
print("NDWI otsu t", round(float(t_nd), 4), "OA", score(w_nd)["OA"], "wrong px", int((w_nd != truth).sum()),
      "wrong cls", np.unique(cls[w_nd != truth]))
print("otsu parts", [round(float(v), 4) for v in otsu_parts(ndwi.ravel(), t_nd)])
print("NDWI class means", {k: round(float(ndwi[cls == k].mean()), 2) for k in range(9)})
print("NDWI land components (8)", sorted(np.bincount(ndi.label(~w_nd, structure=E8)[0].ravel())[1:]))
print("half: water frac", round(float(tr_half.mean()), 4), "t", round(float(t_half), 3), "pred water", round(float(w_half.mean()), 3),
      "OA", round(float((w_half == tr_half).mean()), 3))
print("SAR raw otsu", score(w_raw))
print("SAR t", round(float(t_sar), 2), "class dB", {k: round(float(db[cls == k].mean()), 1) for k in range(9)})
for name, w in (("otsu", s0), ("open", s1), ("close", s2), ("mmu50", s3), ("fill", s4)):
    print(name, score(w))
print("holes filled (8-conn land comps before)", sorted(np.bincount(ndi.label(~s3, structure=E8)[0].ravel())[1:]))
print("close border loss (no pad)", int((s2 & ~s2_nopad).sum()))
print("river px", int(rv.sum()), "true", int((cls == 1).sum()), "overlap", int((rv & (cls == 1)).sum()))
print("skeleton px", int(sk.sum()), "n4", ndi.label(sk)[1], "n8", ndi.label(sk, structure=E8)[1],
      "orth", orth, "diag", diag, "len", round(orth + diag * 2 ** 0.5, 1), "true len", round(float(true_len), 1),
      "width 2*dt mean", round(float((2 * dt[sk]).mean()), 2))
print("ships n4", ndi.label(ship_b)[1], "n8", ndi.label(ship_b, structure=E8)[1], "fill added ndwi", int(w_fill.sum() - w_nd.sum()))
print("toy n4 n8", n4, n8)
