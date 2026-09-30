"""20강 그림: 래스터 = 행·열 격자가 밴드별로 쌓인 3차원 배열(도식, SVG),
공통 장면의 트루컬러·폴스컬러 합성(PNG), 같은 해역을 10·30·60 m로 줄였을 때 선박이 흐려지는 모습(PNG)

자료: content/lessons/data/part4_scene.npz (4부 공통 예제, 10 m 가상 연안 장면)
PNG 안에는 글자를 넣지 않음(패널 설명은 본문 캡션에)
`python3 20.py --numbers` 로 본문 수치를 다시 계산해 출력함
"""
import os, sys, math
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
D = np.load(os.path.join(HERE, "..", "data", "part4_scene.npz"))
R = D["refl"].astype(np.float32)          # (6, 300, 300): B2 B3 B4 B8 B11 B12
CLS = D["cls"]


def arrow(s, x1, y1, x2, y2, cls="s-fg", fcls="f-fg", width=1.5, head=7):
    """직선 화살표 (marker id를 쓰지 않으려고 화살촉을 삼각형 경로로 그림)"""
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
    s.line(x1, y1, bx, by, cls, width)
    px, py = -math.sin(ang) * head * 0.5, math.cos(ang) * head * 0.5
    s.path(f"M {x2:.1f} {y2:.1f} L {bx + px:.1f} {by + py:.1f} L {bx - px:.1f} {by - py:.1f} Z", fcls, 0)


# ---------------------------------------------------------------- 1. 래스터 도식
s = Svg(600, 292, "래스터는 행과 열의 격자가 밴드마다 한 장씩 쌓인 3차원 배열이고, 화소 하나는 밴드 수만큼의 값을 가짐")
C, N, OFF = 30, 5, 22                      # 칸 크기, 격자 5×5, 뒤 밴드로 갈 때 어긋나는 거리
X0, Y0 = 80, 110                           # 맨 앞 밴드 왼쪽 위
HR, HC = 1, 4                              # 강조할 화소 (행 1, 열 4)
names = ["B2 청", "B3 녹", "B4 적", "B8 근적외"]
for k in (3, 2, 1):                        # 뒤 밴드부터 그림
    x, y = X0 + OFF * k, Y0 - OFF * k
    s.rect(x, y, C * N, C * N, "s-mu f-sf", 1.2)
    s.rect(x + HC * C, y + HR * C, C, C, "s-bd f-bds", 1.4)   # 같은 위치의 화소 (앞 밴드에 가려 일부만 보임)
    s.text(x + C * N / 2, y + 15, names[k], "f-mu", 11.5)
# 맨 앞 밴드: 격자
s.rect(X0, Y0, C * N, C * N, "s-fg f-acs", 1.4)
for i in range(1, N):
    s.line(X0 + i * C, Y0, X0 + i * C, Y0 + C * N, "s-mu", 1)
    s.line(X0, Y0 + i * C, X0 + C * N, Y0 + i * C, "s-mu", 1)
s.rect(X0 + HC * C, Y0 + HR * C, C, C, "s-bd f-bds", 2)
s.text(X0 + C / 2, Y0 + C / 2 + 4, "0,0", "f-mu", 10.5)
s.text(X0 + HC * C + C / 2, Y0 + HR * C + C / 2 + 4, "1,4", "f-bd", 10.5, weight="700")
s.text(X0 - 8, Y0 + 12, names[0], "f-mu", 11.5, anchor="end")
# 행·열 축 (원점은 왼쪽 위)
arrow(s, X0 - 16, Y0 + 26, X0 - 16, Y0 + C * N, "s-mu", "f-mu", 1.3)
s.text(X0 - 24, Y0 + C * N / 2 + 18, "행", "f-mu", 12, anchor="end")
arrow(s, X0, Y0 + C * N + 16, X0 + C * N, Y0 + C * N + 16, "s-mu", "f-mu", 1.3)
s.text(X0 + C * N + 8, Y0 + C * N + 20, "열", "f-mu", 12, anchor="start")
# 오른쪽: 화소 하나의 밴드별 값
VX, VY, VW, VH = 400, 72, 150, 28
s.text(VX + VW / 2, VY - 12, "화소 (1, 4)의 값", "f-fg", 12.5, weight="600")
vals = ["0.030", "0.050", "0.030", "0.331"]
for i in range(4):
    y = VY + i * VH
    s.rect(VX, y, VW, VH, "s-fg f-bg", 1.2)
    s.text(VX + 10, y + 18, names[i], "f-mu", 11.5, anchor="start")
    s.text(VX + VW - 10, y + 18, vals[i], "f-fg", 12, anchor="end")
xs = X0 + OFF * 3 + C * N + 12              # 쌓인 밴드의 오른쪽 끝 바깥
arrow(s, xs, VY + 2 * VH, VX - 6, VY + 2 * VH, "s-bd", "f-bd", 1.5)
s.text(VX + VW / 2, VY + 4 * VH + 30, "배열 모양 (밴드, 행, 열)", "f-fg", 12)
s.text(VX + VW / 2, VY + 4 * VH + 50, "이 그림 (4, 5, 5)", "f-mu", 11.5)
s.text(VX + VW / 2, VY + 4 * VH + 68, "공통 장면 (6, 300, 300)", "f-mu", 11.5)
s.save(os.path.join(OUT, "20-raster.svg"))


# ---------------------------------------------------------------- 영상 공통: 늘이기(스트레칭) 후 8비트로
def stretch(bands, lo, hi):
    """밴드별 선형 늘이기. lo·hi는 밴드별 반사율 경계 (21강에서 다룸)"""
    x = (np.stack(bands, -1) - lo) / (hi - lo)
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def pct(bands, a=1, b=99):
    lo = np.array([np.percentile(x, a) for x in bands])
    hi = np.array([np.percentile(x, b) for x in bands])
    return lo, hi


def up(img, k):
    """최근접 확대 (화소를 k×k 블록으로)"""
    return np.repeat(np.repeat(img, k, 0), k, 1)


def save_png(panels, path, gap=16):
    h = panels[0].shape[0]
    w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = np.full((h, w, 3), 255, np.uint8)
    x = 0
    for p in panels:
        canvas[:, x:x + p.shape[1]] = p
        x += p.shape[1] + gap
    Image.fromarray(canvas).quantize(256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    return canvas.shape


# ---------------------------------------------------------------- 2. 트루컬러·폴스컬러
tc = [R[2], R[1], R[0]]                   # 적·녹·청 자리에 B4·B3·B2
fc = [R[3], R[2], R[1]]                   # 적·녹·청 자리에 B8·B4·B3
def fit(img, w):
    """최근접으로 가로 w 화소에 맞춤 (PNG 가로 1,200 화소 이하)"""
    return np.asarray(Image.fromarray(img).resize((w, w), Image.NEAREST))


p1 = fit(stretch(tc, *pct(tc)), 580)
p2 = fit(stretch(fc, *pct(fc)), 580)
shape_scene = save_png([p1, p2], os.path.join(OUT, "20-scene.png"), gap=20)


# ---------------------------------------------------------------- 3. GSD 비교 (10·30·60 m)
def agg(a, k):
    """k×k 블록 평균 = 화소 크기를 k배로 키운 영상을 흉내 냄"""
    H, W = a.shape
    return a[:H // k * k, :W // k * k].reshape(H // k, k, W // k, k).mean((1, 3))


r0, r1, c0, c1 = 84, 228, 0, 144          # 선박 셋이 들어 있는 해역 144×144 화소 (6으로 나누어떨어지게)
lo, hi = pct(tc)
panels = []
for k in (1, 3, 6):
    bands = [agg(b[r0:r1, c0:c1], k) for b in tc]
    panels.append(up(up(stretch(bands, lo, hi), k), 2))   # 같은 크기(288×288)로 보이게 확대
shape_gsd = save_png(panels, os.path.join(OUT, "20-gsd.png"))
print("ok", shape_scene, shape_gsd,
      {f: os.path.getsize(os.path.join(OUT, f)) // 1024 for f in ("20-scene.png", "20-gsd.png")})


# ---------------------------------------------------------------- 본문 수치 재현 (확인용 출력)
if "--numbers" in sys.argv:
    from scipy import ndimage as ndi
    # 드론 GSD = 고도 × 센서 화소 크기 / 초점거리
    pitch, f = 13.2e-3 / 5472, 8.8e-3
    for H in (50, 100):
        g = H * pitch / f
        print(f"고도 {H} m: GSD {g * 100:.2f} cm, 촬영 폭 {5472 * g:.0f} m × {3648 * g:.0f} m")
    print(f"화소 크기 {pitch * 1e6:.2f} um, GSD 3 cm 고도 {0.03 * f / pitch:.1f} m")
    print("단계 수", {b: 2 ** b for b in (8, 12, 14, 16)})
    n = 10980 ** 2
    print(f"S2 타일 10 m 밴드 {n:,} 화소, 16비트 {n * 2 / 1e6:.1f} MB, 8비트 {n / 1e6:.1f} MB")
    print(f"공통 장면 float32 {R.size * 4 / 1e6:.2f} MB, uint16 {R.size * 2 / 1e6:.2f} MB")
    # 방사해상도: 반사율 0~0.5를 12비트/8비트에 담았을 때 바다 화소의 서로 다른 값 수
    sea = CLS == 0
    for b, nm in ((3, "B8"), (2, "B4")):
        d12 = np.round(R[b][sea] / 0.5 * 4095)
        d8 = np.round(R[b][sea] / 0.5 * 255)
        print(nm, f"바다 평균 {R[b][sea].mean():.4f}", "12비트 값 수", len(np.unique(d12)), "8비트 값 수", len(np.unique(d8)), np.unique(d8))
    print(f"한 단계 = 반사율 12비트 {0.5 / 4095:.6f}, 8비트 {0.5 / 255:.5f}")
    # DN → 반사율
    print("DN 1650 -> S2 L2A (DN-1000)/10000:", (1650 - 1000) / 1e4, " Landsat C2 L2 DN*2.75e-5-0.2:", round(1650 * 2.75e-5 - 0.2, 4))
    # 현장에서는: 농경지 적색 0.065·근적외 0.279, 오프셋 1,000을 빼지 않고 10,000으로만 나눈 경우의 NDVI
    r4, r8 = 0.065, 0.279
    d4, d8 = r4 * 1e4 + 1000, r8 * 1e4 + 1000
    print("농경지 장면 평균 B4·B8", R[2][CLS == 3].mean().round(3), R[3][CLS == 3].mean().round(3),
          "DN", d4, d8, "NDVI 바른", round((r8 - r4) / (r8 + r4), 3), "오프셋 안 뺌", round((d8 - d4) / (d8 + d4), 3))
    # 구름: 촬영마다 흐릴 확률 0.6, 서로 독립이라고 가정
    print("30일 안에 맑은 장면 1장 이상:", {"5일(6회)": round(1 - 0.6 ** 6, 3), "16일(2회)": round(1 - 0.6 ** 2, 3)})
    # 선박: 10·30·60 m에서 걸친 화소 수, 화소 안 선박 비율 최댓값, 적색(B4) 최댓값
    lab, nship = ndi.label(CLS == 8)
    print("바다 B4", round(float(R[2][sea].mean()), 3), "선박 B4", round(float(R[2][CLS == 8].mean()), 3))
    for k in (1, 3, 6):
        a = agg(R[2], k)
        row = []
        for i, sl in enumerate(ndi.find_objects(lab)):
            m = agg((lab == i + 1).astype(float), k)
            row.append((f"{sl[0].stop - sl[0].start}x{sl[1].stop - sl[1].start}", int((m > 0).sum()), round(float(m.max()), 3), round(float(a[m > 0].max()), 3)))
        print(f"{k * 10} m", row)
