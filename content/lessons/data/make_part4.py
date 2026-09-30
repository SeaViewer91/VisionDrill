"""4부(영상과 기하) 공통 예제 자료: 가상 연안 장면 영상 한 장 (Sentinel-2 10 m 급을 흉내 낸 합성 영상).

실제 지역·영상 자료가 아님. 결과 파일 part4_scene.npz 안의 배열
- refl: (6, 300, 300) float16 지표 반사율. 밴드 순서 B2(청) B3(녹) B4(적) B8(근적외) B11 B12(단파적외)
- (DN이 필요하면 dn = round(refl / 0.5 * 4095)로 만듦: 반사율 0~0.5를 12비트 0~4095에 담는 가상 규칙)
- cls:  (300, 300) uint8 참 토지피복 (0 바다, 1 하천, 2 산림, 3 농경지, 4 시가지, 5 도로, 6 나지, 7 갯벌, 8 선박)
- sar:  (300, 300) float16 SAR 후방산란 강도(선형). 1-look 스페클(지수분포 곱셈 잡음) 포함 → 필터 예
- sar_clean: 스페클 없는 강도 (float16)
- dem:  (300, 300) float16 지형 고도(m, 수치지형모델에 해당)
- dsm:  (300, 300) float16 지형 + 건물·나무 높이(수치표면모델에 해당)
화소 크기 10 m, 영상 3 km × 3 km. 왼쪽이 바다, 오른쪽으로 갈수록 내륙·고지대.
"""
import os
import numpy as np
from scipy import ndimage as ndi

rng = np.random.default_rng(4)
H = W = 300
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

cls = np.full((H, W), 3, np.uint8)                        # 기본: 농경지
coast = 70 + 12 * np.sin(yy / 37.0) + 6 * np.sin(yy / 11.0)  # 해안선 x 위치
cls[xx < coast] = 0                                        # 바다
cls[(xx >= coast) & (xx < coast + 14) & (yy > 150)] = 7    # 갯벌(남쪽 해안)
# 하천: 동쪽 산지에서 바다로
river_y = 120 + 25 * np.sin(xx / 45.0)
cls[(np.abs(yy - river_y) < 2.5 + xx / 150.0) & (xx >= coast)] = 1
# 산림: 동쪽 고지대(얼룩진 경계)
forest_edge = 215 + 18 * np.sin(yy / 23.0) + ndi.gaussian_filter(rng.normal(0, 25, (H, W)), 6)
cls[(xx > forest_edge) & (cls != 1)] = 2
# 시가지 격자: 북쪽 해안 근처
urban = (yy > 20) & (yy < 100) & (xx > coast + 10) & (xx < 180)
cls[urban & (cls == 3)] = 4
# 도로: 시가지 안 격자 + 동서 간선도로 하나
for r in range(26, 100, 18):
    cls[(np.abs(yy - r) < 1) & urban] = 5
for c in range(95, 180, 20):
    cls[(np.abs(xx - c) < 1) & urban] = 5
cls[(np.abs(yy - 205) < 1.2) & (xx > coast + 14) & (cls != 1)] = 5
# 나지: 농경지 안 몇 곳
for cy, cx, r in [(170, 150, 9), (250, 120, 7), (230, 185, 11)]:
    cls[((yy - cy) ** 2 + (xx - cx) ** 2 < r * r) & (cls == 3)] = 6
# 선박: 바다 위 작은 밝은 점(2×3~3×5 화소)
ships = [(40, 25, 2, 3), (95, 40, 3, 5), (160, 30, 2, 4), (210, 50, 3, 4), (260, 18, 2, 3)]
for y0, x0, h, w in ships:
    cls[y0:y0 + h, x0:x0 + w] = 8

MEAN = {  # 밴드 B2 B3 B4 B8 B11 B12
    0: [0.055, 0.045, 0.025, 0.015, 0.008, 0.005], 1: [0.065, 0.070, 0.050, 0.040, 0.020, 0.012],
    2: [0.030, 0.050, 0.030, 0.330, 0.160, 0.070], 3: [0.050, 0.080, 0.065, 0.280, 0.220, 0.120],
    4: [0.095, 0.105, 0.115, 0.210, 0.250, 0.215], 5: [0.120, 0.125, 0.130, 0.170, 0.230, 0.210],
    6: [0.110, 0.130, 0.160, 0.240, 0.300, 0.260], 7: [0.070, 0.075, 0.075, 0.100, 0.080, 0.055],
    8: [0.200, 0.210, 0.220, 0.250, 0.200, 0.160],
}
refl = np.zeros((6, H, W), np.float32)
# 농경지 필지마다 다른 작물 상태(30×30 화소 블록 단위 곱셈 인자)
field = np.kron(rng.normal(1, 0.15, (10, 10)), np.ones((30, 30)))
texture = ndi.gaussian_filter(rng.normal(0, 1, (H, W)), 1.5)
for k, m in MEAN.items():
    mask = cls == k
    for b in range(6):
        v = m[b] * np.ones((H, W))
        if k == 3:
            v = v * (field if b == 3 else 2 - field)       # 근적외는 생육 좋을수록 높고 가시광은 낮게
        if k in (2, 4, 6):
            v = v * (1 + 0.12 * texture)
        refl[b][mask] = v[mask]
refl *= np.exp(rng.normal(0, 0.03, refl.shape)).astype(np.float32)   # 화소 잡음
refl = np.clip(refl, 0.001, 1.0).astype(np.float32)
dn = np.round(refl / 0.5 * 4095).clip(0, 4095).astype(np.uint16)       # 반사율 0~0.5를 12비트에

# SAR: 바다·하천 어둡고, 시가지 밝음(이중 반사), 선박 매우 밝음
SIG = {0: 0.01, 1: 0.015, 2: 0.10, 3: 0.06, 4: 0.45, 5: 0.03, 6: 0.05, 7: 0.02, 8: 2.0}
sar_clean = np.zeros((H, W), np.float32)
for k, s in SIG.items():
    sar_clean[cls == k] = s
sar_clean = ndi.gaussian_filter(sar_clean, 0.6).astype(np.float32)
sar = (sar_clean * rng.exponential(1.0, (H, W))).astype(np.float32)   # 1-look 스페클

# 지형: 해안 0 m에서 동쪽으로 올라가는 경사 + 언덕, 하천 골짜기
dem = np.clip((xx - coast) * 0.35, 0, None) + 40 * np.exp(-(((xx - 260) / 35) ** 2 + ((yy - 210) / 45) ** 2))
dem -= 6 * np.exp(-((yy - river_y) / 6) ** 2) * (xx > coast)
dem = np.clip(dem, 0, None).astype(np.float32)
height = np.zeros((H, W), np.float32)
bld = (cls == 4) & (ndi.gaussian_filter(rng.random((H, W)), 1) > 0.5)
height[bld] = rng.uniform(8, 40, bld.sum())
height[cls == 2] = 15 + 3 * texture[cls == 2]
dsm = (dem + height).astype(np.float32)

out = os.path.join(os.path.dirname(__file__), "part4_scene.npz")
np.savez_compressed(out, refl=refl.astype(np.float16), cls=cls, sar=sar.astype(np.float16), sar_clean=sar_clean.astype(np.float16), dem=dem.astype(np.float16), dsm=dsm.astype(np.float16))
print("wrote", out, {k: int((cls == k).sum()) for k in range(9)})
