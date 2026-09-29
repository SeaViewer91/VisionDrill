"""3부(머신러닝 기본기) 공통 예제 자료: 가상 연안 지역 Sentinel-2 영상의 토지피복 표본 픽셀.

실제 지역·영상 자료가 아님. 3부에서 다룰 현상을 일부러 넣음
- 표본 픽셀이 폴리곤(patch) 단위로 모여 있음: 같은 폴리곤 픽셀끼리 비슷함 → 무작위 교차검증이 점수를 부풀림 (15강)
- 시가지와 나지, 갯벌과 수계, 농경지와 산림이 일부 헷갈림 (17강 혼동행렬)
- 갯벌이 드묾 (18강 클래스 불균형)
- 라벨 없이 군집·주성분 분석을 해 볼 수 있는 분광 특징 (19강)

열: pixel_id, patch_id(현장 조사 폴리곤), x_km, y_km(영상 안 위치), block(10 km 격자 블록 번호),
    B2, B3, B4, B8, B11, B12(청·녹·적·근적외·단파적외1·2 지표 반사율), ndvi, ndwi, cls(토지피복 클래스)
"""
import csv, os
import numpy as np

rng = np.random.default_rng(3)
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
# 클래스별 평균 반사율(가상, 대략적인 전형값)과 폴리곤 수
CLASSES = {
    "산림":   ([0.030, 0.050, 0.030, 0.330, 0.160, 0.070], 34),
    "농경지": ([0.050, 0.080, 0.065, 0.280, 0.220, 0.120], 30),
    "시가지": ([0.095, 0.105, 0.115, 0.210, 0.250, 0.215], 16),
    "수계":   ([0.060, 0.050, 0.030, 0.020, 0.012, 0.008], 12),
    "나지":   ([0.110, 0.130, 0.160, 0.240, 0.300, 0.260], 8),
    "갯벌":   ([0.070, 0.075, 0.075, 0.100, 0.080, 0.055], 4),
}
# 클래스가 주로 나타나는 곳(영상 60×40 km, x=0 쪽이 바다)
REGION = {
    "산림": (30, 60, 0, 40), "농경지": (12, 45, 0, 40), "시가지": (15, 35, 10, 30),
    "수계": (0, 60, 0, 40), "나지": (12, 50, 0, 40), "갯벌": (0, 8, 0, 40),
}
rows, pid = [], 0
for cls, (mean, n_patch) in CLASSES.items():
    mean = np.array(mean)
    x0, x1, y0, y1 = REGION[cls]
    for _ in range(n_patch):
        pid += 1
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        patch_mean = mean * np.exp(rng.normal(0, 0.16, 6))           # 폴리곤마다 다른 상태(작물 생육, 건물 재질 등)
        if cls == "농경지" and rng.random() < 0.25:                  # 일부는 휴경·수확 직후라 나지에 가까움
            patch_mean = 0.5 * patch_mean + 0.5 * np.array(CLASSES["나지"][0])
        n_pix = int(rng.integers(15, 45))
        for _ in range(n_pix):
            refl = patch_mean * np.exp(rng.normal(0, 0.07, 6))
            refl = np.clip(refl + rng.normal(0, 0.004, 6), 0.001, 0.8)
            x = float(np.clip(cx + rng.normal(0, 0.25), 0, 60))
            y = float(np.clip(cy + rng.normal(0, 0.25), 0, 40))
            rows.append((pid, x, y, cls, refl))

order = rng.permutation(len(rows))
out = os.path.join(os.path.dirname(__file__), "part3_landcover.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["pixel_id", "patch_id", "x_km", "y_km", "block", *BANDS, "ndvi", "ndwi", "cls"])
    for k, i in enumerate(order):
        p, x, y, cls, r = rows[i]
        ndvi = (r[3] - r[2]) / (r[3] + r[2])
        ndwi = (r[1] - r[3]) / (r[1] + r[3])
        block = int(x // 10) + 6 * int(y // 10)
        w.writerow([k + 1, p, f"{x:.3f}", f"{y:.3f}", block, *[f"{v:.4f}" for v in r], f"{ndvi:.3f}", f"{ndwi:.3f}", cls])
print("wrote", out, len(rows), "pixels,", pid, "patches")
