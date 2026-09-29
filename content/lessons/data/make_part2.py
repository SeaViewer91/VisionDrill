"""2부(회귀) 공통 예제 자료: 가상 도시 80개 행정동의 여름철 지표면온도(LST)와 설명변수.

열: dong(동 ID), x_km(해안선에서 동쪽으로 떨어진 거리), y_km(남북 위치), coast_km(해안 거리, x_km와 같음),
    imperv_pct(불투수면 비율 %), ndvi(여름철 평균 NDVI), elev_m(평균 고도 m), pop_density(인구밀도 명/km²),
    landuse(주 용도: 녹지·주거·상업·공업), lst_c(여름철 평균 지표면온도 ℃, 위성 LST 2차 산출물),
    heat_warn(1 = LST 상위 30%, 폭염 취약 동)
공단 동 D43: 해안 반대편 외곽, 불투수면 94%, 예측보다 훨씬 뜨거움

실제 지역·기관 자료가 아님. 회귀 진단에서 다룰 현상을 일부러 넣음
- 불투수면 비율과 NDVI의 강한 음의 상관 (다중공선성, 8강)
- 불투수면이 높을수록 잔차 분산이 커짐 (이분산성, 7강)
- 해안에서 멀고 도심 외곽의 공단 동 하나 (레버리지·영향점, 8강)
- 공간적으로 이어진 잡음 (잔차의 공간 자기상관, 9강)
- 오른쪽 꼬리가 긴 인구밀도 (로그 변환, 10강)
- 폭염 경보 여부 (로지스틱 회귀, 12강)
"""
import csv, os
import numpy as np

rng = np.random.default_rng(51)
n = 80
x = rng.uniform(0, 40, n)          # 동쪽 방향 km (x=0이 해안선)
y = rng.uniform(0, 30, n)          # 북쪽 방향 km
cx, cy = 18.0, 15.0                # 도심
d_center = np.hypot(x - cx, y - cy)
coast_km = x.copy()
imperv = np.clip(95 - 3.0 * d_center + rng.normal(0, 8, n), 4, 96)          # %
ndvi = np.clip(0.78 - 0.0062 * imperv + rng.normal(0, 0.045, n), 0.05, 0.85)
elev = np.clip(8 + 4.5 * x + 3 * np.maximum(0, y - 20) ** 1.3 + rng.normal(0, 15, n), 2, None)  # m
pop = np.exp(np.log(1500) + 0.035 * imperv + rng.normal(0, 0.55, n))        # 명/km²

# 공간적으로 이어진 잡음: 가까운 동끼리 비슷한 값 (지수 공분산, 범위 약 6 km)
D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
C = 1.4 ** 2 * np.exp(-D / 8.0)
spatial = rng.multivariate_normal(np.zeros(n), C)
# 이분산: 불투수면이 높을수록 잡음 큼
hetero = rng.normal(0, 1, n) * (0.0003 * imperv ** 2)
lst = 25.5 + 0.095 * imperv - 5.0 * ndvi - 0.012 * elev + 0.06 * coast_km + spatial + hetero

# 공단 동 하나: 도심에서 멀고(해안 반대편), 불투수면 매우 높고, 예측보다 훨씬 뜨거움
i = int(np.argmax(x))
x[i], y[i] = 39.0, 3.0
coast_km[i] = x[i]
d_center[i] = np.hypot(x[i] - cx, y[i] - cy)
imperv[i], ndvi[i], elev[i], pop[i] = 94.0, 0.07, 140.0, 900.0
lst[i] = 25.5 + 0.095 * 94 - 5.0 * 0.07 - 0.012 * 140 + 0.06 * 39 + 6.5

landuse = np.where(imperv > 75, "상업", np.where(imperv > 45, "주거", "녹지"))
landuse[i] = "공업"
heat_warn = (lst >= np.quantile(lst, 0.7)).astype(int)

out = os.path.join(os.path.dirname(__file__), "part2_lst.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dong", "x_km", "y_km", "coast_km", "imperv_pct", "ndvi", "elev_m", "pop_density", "landuse", "lst_c", "heat_warn"])
    for k in range(n):
        w.writerow([f"D{k+1:02d}", f"{x[k]:.2f}", f"{y[k]:.2f}", f"{coast_km[k]:.2f}", f"{imperv[k]:.1f}", f"{ndvi[k]:.3f}",
                    f"{elev[k]:.0f}", f"{pop[k]:.0f}", landuse[k], f"{lst[k]:.2f}", heat_warn[k]])
print("wrote", out, "industrial dong:", f"D{i+1:02d}")
