"""17강 그림: 혼동행렬에서 생산자·사용자 정확도를 읽는 방향, 수계 판별 ROC 곡선

자료: content/lessons/data/part3_landcover.csv (3부 공통 예제)
- 혼동행렬: 랜덤 포레스트(나무 200그루, random_state=0), 밴드 6개 + NDVI + NDWI,
  폴리곤(patch_id) 단위 5겹 교차검증(GroupKFold) 예측. 행 = 참, 열 = 예측
- ROC: 수계 vs 비수계를 한 가지 값(점수)으로 가를 때. NDVI와 녹색 밴드(B3)는 작을수록 물 쪽이라 부호를 뒤집어 씀
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
d = pd.read_csv(os.path.join(HERE, "..", "data", "part3_landcover.csv"))
F = ["B2", "B3", "B4", "B8", "B11", "B12", "ndvi", "ndwi"]
C = ["산림", "농경지", "시가지", "수계", "나지", "갯벌"]

# ---------------------------------------------------------------- 1. 혼동행렬
rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
pred = cross_val_predict(rf, d[F].values, d.cls.values, groups=d.patch_id, cv=GroupKFold(5))
M = confusion_matrix(d.cls.values, pred, labels=C)
rs, cs, dg = M.sum(1), M.sum(0), np.diag(M)
PA, UA, OA = dg / rs, dg / cs, dg.sum() / M.sum()

s = Svg(680, 350, "토지피복 6개 클래스의 혼동행렬. 행은 참 클래스, 열은 예측 클래스이며, 행을 따라 생산자 정확도, 열을 따라 사용자 정확도를 읽음")
MX, MY, CW, CH = 120, 58, 58, 28          # 행렬 왼쪽 위, 칸 크기
SW, PW = 58, 78                           # 합계 열, 생산자 정확도 열 폭
cx = lambda j: MX + CW * j
cy = lambda i: MY + CH * i
K = len(C)
# 머리글
s.text(cx(K / 2), 20, "예측 클래스 (열)", "f-fg", 13, weight="600")
for j, c in enumerate(C):
    s.text(cx(j) + CW / 2, MY - 10, c, "f-mu", 12)
s.text(cx(K) + SW / 2, MY - 10, "행 합계", "f-mu", 12)
s.text(cx(K) + SW + PW / 2, MY - 10, "생산자 정확도", "f-ac", 12, weight="600")
for i, c in enumerate(C):
    s.text(MX - 10, cy(i) + CH / 2 + 4, c, "f-mu", 12, anchor="end")
s.text(MX - 10, cy(K) + CH / 2 + 4, "열 합계", "f-mu", 12, anchor="end")
s.text(MX - 10, cy(K + 1) + CH / 2 + 4, "사용자 정확도", "f-bd", 12, anchor="end", weight="600")
yv = cy(K / 2)
s.text(34, yv, "참 클래스 (행)", "f-fg", 13, weight="600")
s.parts[-1] = s.parts[-1].replace("<text ", f'<text transform="rotate(-90 34 {yv:.1f})" ')
# 칸
for i in range(K):
    for j in range(K):
        v = M[i, j]
        cls = "s-mu f-acs" if i == j else ("s-mu f-bds" if v >= 50 else "s-mu f-bg")
        s.rect(cx(j), cy(i), CW, CH, cls, 0.8)
        s.text(cx(j) + CW / 2, cy(i) + CH / 2 + 4, f"{v}", "f-fg" if v else "f-mu", 12,
               weight="600" if i == j or v >= 50 else None)
    s.rect(cx(K), cy(i), SW, CH, "s-mu f-sf", 0.8)
    s.text(cx(K) + SW / 2, cy(i) + CH / 2 + 4, f"{rs[i]}", "f-fg", 12)
    s.rect(cx(K) + SW, cy(i), PW, CH, "s-mu f-sf", 0.8)
    s.text(cx(K) + SW + PW / 2, cy(i) + CH / 2 + 4, f"{PA[i]:.3f}", "f-ac", 12, weight="600")
for j in range(K):
    s.rect(cx(j), cy(K), CW, CH, "s-mu f-sf", 0.8)
    s.text(cx(j) + CW / 2, cy(K) + CH / 2 + 4, f"{cs[j]}", "f-fg", 12)
    s.rect(cx(j), cy(K + 1), CW, CH, "s-mu f-sf", 0.8)
    s.text(cx(j) + CW / 2, cy(K + 1) + CH / 2 + 4, f"{UA[j]:.3f}", "f-bd", 12, weight="600")
s.rect(cx(K), cy(K), SW, CH, "s-mu f-sf", 0.8)
s.text(cx(K) + SW / 2, cy(K) + CH / 2 + 4, f"{M.sum()}", "f-fg", 12)
s.rect(cx(K) + SW, cy(K), PW, CH * 2, "s-mu f-acs", 0.8)
s.text(cx(K) + SW + PW / 2, cy(K) + CH - 3, "전체 정확도", "f-fg", 11)
s.text(cx(K) + SW + PW / 2, cy(K) + CH + 15, f"{OA:.3f}", "f-fg", 12, weight="600")
# 시가지 행(→ 생산자 정확도)과 시가지 열(→ 사용자 정확도) 강조
i = C.index("시가지")
x0, x1 = cx(0) - 2, cx(K) + SW + PW + 2
s.path(f"M {x0} {cy(i) - 2} H {x1} V {cy(i) + CH + 2} H {x0} Z", "s-ac", 2.4)
y0, y1 = cy(0) - 3, cy(K + 2) + 3
s.path(f"M {cx(i) - 3} {y0} H {cx(i) + CW + 3} V {y1} H {cx(i) - 3} Z", "s-bd", 2.4)
# 아래 설명
YN = cy(K + 2) + 30
s.text(MX, YN, "파란 틀: 시가지 행 → 참 시가지 421개 중 300개를 맞힘 → 생산자 정확도(재현율)", "f-ac", 12, anchor="start")
s.text(MX, YN + 22, "주황 틀: 시가지 열 → 시가지로 예측한 392개 중 300개가 맞음 → 사용자 정확도(정밀도)", "f-bd", 12, anchor="start")
s.save(os.path.join(OUT, "17-matrix.svg"))

# ---------------------------------------------------------------- 2. ROC 곡선 (수계 vs 비수계)
yw = (d.cls == "수계").astype(int).values
scores = [("NDVI (작을수록 물)", -d.ndvi.values, "s-ac", "f-ac"),
          ("녹색 밴드 B3 (작을수록 물)", -d.B3.values, "s-bd", "f-bd")]
s = Svg(680, 330, "수계와 비수계를 NDVI와 녹색 밴드 반사율로 가를 때의 ROC 곡선과 무작위 기준선")
X0, X1, YT, YB = 80, 330, 25, 275
sx = lambda v: X0 + (X1 - X0) * v
sy = lambda v: YB - (YB - YT) * v
s.rect(X0, YT, X1 - X0, YB - YT, "s-mu f-bg", 1.2)
for v in (0, 0.25, 0.5, 0.75, 1):
    s.line(sx(v), YB, sx(v), YB + 5, "s-mu", 1.2)
    s.text(sx(v), YB + 19, f"{v:g}", "f-mu", 12)
    s.line(X0 - 5, sy(v), X0, sy(v), "s-mu", 1.2)
    s.text(X0 - 9, sy(v) + 4, f"{v:g}", "f-mu", 12, anchor="end")
s.text((X0 + X1) / 2, YB + 40, "거짓 양성률 FPR (= 1 − 특이도)", "f-mu", 12)
s.text(X0 - 44, (YT + YB) / 2, "참 양성률 TPR (= 재현율)", "f-mu", 12)
s.parts[-1] = s.parts[-1].replace("<text ", f'<text transform="rotate(-90 {X0 - 44} {(YT + YB) / 2:.1f})" ')
s.line(sx(0), sy(0), sx(1), sy(1), "s-mu", 1.4, "6 4")
aucs = []
for name, sc, lc, tc in scores:
    fpr, tpr, th = roc_curve(yw, sc)
    s.path("M " + " L ".join(f"{sx(a):.1f} {sy(b):.1f}" for a, b in zip(fpr, tpr)), lc, 2.4)
    aucs.append(roc_auc_score(yw, sc))
# 임계값 하나 = 곡선 위 점 하나 (녹색 밴드 B3 ≤ 0.05를 물로 판정)
t = 0.05
pw = d.B3.values <= t
tp_r = (pw & (yw == 1)).sum() / yw.sum()
fp_r = (pw & (yw == 0)).sum() / (yw == 0).sum()
s.circle(sx(fp_r), sy(tp_r), 5, "f-bd")
s.rect(sx(fp_r) + 7, sy(tp_r) + 5, 176, 15, "f-bg", 0)   # 글자 뒤 대각선 가림
s.text(sx(fp_r) + 10, sy(tp_r) + 16, f"B3 ≤ {t}: FPR {fp_r:.2f}, TPR {tp_r:.2f}", "f-bd", 11, anchor="start")
# 범례
LX, LY = 372, 60
for k, (name, sc, lc, tc) in enumerate(scores):
    yy = LY + 30 * k
    s.line(LX, yy, LX + 30, yy, lc, 2.4)
    s.text(LX + 40, yy + 4, f"{name}  AUC {aucs[k]:.3f}", tc, 12, anchor="start", weight="600")
s.line(LX, LY + 60, LX + 30, LY + 60, "s-mu", 1.4, "6 4")
s.text(LX + 40, LY + 64, "무작위 점수  AUC 0.5", "f-mu", 12, anchor="start")
s.text(LX, LY + 110, "물 판정 기준을 느슨하게 할수록 오른쪽 위로", "f-fg", 12, anchor="start")
s.text(LX, LY + 132, "왼쪽 위 모서리에 붙을수록 좋은 점수", "f-fg", 12, anchor="start")
s.save(os.path.join(OUT, "17-roc.svg"))
print("ok", M.tolist(), round(OA, 4), [round(a, 4) for a in aucs], round(fp_r, 4), round(tp_r, 4))
