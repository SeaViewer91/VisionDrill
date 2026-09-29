"""18강 그림: 드문 클래스(갯벌)의 ROC 곡선과 PR 곡선, 포컬 로스 곡선
자료: content/lessons/data/part3_landcover.csv (3부 공통 예제)
분류기: 가시광 3밴드(B2·B3·B4) 표준화 + 다항 로지스틱 회귀, 폴리곤 단위 4겹 교차검증(StratifiedGroupKFold, 시드 0)의
       검증 겹 예측 확률 P(갯벌)"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
d = pd.read_csv(os.path.join(HERE, "..", "data", "part3_landcover.csv"))
X, y, g = d[["B2", "B3", "B4"]].values, d.cls.values, d.patch_id.values
p = np.zeros(len(y))
for tr, te in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0).split(X, y, g):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[tr], y[tr])
    p[te] = m.predict_proba(X[te])[:, list(m.classes_).index("갯벌")]
t = y == "갯벌"
base = t.mean()
roc_auc, ap = roc_auc_score(t, p), average_precision_score(t, p)
fpr, tpr, _ = roc_curve(t, p)
prec, rec, _ = precision_recall_curve(t, p)
i50 = int(np.argmax(tpr >= 0.5))                # 재현율 0.5를 처음 얻는 지점
op_fpr, op_tpr = fpr[i50], tpr[i50]
op_prec = op_tpr * t.sum() / (op_tpr * t.sum() + op_fpr * (~t).sum())

# ---------------------------------------------------------------- 1. ROC vs PR
s = Svg(680, 330, "드문 클래스 갯벌에 대한 같은 분류기의 ROC 곡선과 정밀도-재현율 곡선")
W, YT, YB = 240, 40, 270


def panel(x0, title, xlab, ylab):
    sx = lambda v: x0 + W * v
    sy = lambda v: YB - (YB - YT) * v
    s.line(x0, YB, x0 + W, YB, "s-mu", 1.2)
    s.line(x0, YT, x0, YB, "s-mu", 1.2)
    for v in (0, 0.5, 1):
        s.line(sx(v), YB, sx(v), YB + 4, "s-mu", 1.2)
        s.text(sx(v), YB + 18, f"{v:g}", "f-mu", 12)
        s.line(x0 - 4, sy(v), x0, sy(v), "s-mu", 1.2)
        s.text(x0 - 8, sy(v) + 4, f"{v:g}", "f-mu", 12, anchor="end")
    s.text(x0 + W / 2, YB + 40, xlab, "f-mu", 12)
    s.text(x0, YT - 12, ylab, "f-mu", 12, anchor="start")
    s.text(x0 + W / 2, 18, title, "f-fg", 13, weight="600")
    return sx, sy


# 왼쪽: ROC
sx, sy = panel(70, f"ROC 곡선 (AUC {roc_auc:.2f})", "거짓 양성률", "재현율")
s.line(sx(0), sy(0), sx(1), sy(1), "s-mu", 1.2, "5 4")
s.text(sx(0.62), sy(0.50), "무작위 0.5", "f-mu", 12, anchor="start")
s.path("M " + " L ".join(f"{sx(a):.1f} {sy(b):.1f}" for a, b in zip(fpr, tpr)), "s-ac", 2.2)
s.circle(sx(op_fpr), sy(op_tpr), 5, "f-bd")
s.text(sx(op_fpr) + 9, sy(op_tpr) + 16, f"거짓 양성률 {op_fpr:.2f}", "f-bd", 12, anchor="start", weight="600")

# 오른쪽: PR
sx, sy = panel(400, f"PR 곡선 (AP {ap:.2f})", "재현율", "정밀도")
s.line(sx(0), sy(base), sx(1), sy(base), "s-mu", 1.2, "5 4")
s.text(sx(0.12), sy(base) - 7, f"무작위 = 양성 비율 {base:.3f}", "f-mu", 12, anchor="start")
s.path("M " + " L ".join(f"{sx(a):.1f} {sy(b):.1f}" for a, b in zip(rec[:-1], prec[:-1])), "s-ac", 2.2)  # 끝의 관례점(재현율 0, 정밀도 1)은 뺌
s.circle(sx(op_tpr), sy(op_prec), 5, "f-bd")
s.text(sx(op_tpr) + 8, sy(op_prec) - 44, f"정밀도 {op_prec:.2f}", "f-bd", 12, anchor="start", weight="600")
s.line(sx(op_tpr) + 4, sy(op_prec) - 6, sx(op_tpr) + 14, sy(op_prec) - 38, "s-bd", 1.2)
s.save(os.path.join(OUT, "18-roc-pr.svg"))

# ---------------------------------------------------------------- 2. 포컬 로스
s = Svg(680, 320, "정답 클래스 확률 p_t에 따른 교차 엔트로피와 포컬 로스(감마 1, 2, 5)")
X0, X1, YT, YB = 70, 610, 25, 265
ylo, yhi = 0, 3
sx = lambda v: X0 + (X1 - X0) * v
sy = lambda v: YB - (YB - YT) * (min(v, yhi) - ylo) / (yhi - ylo)
# 쉬운 표본 구간
s.rect(sx(0.6), YT, sx(1) - sx(0.6), YB - YT, "f-sf", 0)
s.text((sx(0.6) + sx(1)) / 2 + 20, YT + 20, "이미 잘 맞히는 쉬운 표본", "f-mu", 12)
s.line(X0, YB, X1, YB, "s-mu", 1.2)
s.line(X0, YT, X0, YB, "s-mu", 1.2)
for v in (0, 0.2, 0.4, 0.6, 0.8, 1):
    s.line(sx(v), YB, sx(v), YB + 4, "s-mu", 1.2)
    s.text(sx(v), YB + 18, f"{v:g}", "f-mu", 12)
for v in (0, 1, 2, 3):
    s.line(X0 - 4, sy(v), X0, sy(v), "s-mu", 1.2)
    s.text(X0 - 8, sy(v) + 4, f"{v}", "f-mu", 12, anchor="end")
s.text((X0 + X1) / 2, YB + 40, "정답 클래스에 준 확률 p_t", "f-mu", 12)
s.text(X0 - 8, YT - 8, "손실", "f-mu", 12, anchor="end")
xs = np.linspace(0.05, 1, 300)
styles = {0: ("s-mu", None), 1: ("s-ac", "6 4"), 2: ("s-ac", None), 5: ("s-bd", None)}
for gm, (cls, dash) in styles.items():
    ls = -(1 - xs) ** gm * np.log(xs)
    keep = ls <= yhi
    pts = " L ".join(f"{sx(a):.1f} {sy(b):.1f}" for a, b in zip(xs[keep], ls[keep]))
    s.path("M " + pts, cls, 2.3 if gm in (0, 2) else 1.8)
    if dash:
        s.parts[-1] = s.parts[-1].replace("/>", f' stroke-dasharray="{dash}"/>')
# 범례 (곡선이 지나지 않는 위쪽 빈 곳)
lab_cls = {0: "f-mu", 1: "f-ac", 2: "f-ac", 5: "f-bd"}
for k, gm in enumerate((0, 1, 2, 5)):
    yy = 62 + 24 * k
    cls, dash = styles[gm]
    s.line(240, yy, 272, yy, cls, 2.3 if gm in (0, 2) else 1.8, dash)
    name = "교차 엔트로피 (γ = 0)" if gm == 0 else f"포컬 로스 γ = {gm}"
    s.text(280, yy + 4, name, lab_cls[gm], 12, anchor="start", weight="600")
s.save(os.path.join(OUT, "18-focal.svg"))
print(f"ROC-AUC {roc_auc:.3f} AP {ap:.3f} base {base:.4f} op fpr {op_fpr:.3f} tpr {op_tpr:.3f} prec {op_prec:.3f}")
