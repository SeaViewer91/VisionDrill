"""16강 그림: 작은 결정트리(NDVI·NDWI 두 특징, 잎 4개), 나무 깊이에 따른 정확도,
불순도 중요도와 순열 중요도 비교

자료: content/lessons/data/part3_landcover.csv (3부 공통 예제). scikit-learn, random_state=0
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "fig")
d = pd.read_csv(os.path.join(HERE, "..", "data", "part3_landcover.csv"))
F = ["B2", "B3", "B4", "B8", "B11", "B12", "ndvi", "ndwi"]
X, y, g = d[F].values, d.cls.values, d.patch_id.values
kf, gk = KFold(5, shuffle=True, random_state=0), GroupKFold(5)


def gini(c):
    p = np.asarray(c, float) / np.sum(c)
    return 1 - (p ** 2).sum()


def vtext(s, x, y, txt):
    s.text(x, y, txt, "f-mu", 12)
    s.parts[-1] = s.parts[-1].replace("<text ", f'<text transform="rotate(-90 {x:.1f} {y:.1f})" ')


# ---------------------------------------------------------------- 1. 작은 결정트리
t = DecisionTreeClassifier(max_leaf_nodes=4, random_state=0).fit(d[["ndvi", "ndwi"]], y)
tr = t.tree_
cls = list(t.classes_)
names = ["NDVI", "NDWI"]
info = {}
for i in range(tr.node_count):
    v = tr.value[i][0] * tr.n_node_samples[i]          # sklearn 1.4+ value는 비율
    n = int(round(tr.n_node_samples[i]))
    info[i] = dict(n=n, gini=gini(v), top=cls[int(np.argmax(v))], hit=int(round(v.max())),
                   leaf=tr.children_left[i] < 0,
                   rule=None if tr.children_left[i] < 0 else f"{names[tr.feature[i]]} ≤ {tr.threshold[i]:.3g} ?")
# 배치: 노드 번호 → (x, 층)
root = 0
L1, R1 = tr.children_left[root], tr.children_right[root]
L2, R2 = tr.children_left[L1], tr.children_right[L1]
L3, R3 = tr.children_left[L2], tr.children_right[L2]
pos = {root: (400, 0), L1: (250, 1), R1: (565, 1), L2: (150, 2), R2: (380, 2), L3: (85, 3), R3: (255, 3)}
for k in pos:
    if k not in (root, L1, L2):
        assert info[k]["leaf"], k
s = Svg(680, 360, "NDVI와 NDWI 두 특징으로 토지피복 6개 클래스를 가르는 잎 4개짜리 결정트리")
BW, BH, Y0, DY = 150, 50, 14, 92
cy = lambda lv: Y0 + lv * DY
for k, (x, lv) in pos.items():                      # 가지(선)와 예/아니오 표시 먼저
    if info[k]["leaf"]:
        continue
    for child, lab in ((tr.children_left[k], "예"), (tr.children_right[k], "아니오")):
        cx, clv = pos[child]
        x1, y1, x2, y2 = x, cy(lv) + BH, cx, cy(clv)
        s.line(x1, y1, x2, y2, "s-mu", 1.4)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        s.text(mx + (-8 if x2 < x1 else 8), my + 4, lab, "f-mu", 12, anchor="end" if x2 < x1 else "start")
for k, (x, lv) in pos.items():
    o = info[k]
    if o["leaf"]:
        s.rect(x - BW / 2, cy(lv), BW, BH, "s-ac f-acs", 1.5, rx=8)
        s.text(x, cy(lv) + 21, o["top"], "f-ac", 14, weight="700")
        s.text(x, cy(lv) + 40, f"{o['n']:,}개 중 {o['hit']:,}개 · 지니 {o['gini']:.2f}", "f-mu", 11)
    else:
        s.rect(x - BW / 2, cy(lv), BW, BH, "s-fg f-sf", 1.5, rx=4)
        s.text(x, cy(lv) + 21, o["rule"], "f-fg", 14, weight="700")
        s.text(x, cy(lv) + 40, f"{o['n']:,}개 · 지니 {o['gini']:.2f}", "f-mu", 11)
s.save(os.path.join(OUT, "16-tree.svg"))
print("tree", {k: (info[k]["rule"] or info[k]["top"], info[k]["n"], info[k]["hit"], round(info[k]["gini"], 3)) for k in pos},
      "acc", round(t.score(d[["ndvi", "ndwi"]], y), 4))

# ---------------------------------------------------------------- 2. 깊이에 따른 정확도
full = DecisionTreeClassifier(random_state=0).fit(X, y)
DMAX = full.get_depth()
depths = list(range(1, DMAX + 1))
rows = []
for dep in depths:
    m = DecisionTreeClassifier(max_depth=dep, random_state=0)
    trn = m.fit(X, y).score(X, y)
    rows.append((trn, cross_val_score(m, X, y, cv=kf).mean(), cross_val_score(m, X, y, cv=gk, groups=g).mean()))
rows = np.array(rows)
print("depth", DMAX, np.round(rows, 3).tolist())

s = Svg(680, 320, "결정트리 깊이에 따른 학습 정확도, 무작위 5겹 교차검증 정확도, 폴리곤 단위 교차검증 정확도")
X0, X1, YT, YB = 70, 520, 20, 260
lo, hi = 0.55, 1.0
sx = lambda v: X0 + (X1 - X0) * (v - 1) / (DMAX - 1)
sy = lambda a: YB - (YB - YT) * (a - lo) / (hi - lo)
s.line(X0, YB, X1, YB, "s-mu", 1.2)
s.line(X0, YT, X0, YB, "s-mu", 1.2)
for a in (0.6, 0.7, 0.8, 0.9, 1.0):
    s.line(X0, sy(a), X1, sy(a), "s-mu", 0.6, "2 4")
    s.text(X0 - 8, sy(a) + 4, f"{a:.1f}", "f-mu", 12, anchor="end")
for v in depths:
    if v in (1, 4, 8, 12, DMAX):
        s.line(sx(v), YB, sx(v), YB + 4, "s-mu", 1.2)
        s.text(sx(v), YB + 18, f"{v}", "f-mu", 12)
s.text((X0 + X1) / 2, YB + 40, "나무 깊이(최대)", "f-mu", 12)
vtext(s, 22, (YT + YB) / 2, "정확도")
STY = [("s-bd", None, "f-bd", "학습 자료"), ("s-mu", "6 4", "f-mu", "무작위 5겹"), ("s-ac", None, "f-ac", "폴리곤 단위 5겹")]
for j, (sc, dash, fc, lab) in enumerate(STY):
    pts = " L ".join(f"{sx(v):.1f} {sy(a):.1f}" for v, a in zip(depths, rows[:, j]))
    s.path("M " + pts, sc, 2.2)
    if dash:
        s.parts[-1] = s.parts[-1].replace("/>", f' stroke-dasharray="{dash}"/>')
    for v, a in zip(depths, rows[:, j]):
        s.circle(sx(v), sy(a), 2.6, fc)
    # 오른쪽 끝 값 표시
    s.text(X1 + 10, sy(rows[-1, j]) + 4 + (-6 if j == 0 else 6 if j == 1 else 0), f"{lab} {rows[-1, j]:.2f}", fc, 12,
           anchor="start", weight="600")
s.save(os.path.join(OUT, "16-depth.svg"))

# ---------------------------------------------------------------- 3. 불순도 중요도 vs 순열 중요도
rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1).fit(X, y)
mdi = rf.feature_importances_
pis = []
for trn, tst in gk.split(X, y, g):
    m = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1).fit(X[trn], y[trn])
    pis.append(permutation_importance(m, X[tst], y[tst], n_repeats=10, random_state=0, n_jobs=-1).importances_mean)
perm = np.mean(pis, 0)
print("mdi", dict(zip(F, np.round(mdi, 3))), "perm", dict(zip(F, np.round(perm, 3))))
order = np.argsort(-mdi)
LBL = {"ndvi": "NDVI", "ndwi": "NDWI"}
s = Svg(680, 292, "랜덤 포레스트의 불순도 기반 중요도와 폴리곤 단위 검증 자료에서 잰 순열 중요도를 특징별로 비교한 막대그래프")
TOP, RH = 46, 30
panels = [(110, 300, mdi, 0.20, "불순도 중요도 (합 = 1)", "s-mu f-acs"), (430, 620, perm, 0.05, "순열 중요도 (정확도 하락)", "s-mu f-bds")]
for (PX0, PX1, val, vmax, title, fcls) in panels:
    s.text((PX0 + PX1) / 2, 20, title, "f-fg", 13, weight="700")
    s.line(PX0, TOP - 6, PX0, TOP + RH * len(F) - 4, "s-mu", 1.2)
    for r, i in enumerate(order):
        yy = TOP + r * RH
        w = (PX1 - PX0) * max(val[i], 0) / vmax
        hl = F[i] == "ndwi"
        s.rect(PX0, yy, w, RH - 10, ("s-bd " + fcls.split()[1]) if hl else fcls, 1.6 if hl else 1)
        s.text(PX0 + w + 6, yy + 15, f"{val[i]:.3f}", "f-bd" if hl else "f-mu", 12, anchor="start", weight="600" if hl else None)
        s.text(PX0 - 8, yy + 15, LBL.get(F[i], F[i]), "f-bd" if hl else "f-fg", 12, anchor="end", weight="600" if hl else None)
s.save(os.path.join(OUT, "16-importance.svg"))
print("ok")


# ---------------------------------------------------------------- 본문 수치 재현 (python3 16.py --numbers, 수 분 걸림)
def numbers():
    from sklearn.ensemble import HistGradientBoostingClassifier
    rf500 = lambda **k: RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1, **k)
    ent = lambda c: -sum(q * np.log2(q) for q in np.asarray(c, float) / np.sum(c) if q > 0)
    print("손계산 지니", gini([4, 4, 2]), gini([4, 2]), 0.6 * gini([4, 2]), 0.6 * gini([2, 4]) + 0.4 * gini([2, 2]))
    print("손계산 엔트로피 감소", ent([4, 4, 2]) - 0.6 * ent([4, 2]), ent([4, 4, 2]) - 0.6 * ent([2, 4]) - 0.4 * ent([2, 2]))
    for dep in (3, 5, None):
        m = DecisionTreeClassifier(max_depth=dep, random_state=0).fit(X, y)
        print("깊이", dep, m.get_depth(), m.get_n_leaves(), round(m.score(X, y), 3),
              round(cross_val_score(m, X, y, cv=kf).mean(), 3), round(cross_val_score(m, X, y, cv=gk, groups=g).mean(), 3))
    m = DecisionTreeClassifier(ccp_alpha=0.003, random_state=0).fit(X, y)
    print("ccp 0.003", m.get_n_leaves(), round(cross_val_score(m, X, y, cv=gk, groups=g).mean(), 3))
    print("OOB 비율", (1 - 1 / len(y)) ** len(y), 500 * (1 - 1 / len(y)) ** len(y))
    for name, kw in (("배깅", dict(max_features=None)), ("RF", {})):
        print(name, round(cross_val_score(rf500(**kw), X, y, cv=kf).mean(), 3),
              round(cross_val_score(rf500(**kw), X, y, cv=gk, groups=g).mean(), 3), round(rf500(oob_score=True, **kw).fit(X, y).oob_score_, 3))
    at = []
    for trn, tst in gk.split(X, y, g):
        m = rf500().fit(X[trn], y[trn])
        at.append(np.mean([(m.classes_[e.predict(X[tst]).astype(int)] == y[tst]).mean() for e in m.estimators_]))
    print("나무 한 그루 평균", round(np.mean(at), 3))
    classes = np.unique(y); yi = np.searchsorted(classes, y); rng = np.random.default_rng(0); P = np.unique(g)
    votes = np.zeros((len(y), len(classes)))
    for b in range(500):                                   # 폴리곤 단위 부트스트랩 OOB
        w = d.patch_id.map(pd.Series(rng.choice(P, len(P), replace=True)).value_counts()).fillna(0).values
        tr_ = w > 0
        t_ = DecisionTreeClassifier(max_features="sqrt", random_state=b).fit(X[tr_], yi[tr_], sample_weight=w[tr_])
        votes[np.where(~tr_)[0], t_.predict(X[~tr_])] += 1
    print("폴리곤 OOB", round((votes.argmax(1) == yi).mean(), 3))
    hgb = HistGradientBoostingClassifier(random_state=0)
    print("부스팅", round(cross_val_score(hgb, X, y, cv=kf).mean(), 3), round(cross_val_score(hgb, X, y, cv=gk, groups=g).mean(), 3))
    for FF in (["B2", "B3", "B4", "B8", "B11", "B12"], ["B2", "B3", "B11", "B12", "ndvi", "ndwi"]):
        print("특징", FF, dict(zip(FF, np.round(rf500().fit(d[FF], y).feature_importances_, 3))),
              round(cross_val_score(rf500(), d[FF].values, y, cv=gk, groups=g).mean(), 3))
    idx = [F.index(f) for f in ("B4", "B8", "ndvi", "ndwi")]; gp = []
    for trn, tst in gk.split(X, y, g):                     # 묶음 순열
        m = rf500().fit(X[trn], y[trn]); base = m.score(X[tst], y[tst]); r2 = np.random.default_rng(1); dr = []
        for _ in range(10):
            Xp = X[tst].copy(); Xp[:, idx] = X[tst][r2.permutation(len(tst))][:, idx]; dr.append(base - m.score(Xp, y[tst]))
        gp.append(np.mean(dr))
    print("묶음 순열", round(np.mean(gp), 3))
    rng = np.random.default_rng(0); d["noise"] = rng.uniform(0, 1, len(d)); d["coin"] = rng.integers(0, 2, len(d))
    for FF in (["B2", "B3", "noise", "coin"], F + ["x_km"]):
        X2 = d[FF].values; pi = []
        for trn, tst in gk.split(X2, y, g):
            m = rf500().fit(X2[trn], y[trn])
            pi.append(permutation_importance(m, X2[tst], y[tst], n_repeats=10, random_state=0, n_jobs=-1).importances_mean)
        print(FF, "MDI", dict(zip(FF, np.round(rf500().fit(X2, y).feature_importances_, 3))), "순열", dict(zip(FF, np.round(np.mean(pi, 0), 3))),
              "폴리곤 CV", round(cross_val_score(rf500(), X2, y, cv=gk, groups=g).mean(), 3))


if "--numbers" in sys.argv:
    numbers()
