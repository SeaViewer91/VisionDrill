"""13강 그림: 레이블 유무로 나눈 학습 방식 네 가지, 학습과 추론 단계의 흐름
(개념 도식이라 자료를 읽지 않음. 2,400만 픽셀 = 60×40 km 장면을 10 m 격자로 나눈 값)"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from svglib import Svg

OUT = os.path.join(os.path.dirname(__file__), "..", "fig")


def arrow(s, x1, y1, x2, y2, cls="s-fg", fcls="f-fg", width=1.6, dash=None, head=7):
    """직선 화살표. 화살촉은 삼각형 경로로 직접 그림(marker의 id를 쓰지 않기 위해)"""
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
    s.line(x1, y1, bx, by, cls, width, dash)
    px, py = -math.sin(ang) * head * 0.5, math.cos(ang) * head * 0.5
    s.path(f"M {x2:.1f} {y2:.1f} L {bx + px:.1f} {by + py:.1f} L {bx - px:.1f} {by - py:.1f} Z", fcls, 0)


def orect(s, x, y, w, h, cls="s-fg", width=1.2, rx=0, dash=None):
    """채움 없는 사각형 (군집·영상 테두리)"""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return s.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" class="{cls}" fill="none" stroke-width="{width}"{d}/>')


# ---------------------------------------------------------------- 1. 학습 방식 네 가지
s = Svg(680, 275, "레이블이 얼마나 있느냐에 따라 나눈 지도학습, 준지도학습, 비지도학습, 자기지도학습")
CW, GAP, X00 = 160, 10, 5
CLS = [["f-ok", "f-ok", "f-ac", "f-ac"],
       ["f-ok", "f-ok", "f-ac", "f-ac"],
       ["f-bd", "f-bd", "f-bd", "f-bd"]]
titles = [("지도학습", "레이블 전부"), ("준지도학습", "레이블 일부"),
          ("비지도학습", "레이블 없음"), ("자기지도학습", "자료가 정답을 만듦")]
goals = [("입력 → 정답 맞히기", "예: 토지피복 분류"), ("소수 정답 + 무라벨", "예: 일부 폴리곤만 조사"),
         ("비슷한 것끼리 묶기", "예: K-평균 군집"), ("가린 부분 복원 등", "예: 기초모델 사전학습")]
SQ, SG = 22, 10                                   # 표본 칸 크기, 간격
PY0, PH = 60, 118                                 # 패널 위치·높이
for k in range(4):
    x0 = X00 + k * (CW + GAP)
    cx = x0 + CW / 2
    s.text(cx, 24, titles[k][0], "f-fg", 14, weight="700")
    s.text(cx, 44, titles[k][1], "f-mu", 12)
    s.rect(x0, PY0, CW, PH, "s-mu f-sf", 1, rx=6)
    gw, gh = 4 * SQ + 3 * SG, 3 * SQ + 2 * SG
    gx, gy = cx - gw / 2, PY0 + (PH - gh) / 2
    if k < 3:
        for r in range(3):
            for c in range(4):
                x, y = gx + c * (SQ + SG), gy + r * (SQ + SG)
                labeled = (k == 0) or (k == 1 and (r, c) in {(0, 0), (1, 3), (2, 1)})
                s.rect(x, y, SQ, SQ, ("s-mu " + CLS[r][c]) if labeled else "s-mu f-bg", 1, rx=3)
        if k == 2:                                # 비지도: 비슷한 것끼리 묶인 군집을 점선으로
            pad = 3.5
            for (c0, r0, c1, r1) in ((0, 0, 1, 1), (2, 0, 3, 1), (0, 2, 3, 2)):
                x = gx + c0 * (SQ + SG) - pad
                y = gy + r0 * (SQ + SG) - pad
                w = (c1 - c0) * (SQ + SG) + SQ + 2 * pad
                h = (r1 - r0) * (SQ + SG) + SQ + 2 * pad
                orect(s, x, y, w, h, "s-fg", 1.2, rx=8, dash="4 3")
    else:                                         # 자기지도: 한 장의 영상을 패치로 나누고 일부를 가림
        P = 28
        ix, iy = cx - 2 * P, PY0 + (PH - 3 * P) / 2
        masked = {(0, 1), (1, 3), (2, 0), (2, 2)}
        for r in range(3):
            for c in range(4):
                x, y = ix + c * P, iy + r * P
                if (r, c) in masked:
                    s.rect(x, y, P, P, "s-mu f-mu", 1)
                    s.text(x + P / 2, y + P / 2 + 5, "?", "f-bg", 14, weight="700")
                else:
                    s.rect(x, y, P, P, "s-mu f-acs", 1)
        orect(s, ix, iy, 4 * P, 3 * P, "s-fg", 1.4)
    s.text(cx, PY0 + PH + 22, goals[k][0], "f-fg", 12)
    s.text(cx, PY0 + PH + 40, goals[k][1], "f-mu", 11.5)
# 아래 축: 사람이 붙인 레이블의 양
ya = 252
arrow(s, 60, ya, 620, ya, "s-mu", "f-mu", 1.4)
s.text(60, ya + 17, "사람이 붙인 레이블 많음", "f-mu", 11.5, anchor="start")
s.text(620, ya + 17, "없음", "f-mu", 11.5, anchor="end")
s.save(os.path.join(OUT, "13-paradigms.svg"))


# ---------------------------------------------------------------- 2. 학습과 추론의 흐름
s = Svg(680, 305, "학습 단계(레이블과 손실로 파라미터를 갱신)와 추론 단계(고정된 모델로 새 장면을 예측)의 흐름")


def box(x, y, w, h, t1, t2, cls="s-fg f-acs"):
    s.rect(x, y, w, h, cls, 1.4, rx=6)
    s.text(x + w / 2, y + h / 2 - 3, t1, "f-fg", 13, weight="600")
    s.text(x + w / 2, y + h / 2 + 14, t2, "f-mu", 11.5)


Y1, H = 62, 54
s.text(20, 22, "학습 (training)", "f-fg", 13, anchor="start", weight="700")
box(20, Y1, 130, H, "학습 자료", "픽셀 + 레이블")
box(200, Y1, 110, H, "모델", "파라미터 조정 중")
box(360, Y1, 100, H, "예측", "클래스 확률", "s-fg f-bg")
box(510, Y1, 150, H, "손실함수", "예측과 레이블 비교", "s-bd f-bds")
ym = Y1 + H / 2
arrow(s, 150, ym, 200, ym)
arrow(s, 310, ym, 360, ym)
arrow(s, 460, ym, 510, ym)
# 레이블은 손실함수로 바로 감 (위쪽 점선)
s.add(f'<path d="M 120 {Y1} L 120 42 L 600 42" class="s-mu" fill="none" stroke-width="1.3" stroke-dasharray="4 3"/>')
arrow(s, 600, 42, 600, Y1, "s-mu", "f-mu", 1.3, "4 3")
s.text(360, 36, "레이블", "f-mu", 11.5)
# 손실 → 파라미터 갱신 (아래쪽 되돌림)
yb = Y1 + H + 28
s.path(f"M 585 {Y1 + H} L 585 {yb} L 285 {yb}", "s-bd", 1.6)
arrow(s, 285, yb, 285, Y1 + H, "s-bd", "f-bd", 1.6)
s.text(435, yb + 18, "손실이 줄도록 파라미터 갱신 (반복)", "f-bd", 12, weight="600")
# 구분선
s.line(20, 185, 660, 185, "s-mu", 1, "2 4")
# 추론
Y2 = 235
s.text(20, 212, "추론 (inference)", "f-fg", 13, anchor="start", weight="700")
box(20, Y2, 130, H, "새 장면", "레이블 없음")
box(200, Y2, 110, H, "학습된 모델", "파라미터 고정")
box(360, Y2, 150, H, "토지피복도", "픽셀마다 예측", "s-ok f-bg")
ym2 = Y2 + H / 2
arrow(s, 150, ym2, 200, ym2)
arrow(s, 310, ym2, 360, ym2)
# 학습 → 추론: 파라미터 저장·불러오기
arrow(s, 230, Y1 + H, 230, Y2, "s-ac", "f-ac", 1.6, "5 3")
s.text(222, 160, "저장·불러오기", "f-ac", 11.5, anchor="end", weight="600")
s.text(530, ym2 - 3, "손실 계산 없음", "f-mu", 11.5, anchor="start")
s.text(530, ym2 + 14, "전처리는 학습과 같게", "f-mu", 11.5, anchor="start")
s.save(os.path.join(OUT, "13-train-infer.svg"))
print("ok")


# ---------------------------------------------------------------- 본문 수치 재현 (그림과 무관, 확인용 출력)
if __name__ == "__main__" and "--numbers" in sys.argv:
    import warnings
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.semi_supervised import SelfTrainingClassifier
    from sklearn.tree import DecisionTreeClassifier
    warnings.filterwarnings("ignore")
    D = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "part3_landcover.csv"))
    F = ["B2", "B3", "B4", "B8", "B11", "B12", "ndvi", "ndwi"]
    # 2절 비지도: 밴드 6개 표준화 → K-평균 6군집과 실제 클래스의 교차표
    km = KMeans(6, n_init=10, random_state=0).fit(StandardScaler().fit_transform(D[F[:6]]))
    print(pd.crosstab(km.labels_, D.cls))
    # 4절 용량 표와 2절 준지도: 클래스마다 폴리곤 1/4을 떼어 두는 일을 시드 0~19로 20번
    deps = [1, 2, 4, 6, 8, None]
    cap = {d: [] for d in deps}
    semi = []
    ymap = {c: i for i, c in enumerate(sorted(D.cls.unique()))}
    for seed in range(20):
        rng = np.random.default_rng(seed)
        test_p, lab_p = [], []
        for c, g in D.groupby("cls"):
            ps = np.sort(g.patch_id.unique())
            t = rng.choice(ps, max(1, round(len(ps) * 0.25)), replace=False)
            test_p += list(t)
            lab_p += list(rng.choice([p for p in ps if p not in t], 1, replace=False))
        te = D.patch_id.isin(test_p); tr = ~te; lab = D.patch_id.isin(lab_p)
        for d in deps:
            m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(D.loc[tr, F], D.cls[tr])
            cap[d].append((m.get_n_leaves(), m.score(D.loc[tr, F], D.cls[tr]), m.score(D.loc[te, F], D.cls[te])))
        lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        yte = D.cls[te].map(ymap)
        few = lr().fit(D.loc[lab, F], D.cls[lab].map(ymap)).score(D.loc[te, F], yte)
        yi = np.where(lab[tr], D.cls[tr].map(ymap), -1)
        st = SelfTrainingClassifier(lr(), threshold=0.9).fit(D.loc[tr, F], yi)
        added = st.labeled_iter_ > 0
        wrong = (st.transduction_[added] != D.cls[tr].map(ymap).values[added]).mean()
        full = lr().fit(D.loc[tr, F], D.cls[tr].map(ymap)).score(D.loc[te, F], yte)
        semi.append((lab.sum(), few, st.score(D.loc[te, F], yte), full, added.sum(), wrong))
    for d in deps:
        print("depth", d, np.array(cap[d]).mean(0).round(4))
    a = np.array(semi)
    print("semi mean (labeled px, few, self-training, full):", a[:, :4].mean(0).round(3))
    print("self-training better in", int((a[:, 2] > a[:, 1]).sum()), "of 20; worst:", a[np.argmin(a[:, 2] - a[:, 1])].round(3))
