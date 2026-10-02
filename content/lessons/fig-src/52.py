"""52강 그림과 본문 수치 확인: 오차 판정 규칙 도식, 오차 개수 vs 오라클 ΔAP 막대, NMS 임계값별 중복·미탐(오차 교환 탄력도)

주 수치는 data/part8_runs.json의 tide(run_part8.py exp_tide·classify_dets)에서 읽음. 학습 없음.
JSON에 없는 본문 수치는 check()에서 run_part8의 classify_dets·nms_track과 6부 기본 탐지 결과로 직접 계산해 출력함.
- 오차 비율, 분류 오차의 클래스 짝, 클래스별 AP(분류 오라클), 정답 크기 구간별 상태
- 오라클(분류·위치)의 클래스별 AP와 여섯 가지를 한꺼번에 고친 mAP50(run_part8과 같은 TIDE 방식), 위치 오차 가운데 이미 찾은 정답 둘레의 두 번째 상자 수
- 위치 오차 정답의 한 변(√넓이) 중앙값, 고신뢰 배경 오탐 실험의 세부 개수
- NMS 임계값별 중복·미탐 개수(6부 기본 타일, 53강 밀집 계류 타일), 구간 탄력도(중간점 방식)
실행: python3 content/lessons/fig-src/52.py  (run_part8 import 때문에 10초 남짓)
"""
import contextlib, io, json, math, os, sys
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg

with contextlib.redirect_stdout(io.StringIO()):
    import run_part8 as RP          # classify_dets, nms_track, T6·RAW6·BASE6 (쓰기 없음)
from make_part6 import detect, obb_to_hbb
from part6_eval import coco_eval, iou_matrix, nms_dets

OUT = os.path.join(HERE, "..", "fig")
TD = json.load(open(os.path.join(DATA, "part8_runs.json"), encoding="utf-8"))["tide"]
ND = json.load(open(os.path.join(DATA, "part8_runs.json"), encoding="utf-8"))["nmsdiag"]
B = RP.BASE6
THR_BASE = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
THR_DENSE = (0.3, 0.4, 0.5, 0.6, 0.7)


def ap50(images):
    return coco_eval(images, thrs=np.array([0.5]))


def dense_tiles():
    """run_part8.exp_nmsdiag의 밀집 계류 타일 6장을 같은 시드로 다시 만듦(53강)"""
    rng = np.random.default_rng(81)
    out = []
    for k in range(6):
        objs = []
        for row in range(3):
            for i in range(8):
                L = rng.uniform(12, 16) / 0.5; W = L * 0.32
                th = 45 + rng.normal(0, 2)
                step = W + 2.0
                cx = 120 + row * 130 + i * step / math.sqrt(2) + rng.normal(0, 0.5)
                cy = 110 + i * step / math.sqrt(2) + row * 60 + rng.normal(0, 0.5)
                o = dict(cls=1, cx=cx, cy=cy, w=L, h=W, theta=th)
                o["obb"] = [cx, cy, L, W, th]; o["hbb"] = obb_to_hbb((cx, cy, L, W, th)).tolist()
                hb = o["hbb"]; o["area"] = (hb[2] - hb[0]) * (hb[3] - hb[1])
                objs.append(o)
        out.append((objs, detect(objs, [], 1.0, seed=500 + k)))
    return out


def sweep(scenes, thrs):
    """NMS 임계값별 (중복 오차 수, 미탐 오차 수, 짝 못 지은 정답 수)"""
    res = {}
    for thr in thrs:
        d = m = nontp = 0
        for objs, raw in scenes:
            keep, _ = RP.nms_track(raw, thr)
            _, ty, _, gs = RP.classify_dets(objs, [raw[i] for i in keep])
            d += ty.count("Dupe"); m += gs.count("Miss"); nontp += len(gs) - gs.count("TP")
        res[thr] = (d, m, nontp)
    return res


def arc_elasticity(a, b):
    """중간점(호) 탄력도: −(ΔD/D̄)/(ΔM/M̄). 미탐이 그대로면 무한대, 둘 다 0이면 정의 안 됨"""
    (D0, M0), (D1, M1) = a, b
    if D0 + D1 == 0:
        return None
    dD = (D1 - D0) / ((D0 + D1) / 2); dM = (M1 - M0) / ((M0 + M1) / 2)
    return math.inf if dM == 0 else -dD / dM


BASE_SW = sweep([(t["objs"], r) for t, r in zip(RP.T6, RP.RAW6)], THR_BASE)
DENSE_SW = sweep(dense_tiles(), THR_DENSE)


# ================================================================ 본문 수치 확인
def check():
    c = TD["counts"]; fp = TD["fp_total"]
    print("탐지", sum(c.values()), "정답", TD["n_gt"], "TP", c["TP"], "Miss", TD["miss"], "Covered", TD["covered_gt"], "base", TD["base_map50"])
    print("오탐 구성(%)", {k: round(100 * c[k] / fp, 1) for k in c if k != "TP"})
    hi = TD["counts_score_ge_0_5"]; fph = sum(v for k, v in hi.items() if k != "TP")
    print("점수 ≥ 0.5 오탐", fph, {k: round(100 * hi[k] / fph, 1) for k in hi if k != "TP"})
    dap = TD["delta_ap50_oracle"]
    print("ΔAP 합 %.4f, 1 − mAP50 %.4f" % (sum(dap.values()), 1 - TD["base_map50"]))
    # 분류 오차의 클래스 짝(정답, 예측)과 클래스별 AP
    pairs = Counter(); loc_on_matched = loc_low = 0; iou_tp = []; side = []
    gstat = Counter()
    tide_like = {"Cls": [], "Loc": []}
    simple_all = []
    for gts, dets in B:
        ds, ty, tg, gs = RP.classify_dets(gts, dets)
        tp_of = {j: d for d, t, j in zip(ds, ty, tg) if t == "TP"}
        for d, t, j in zip(ds, ty, tg):
            if t == "Cls":
                pairs[(gts[j]["cls"], d["cls"])] += 1
            if t == "Loc":
                g = gts[j]["hbb"]; side.append(math.sqrt((g[2] - g[0]) * (g[3] - g[1])))
                if j in tp_of:
                    loc_on_matched += 1; loc_low += d["score"] < tp_of[j]["score"]
                    iou_tp.append(iou_matrix([d["hbb"]], [tp_of[j]["hbb"]])[0, 0])
        for g, st in zip(gts, gs):
            gstat[("S" if g["area"] < 32 ** 2 else "M+", st)] += 1
        owner = {}                               # 짝 없는 정답마다 그것을 가리키는 Cls·Loc 오차 중 점수 1위(tidecv BestGTMatch)
        for ii, (t, j) in enumerate(zip(ty, tg)):
            if t in ("Cls", "Loc") and gs[j] != "TP":
                owner.setdefault(j, ii)
        for k in tide_like:                      # run_part8과 같은 TIDE 오라클(짝 있는 정답을 가리키거나 점수 1위가 아니면 지움)
            nd, used = [], {jj for jj, st in enumerate(gs) if st == "TP"}
            for ii, (d, t, j) in enumerate(zip(ds, ty, tg)):
                d2 = dict(d)
                if t == k:
                    if j in used or owner.get(j) != ii:
                        continue
                    used.add(j)
                    if k == "Cls":
                        d2["cls"] = gts[j]["cls"]
                    else:
                        d2["hbb"] = list(gts[j]["hbb"])
                nd.append(d2)
            tide_like[k].append((gts, nd))
        nd, used = [], {jj for jj, st in enumerate(gs) if st == "TP"}    # 한꺼번에: 점수 순 첫 Cls·Loc 오차 = 점수 1위만 고침
        for d, t, j in zip(ds, ty, tg):
            d2 = dict(d)
            if t in ("Cls", "Loc"):
                if j in used:
                    continue
                used.add(j)
                if t == "Cls":
                    d2["cls"] = gts[j]["cls"]
                else:
                    d2["hbb"] = list(gts[j]["hbb"])
            elif t in ("Both", "Dupe", "Bkg"):
                continue
            nd.append(d2)
        simple_all.append(([g for g, st in zip(gts, gs) if st != "Miss"], nd))
    print("분류 오차 짝(정답→예측; 0 선박 1 소형선박 2 차량)", dict(pairs))
    base = ap50(B)
    print("기본 클래스별 AP50", {k: v[0] for k, v in base["per_class"].items()})
    for k in tide_like:
        e = ap50(tide_like[k])
        print("TIDE 오라클", k, "ΔAP %.4f" % (e["ap50"] - base["ap50"]), "JSON", TD["delta_ap50_oracle"][k], {q: v[0] for q, v in e["per_class"].items()})
    print("여섯 가지 한꺼번에 mAP50 %.4f, maxDets 1000이면 %.4f" % (ap50(simple_all)["ap50"], coco_eval(simple_all, thrs=np.array([0.5]), max_dets=1000)["ap50"]))
    print("JSON on_matched_gt", TD["loc"].get("on_matched_gt"))
    only = []
    for gts, dets in B:
        ds, ty, tg, gs = RP.classify_dets(gts, dets)
        used = {jj for jj, st in enumerate(gs) if st == "TP"}
        only.append((gts, [d for d, t, j in zip(ds, ty, tg) if not (t == "Loc" and j in used)]))
    print("두 번째 상자 190개만 지운 ΔAP %.4f" % (ap50(only)["ap50"] - base["ap50"]))
    print("위치 오차 중 이미 찾은 정답 둘레", loc_on_matched, "/", TD["loc"]["n"], "그중 점수가 TP보다 낮음", loc_low,
          "TP 상자와 IoU 중앙값 %.3f, 10~90%% %s" % (np.median(iou_tp), np.round(np.percentile(iou_tp, [10, 90]), 3)))
    L = TD["loc"]
    print("위치 오차", L, "중심 우세 개수", round(L["center_dominant_frac"] * L["n"]),
          "정답 한 변 중앙값 %.2f화소 → 중심 어긋남 약 %.2f화소" % (np.median(side), L["center_median"] * np.median(side)),
          "ln1.25 = %.3f" % math.log(1.25))
    print("정답 크기·상태", dict(gstat))
    s_gt = sum(v for (z, _), v in gstat.items() if z == "S"); m_gt = sum(v for (z, _), v in gstat.items() if z == "M+")
    print("소형 정답", s_gt, "미탐률 %.3f" % (gstat[("S", "Miss")] / s_gt), "중형 이상", m_gt, "미탐률 %.3f" % (gstat[("M+", "Miss")] / m_gt))
    print("크기별(JSON)", TD["by_size"])
    # 고신뢰 배경 오탐: 점수 분포와 지운 라벨 수
    sc = np.array([d["score"] for _, ds in B for d in ds])
    print("점수 ≥ 0.5 / 0.7 / 0.85 탐지 수", [(sc >= t).sum() for t in (0.5, 0.7, 0.85)], "전체", len(sc))
    rng = np.random.default_rng(5)
    dropped = [([o for o in g if rng.random() >= 0.2], d) for g, d in B]
    print("지운 정답", TD["n_gt"] - sum(len(g) for g, _ in dropped))
    for thr in (0.5, 0.7, 0.85):
        row = []
        for imgs in (B, dropped):
            hi_b = allfp = 0
            for gts, dets in imgs:
                ds, ty, _, _ = RP.classify_dets(gts, dets)
                for d, t in zip(ds, ty):
                    allfp += t != "TP"; hi_b += (t == "Bkg" and d["score"] >= thr)
            row.append((hi_b, allfp, round(hi_b / allfp, 4)))
        print("점수 ≥", thr, "원래 / 20% 지움", row, "JSON", TD["high_conf_bkg_ratio"]["by_threshold"][str(thr)])
    # 오차 교환 탄력도
    for name, sw, thrs in (("기본 타일", BASE_SW, THR_BASE), ("밀집 계류", DENSE_SW, THR_DENSE)):
        print(name, {t: sw[t] for t in thrs})
        for a, b in zip(thrs, thrs[1:]):
            print("   ", a, "→", b, "탄력도", arc_elasticity(sw[a][:2], sw[b][:2]))
        print("    중복+미탐(가중치 1:1)", {t: sw[t][0] + sw[t][1] for t in thrs})
    print("53강 JSON 밀집 계류 짝 못 지은 정답", {t: ND["dense_mooring"][f"nms_{t}"]["misses"] for t in (0.3, 0.5, 0.7)})
    print("기본 타일 NMS 0.5 재현:", BASE_SW[0.5], "JSON Dupe", TD["counts"]["Dupe"], "Miss", TD["miss"])


# ================================================================ 그림 1: 판정 규칙
def fig_rules():
    s = Svg(680, 300, "탐지 하나를 가장 많이 겹친 정답과의 IoU(0.1 미만, 0.1~0.5, 0.5 이상)와 클래스 일치 여부로 여섯 유형에 배정하는 규칙")
    X = [150, 290, 440, 660]          # 열 경계: IoU < 0.1 | 0.1~0.5 | ≥ 0.5
    Y = [56, 126, 196]                # 행 경계: 같은 클래스 | 다른 클래스
    s.text((X[0] + X[1]) / 2, 26, "IoU < 0.1", "f-mu", 13)
    s.text((X[1] + X[2]) / 2, 26, "0.1 ≤ IoU < 0.5", "f-mu", 13)
    s.text((X[2] + X[3]) / 2, 26, "IoU ≥ 0.5", "f-mu", 13)
    s.text((X[0] + X[3]) / 2, 46, "← 가장 많이 겹친 정답과의 IoU →", "f-mu", 11)
    s.text(X[0] - 12, 96, "같은 클래스", "f-fg", 13, anchor="end")
    s.text(X[0] - 12, 166, "다른 클래스", "f-fg", 13, anchor="end")
    # 배경: 두 행을 합친 칸
    s.rect(X[0], Y[0], X[1] - X[0], Y[2] - Y[0], "s-mu f-bds", 1.2)
    s.text((X[0] + X[1]) / 2, 120, "배경 오인", "f-bd", 14, weight="600")
    s.text((X[0] + X[1]) / 2, 140, "Bkg", "f-mu", 12)
    cells = [((1, 0), "위치 오차", "Loc", "f-acs"), ((1, 1), "분류·위치 동시", "Both", "f-acs"), ((2, 1), "분류 오차", "Cls", "f-acs")]
    for (cx, cy), ko, en, fill in cells:
        s.rect(X[cx], Y[cy], X[cx + 1] - X[cx], Y[cy + 1] - Y[cy], "s-mu " + fill, 1.2)
        s.text((X[cx] + X[cx + 1]) / 2, Y[cy] + 32, ko, "f-ac", 14, weight="600")
        s.text((X[cx] + X[cx + 1]) / 2, Y[cy] + 52, en, "f-mu", 12)
    # 같은 클래스 · IoU ≥ 0.5: 정답 짝 여부로 TP / 중복
    x0, x1, xm = X[2], X[3], (X[2] + X[3]) / 2
    s.rect(x0, Y[0], x1 - x0, Y[1] - Y[0], "s-mu f-sf", 1.2)
    s.line(xm, Y[0] + 8, xm, Y[1] - 8, "s-mu", 1.0, "3 3")
    s.text((x0 + xm) / 2, Y[0] + 28, "정탐 TP", "f-ok", 14, weight="600")
    s.text((x0 + xm) / 2, Y[0] + 48, "정답이 아직 비었음", "f-mu", 11)
    s.text((xm + x1) / 2, Y[0] + 28, "중복 Dupe", "f-bd", 14, weight="600")
    s.text((xm + x1) / 2, Y[0] + 48, "이미 짝이 있음", "f-mu", 11)
    # 정답 쪽
    s.rect(X[0], 222, X[3] - X[0], 52, "s-bd f-bg", 1.2)
    s.text(X[0] + 14, 244, "정답 쪽: 끝까지 정탐 짝이 없는 정답 = 미탐(Miss)", "f-bd", 13, anchor="start", weight="600")
    s.text(X[0] + 14, 263, "단, 분류·위치 오차 상자가 가리킨 정답은 미탐에서 뺌(그 오차를 고치면 찾아지므로)", "f-mu", 11, anchor="start")
    s.text(X[0] - 12, 252, "놓친 정답", "f-fg", 13, anchor="end")
    s.save(os.path.join(OUT, "52-rules.svg"))


# ================================================================ 그림 2: 개수 vs 오라클 ΔAP
def fig_oracle():
    c = TD["counts"]; dap = TD["delta_ap50_oracle"]
    rows = [("Cls", "분류"), ("Miss", "미탐"), ("Loc", "위치"), ("Dupe", "중복"), ("Bkg", "배경 오인"), ("Both", "분류·위치")]
    cnt = {k: (TD["miss"] if k == "Miss" else c[k]) for k, _ in rows}
    s = Svg(680, 268, "여섯 오차 유형의 개수(왼쪽)와, 그 유형만 고쳤을 때 mAP50이 오르는 폭(오라클 ΔAP, 오른쪽). 오탐 중 가장 많은 위치 오차보다 개수가 적은 분류 오차의 ΔAP가 훨씬 큼")
    LX, L0, L1, R0, R1, YT, RH = 110, 128, 330, 410, 640, 44, 34
    s.text((L0 + L1) / 2, 24, "개수 (오탐은 탐지, 미탐은 정답)", "f-mu", 12)
    s.text((R0 + R1) / 2, 24, "그 유형만 고친 mAP50 상승 (ΔAP)", "f-mu", 12)
    cmax, dmax = 300, 0.12
    for i, (k, ko) in enumerate(rows):
        y = YT + i * RH
        s.text(LX, y + 17, ko, "f-fg", 13, anchor="end")
        w = (L1 - L0) * cnt[k] / cmax
        s.rect(L0, y + 4, w, 20, "f-mu", 0)
        s.text(L0 + w + 6, y + 19, f"{cnt[k]}", "f-mu", 12, anchor="start")
        w2 = (R1 - R0) * dap[k] / dmax
        cls = "f-bd" if dap[k] >= sorted(dap.values())[-2] else "f-ac"    # 상위 두 유형 강조
        s.rect(R0, y + 4, w2, 20, cls, 0)
        s.text(R0 + w2 + 6, y + 19, f"{dap[k]:.3f}", cls, 12, anchor="start", weight="600")
    yb = YT + len(rows) * RH + 2
    s.line(L0, YT - 2, L0, yb, "s-mu", 1.2)
    s.line(R0, YT - 2, R0, yb, "s-mu", 1.2)
    for v in (0, 100, 200, 300):
        x = L0 + (L1 - L0) * v / cmax
        s.line(x, yb, x, yb + 4, "s-mu", 1.0); s.text(x, yb + 17, f"{v}", "f-mu", 11)
    for v in (0, 0.04, 0.08, 0.12):
        x = R0 + (R1 - R0) * v / dmax
        s.line(x, yb, x, yb + 4, "s-mu", 1.0); s.text(x, yb + 17, f"{v:g}", "f-mu", 11)
    s.save(os.path.join(OUT, "52-oracle.svg"))


# ================================================================ 그림 3: NMS 임계값별 중복·미탐
def fig_elastic():
    s = Svg(680, 318, "NMS IoU 임계값을 올릴 때 중복 오차와 미탐 오차의 개수. 왼쪽 6부 기본 타일은 미탐이 거의 그대로이고 중복만 늘며, 오른쪽 밀집 계류 타일은 미탐이 줄고 중복이 늘어 둘이 맞바뀜")

    def panel(X0, X1, sw, thrs, ymax, yt, title):
        YT, YB = 50, 238
        sx = lambda v: X0 + (X1 - X0) * (v - 0.3) / 0.5
        sy = lambda v: YB - (YB - YT) * v / ymax
        s.text((X0 + X1) / 2, 22, title, "f-fg", 13, weight="600")
        s.rect(X0, YT, X1 - X0, YB - YT, "s-mu f-bg", 1.2)
        for v in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            s.line(sx(v), YB, sx(v), YB + 5, "s-mu", 1.2)
            s.text(sx(v), YB + 19, f"{v:.1f}", "f-mu", 11)
        for v in yt:
            s.line(X0 - 5, sy(v), X0, sy(v), "s-mu", 1.2)
            s.text(X0 - 8, sy(v) + 4, f"{v:g}", "f-mu", 11, anchor="end")
        s.line(sx(0.5), YT, sx(0.5), YB, "s-mu", 1.0, "3 3")
        for idx, lc, fc in ((0, "s-ac", "f-ac"), (1, "s-bd", "f-bd")):
            pts = [(sx(t), sy(sw[t][idx])) for t in thrs]
            s.path("M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts), lc, 2.2)
            for (x, y), t in zip(pts, thrs):
                s.circle(x, y, 3.5, fc)
        s.text((X0 + X1) / 2, YB + 38, "NMS IoU 임계값", "f-mu", 12)
        return sx, sy

    sx, sy = panel(70, 320, BASE_SW, THR_BASE, 1400, (0, 400, 800, 1200), "6부 기본 타일 24장")
    s.text(sx(0.5) + 8, sy(BASE_SW[0.5][0]) + 15, f"{BASE_SW[0.5][0]}", "f-ac", 11, anchor="start")
    s.text(sx(0.8) - 6, sy(BASE_SW[0.8][0]) - 8, f"{BASE_SW[0.8][0]}", "f-ac", 11, anchor="end")
    s.text(sx(0.8) - 4, sy(BASE_SW[0.8][1]) - 9, f"{BASE_SW[0.3][1]}→{BASE_SW[0.8][1]}", "f-bd", 11, anchor="end")
    sx, sy = panel(400, 630, DENSE_SW, THR_DENSE, 100, (0, 25, 50, 75, 100), "밀집 계류 타일 6장 (53강)")
    for t in (0.3, 0.5, 0.7):
        s.text(sx(t) + 6, sy(DENSE_SW[t][1]) - 7, f"{DENSE_SW[t][1]}", "f-bd", 11, anchor="start")
    s.text(sx(0.7) - 6, sy(DENSE_SW[0.7][0]) + 4, f"{DENSE_SW[0.7][0]}", "f-ac", 11, anchor="end")
    # 범례
    s.line(250, 308, 274, 308, "s-ac", 2.2); s.text(280, 312, "중복 오차", "f-ac", 12, anchor="start", weight="600")
    s.line(370, 308, 394, 308, "s-bd", 2.2); s.text(400, 312, "미탐 오차", "f-bd", 12, anchor="start", weight="600")
    s.save(os.path.join(OUT, "52-elastic.svg"))


if __name__ == "__main__":
    check()
    fig_rules()
    fig_oracle()
    fig_elastic()
