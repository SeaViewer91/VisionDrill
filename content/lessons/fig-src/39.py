"""39강 그림과 본문 수치 확인: 소형선박 PR 곡선과 보간 계단(SVG), IoU 임계값별 클래스 AP(SVG)

결과 수치는 data/part6_runs.json의 map·check(run_part6.py exp_map·exp_check)에서 읽음. 학습 없음.
PR 곡선 점은 기본 탐지 결과(클래스별 NMS 0.5)로 part6_eval.pr_points를 직접 불러 만듦(IoU 0.5, maxDets 100).
JSON에 없는 본문 수치(11점 보간의 재현율별 값, 최대 재현율, 방식 × maxDets 조합, 차량 예의 화소 넓이,
pycocotools에서 maxDets를 바꿨을 때의 요약 값)는 check()에서 계산해 출력함.
"""
import contextlib, io, json, os, sys
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg
from make_part6 import tiles, detect, TILE
from part6_eval import pr_points, ap_from_pr, nms_dets, IOU_THRS

OUT = os.path.join(HERE, "..", "fig")
R = json.load(open(os.path.join(DATA, "part6_runs.json"), encoding="utf-8"))
M = R["map"]
T = tiles()
BASE = [(t["objs"], nms_dets(detect(t["objs"], t["clutter"], 1.0, seed=t["image_id"]), 0.5)) for t in T]
NAMES = ["선박", "소형선박", "차량"]


def curve(c, max_dets=100):
    _, tp, n = pr_points(BASE, c, 0.5, max_dets=max_dets)
    tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
    rec, prec = tpc / n, tpc / (tpc + fpc)
    env = prec.copy()
    for i in range(len(env) - 2, -1, -1):
        env[i] = max(env[i], env[i + 1])
    return tp, n, rec, prec, env


# ================================================================ 본문 수치 확인
def check():
    for c in range(3):
        tp, n, rec, prec, env = curve(c)
        p11 = [float(env[np.searchsorted(rec, r)]) if np.searchsorted(rec, r) < len(env) else 0.0 for r in np.linspace(0, 1, 11)]
        print(NAMES[c], "정답", n, "탐지", len(tp), "최대 재현율 %.4f" % rec[-1], "11점", np.round(p11, 3))
        print("   방식별 AP50", {m: round(ap_from_pr(tp, n, m), 4) for m in ("voc11", "voc", "coco")}, "JSON", M[f"ap50_methods_{c}"])
    for md in (100, 300):
        for m in ("voc11", "voc", "coco"):
            v = [ap_from_pr(*pr_points(BASE, c, 0.5, max_dets=md)[1:], m) for c in range(3)]
            print("maxDets", md, m, "mAP50 %.4f" % np.mean(v))
    pc = M["coco"]["per_class"]
    for c in range(3):
        a = np.array(pc[str(c)])
        print(NAMES[c], "AP50 %.3f AP75 %.3f AP50-95 %.3f, AP75/AP50 %.2f" % (a[0], a[5], a.mean(), a[5] / a[0]))
    n = np.array(M["n_per_class"]); ap = np.array([pc[str(c)][0] for c in range(3)])
    print("매크로 %.4f, 정답 수 가중 %.4f" % (ap.mean(), (ap * n).sum() / n.sum()), M["macro_ap50"], M["weighted_ap50"])
    b = R["data"]["size_bins_small_medium_large"]
    print("크기 구간 정답 수(소·중·대)", [sum(b[str(c)][k] for c in range(3)) for k in range(3)])
    print("차량이 100대 넘는 타일", [v for v in R["data"]["vehicles_per_tile"] if v > 100])
    # 승용차 4.5 m × 1.8 m의 수평 박스 넓이(화소²): 축에 나란할 때, 45° 돌았을 때
    for g in (0.5, 0.1, 0.05):
        print("GSD", g, "축 나란 %.0f" % (4.5 * 1.8 / g ** 2), "45도 %.0f" % (((4.5 + 1.8) * np.cos(np.pi / 4)) ** 2 / g ** 2))
    # pycocotools에서 maxDets를 [1, 10, 300]으로 바꿨을 때 요약
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = {"images": [{"id": t["image_id"], "width": TILE, "height": TILE} for t in T],
          "categories": [{"id": c} for c in range(3)], "annotations": []}
    for t in T:
        for o in t["objs"]:
            x1, y1, x2, y2 = o["hbb"]
            gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": t["image_id"], "category_id": o["cls"],
                                      "bbox": [x1, y1, x2 - x1, y2 - y1], "area": o["area"], "iscrowd": 0})
    dt = [{"image_id": t["image_id"], "category_id": x["cls"], "score": x["score"],
           "bbox": [x["hbb"][0], x["hbb"][1], x["hbb"][2] - x["hbb"][0], x["hbb"][3] - x["hbb"][1]]}
          for t, (_, d) in zip(T, BASE) for x in d]
    with contextlib.redirect_stdout(io.StringIO()):
        C = COCO(); C.dataset = gt; C.createIndex()
        E = COCOeval(C, C.loadRes(dt), "bbox"); E.params.maxDets = [1, 10, 300]
        E.evaluate(); E.accumulate(); E.summarize()
    print("pycocotools maxDets [1,10,300] 요약", np.round(E.stats, 4))
    print("check", R["check"])


# ================================================================ 그림 1: PR 곡선과 보간 계단
def fig_pr():
    tp, n, rec, prec, env = curve(1)
    s = Svg(680, 340, "소형선박의 정밀도-재현율 곡선(IoU 0.5)과, 각 재현율 이상에서의 최대 정밀도로 깎은 보간 계단, 11점 보간 지점. 오른쪽은 재현율 0.17~0.25, 정밀도 0.975~1 확대")
    rm = rec[-1]
    # 계단 꼭짓점
    pts = [(0.0, env[0])]
    prev_r = 0.0
    for r, p in zip(rec, env):
        if r != prev_r:
            pts.append((prev_r, p))
            pts.append((r, p))
            prev_r = r
    pts.append((rm, env[-1]))

    def panel(X0, X1, YT, YB, rx, ry, ticks_x, ticks_y, main):
        sx = lambda v: X0 + (X1 - X0) * (v - rx[0]) / (rx[1] - rx[0])
        sy = lambda v: YB - (YB - YT) * (v - ry[0]) / (ry[1] - ry[0])
        s.rect(X0, YT, X1 - X0, YB - YT, "s-mu f-bg", 1.2)
        for v in ticks_x:
            s.line(sx(v), YB, sx(v), YB + 5, "s-mu", 1.2)
            s.text(sx(v), YB + 19, f"{v:g}", "f-mu", 12 if main else 11)
        for v in ticks_y:
            s.line(X0 - 5, sy(v), X0, sy(v), "s-mu", 1.2)
            s.text(X0 - 9, sy(v) + 4, f"{v:g}", "f-mu", 12 if main else 11, anchor="end")
        clip = lambda r, p: (min(max(r, rx[0]), rx[1]), min(max(p, ry[0]), ry[1]))
        st = [clip(r, p) for r, p in pts if rx[0] - 0.02 <= r <= rx[1] + 0.02]
        if main:
            step = " L ".join(f"{sx(r):.1f} {sy(p):.1f}" for r, p in pts)
            s.path(f"M {sx(0):.1f} {sy(0):.1f} L {step} L {sx(rm):.1f} {sy(0):.1f} Z", "f-acs", 0)
        else:
            step = " L ".join(f"{sx(r):.1f} {sy(p):.1f}" for r, p in st)
            s.path(f"M {sx(st[0][0]):.1f} {sy(ry[0]):.1f} L {step} L {sx(st[-1][0]):.1f} {sy(ry[0]):.1f} Z", "f-acs", 0)
        s.path("M " + step, "s-ac", 2.4 if main else 2.0)
        raw = [clip(r, p) for r, p in zip(rec, prec) if rx[0] <= r <= rx[1]]
        s.path("M " + " L ".join(f"{sx(r):.1f} {sy(p):.1f}" for r, p in raw), "s-fg" if not main else "s-mu", 1.0)
        for r in np.linspace(0, 1, 11):
            k = np.searchsorted(rec, r)
            p = float(env[k]) if k < len(env) else 0.0
            if rx[0] <= r <= rx[1] and ry[0] <= p <= ry[1]:
                s.circle(sx(r), sy(p), 4, "f-bd")
        return sx, sy

    sx, sy = panel(70, 390, 22, 282, (0, 1), (0, 1), (0, 0.2, 0.4, 0.6, 0.8, 1), (0, 0.2, 0.4, 0.6, 0.8, 1), True)
    s.text(230, 322, "재현율", "f-mu", 12)
    s.text(26, 152, "정밀도", "f-mu", 12)
    s.parts[-1] = s.parts[-1].replace("<text ", '<text transform="rotate(-90 26 152)" ')
    s.line(sx(rm), sy(env[-1]), sx(rm), sy(0), "s-ac", 1.2, "4 3")
    s.text(sx(rm) - 6, sy(0.30), f"최대 재현율 {rm:.3f}", "f-ac", 11, anchor="end")
    rx, ry = (0.17, 0.25), (0.975, 1.0)
    s.rect(sx(rx[0]), sy(ry[1]) - 4, sx(rx[1]) - sx(rx[0]), sy(ry[0]) - sy(ry[1]) + 8, "s-fg", 1.0)
    s.add(s.parts.pop().replace("/>", ' fill="none"/>'))
    # 확대 패널
    panel(470, 650, 22, 172, rx, ry, (0.18, 0.2, 0.22, 0.24), (0.98, 0.99, 1.0), False)
    s.text(560, 208, "확대 (네모 부분)", "f-mu", 11)
    # 범례
    LX, LY = 430, 236
    s.rect(LX, LY - 9, 26, 12, "f-acs", 0)
    s.text(LX + 34, LY + 2, f"계단 아래 넓이 = AP (101점 {M['ap50_methods_1']['coco']:.3f})", "f-fg", 12, anchor="start")
    s.line(LX, LY + 22, LX + 26, LY + 22, "s-fg", 1.0)
    s.text(LX + 34, LY + 26, "원래 곡선 (지그재그)", "f-mu", 12, anchor="start")
    s.line(LX, LY + 46, LX + 26, LY + 46, "s-ac", 2.4)
    s.text(LX + 34, LY + 50, "보간 계단", "f-ac", 12, anchor="start", weight="600")
    s.circle(LX + 13, LY + 70, 4, "f-bd")
    s.text(LX + 34, LY + 74, f"11점 보간 지점 (평균 {M['ap50_methods_1']['voc11']:.3f})", "f-bd", 12, anchor="start")
    s.save(os.path.join(OUT, "39-pr-interp.svg"))


# ================================================================ 그림 2: IoU 임계값별 클래스 AP
def fig_iou():
    pc = M["coco"]["per_class"]
    s = Svg(680, 320, "IoU 임계값을 0.50에서 0.95까지 올릴 때 선박·소형선박·차량의 AP와 세 클래스 평균(mAP)")
    X0, X1, YT, YB = 70, 420, 22, 262
    th = [float(t) for t in IOU_THRS]
    sx = lambda v: X0 + (X1 - X0) * (v - 0.5) / 0.45
    sy = lambda v: YB - (YB - YT) * v
    s.rect(X0, YT, X1 - X0, YB - YT, "s-mu f-bg", 1.2)
    for v in (0.5, 0.6, 0.7, 0.8, 0.9):
        s.line(sx(v), YB, sx(v), YB + 5, "s-mu", 1.2)
        s.text(sx(v), YB + 19, f"{v:.1f}", "f-mu", 12)
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1):
        s.line(X0 - 5, sy(v), X0, sy(v), "s-mu", 1.2)
        s.text(X0 - 9, sy(v) + 4, f"{v:g}", "f-mu", 12, anchor="end")
    s.text((X0 + X1) / 2, YB + 40, "IoU 임계값", "f-mu", 12)
    s.text(X0 - 44, (YT + YB) / 2, "AP", "f-mu", 12)
    s.parts[-1] = s.parts[-1].replace("<text ", f'<text transform="rotate(-90 {X0 - 44} {(YT + YB) / 2:.1f})" ')
    s.line(sx(0.75), YT, sx(0.75), YB, "s-mu", 1, "3 3")
    series = [("선박", pc["0"], "s-ok", "f-ok", None), ("소형선박", pc["1"], "s-ac", "f-ac", None),
              ("차량", pc["2"], "s-bd", "f-bd", None),
              ("세 클래스 평균", list(np.mean([pc[k] for k in ("0", "1", "2")], axis=0)), "s-fg", "f-fg", "6 4")]
    for name, ys, lc, fc, dash in series:
        s.path("M " + " L ".join(f"{sx(x):.1f} {sy(y):.1f}" for x, y in zip(th, ys)), lc, 2.2 if dash is None else 1.8)
        if dash:
            s.parts[-1] = s.parts[-1].replace("/>", f' stroke-dasharray="{dash}"/>')
        for x, y in zip(th, ys):
            s.circle(sx(x), sy(y), 3, fc)
    LX, LY = 452, 50
    s.text(LX + 120, LY - 18, "AP50 → AP75, 열 개 평균", "f-mu", 11, anchor="middle")
    for k, (name, ys, lc, fc, dash) in enumerate(series):
        yy = LY + 38 * k
        s.line(LX, yy, LX + 30, yy, lc, 2.2, dash)
        s.text(LX + 40, yy + 4, f"{name}", fc, 12, anchor="start", weight="600")
        s.text(LX + 40, yy + 19, f"{ys[0]:.3f} → {ys[5]:.3f}, {np.mean(ys):.3f}", "f-mu", 11, anchor="start")
    s.text(LX, LY + 180, "차량은 IoU를 조금만 올려도", "f-fg", 12, anchor="start")
    s.text(LX, LY + 200, "AP가 가장 빨리 떨어짐", "f-fg", 12, anchor="start")
    s.save(os.path.join(OUT, "39-iou-ap.svg"))


if __name__ == "__main__":
    check()
    fig_pr()
    fig_iou()
