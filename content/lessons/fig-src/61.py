"""61강 그림과 본문 수치 검산

그림: 시험·연무·계절 자료의 2×2 오차 상속 행렬(플립 표), 5대 게이트 판정 도식, 섀도우·카나리 배포 도식(모두 SVG)
수치는 data/part8_runs.json의 ops 실험(run_part8.py exp_ops)에서 읽음. 57강 robust, 55강 calib, 51강 prx 값도 대조.
검산: N11·N10·N01·N00 합과 정확도, 순증, 회귀율(RR), 전체 대비 플립 비율, 맥니마 정확 검정(17강), 게이트 판정 재현, 지연시간 비율
    python3 61.py
"""
import json, math, os, sys
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from svglib import Svg

OUT = os.path.join(HERE, "..", "fig")
R = json.load(open(os.path.join(HERE, "..", "data", "part8_runs.json"), encoding="utf-8"))
OPS = R["ops"]
G = OPS["gate"]


# ================================================================ 검산
def check():
    for sp in ("test", "shift"):
        o = OPS[sp]
        n = o["N11"] + o["N10"] + o["N01"] + o["N00"]
        accA = (o["N11"] + o["N10"]) / n; accB = (o["N11"] + o["N01"]) / n
        rr = 100 * o["N10"] / (o["N11"] + o["N10"])
        nfr = 100 * o["N10"] / n
        p = stats.binomtest(min(o["N10"], o["N01"]), o["N10"] + o["N01"], 0.5).pvalue
        print(f"{sp}: N={n} accA {accA:.4f}(JSON {o['acc_A']}) accB {accB:.4f}(JSON {o['acc_B']}) 순증 {o['N01'] - o['N10']} "
              f"= {100 * (accB - accA):+.2f}%p, RR {rr:.2f}% (JSON {o['RR_pct']}), N10/N {nfr:.2f}%, 맥니마 정확 p {p:.4f}")
        print("   N10 클래스별", o["N10_per_class"], "합", sum(o["N10_per_class"].values()))
    mA, mB, t = G["metrics"]["A"], G["metrics"]["B"], G["thresholds"]
    drops = G["slice_drop_A_to_B"]
    print("A", {k: mA[k] for k in ("acc", "max_rdi", "worst_rdi", "mce", "ece", "min_slice")})
    print("B", {k: mB[k] for k in ("acc", "max_rdi", "worst_rdi", "mce", "ece", "min_slice")})
    print("최대 슬라이스 하락", max(drops.values()), max(drops, key=drops.get), "/ 연무·계절 변화", -drops["연무·계절 시험"])
    lat = G["latency_ms_B"]
    gates = dict(G_macro=mB["acc"] >= mA["acc"],
                 G_slice=mB["min_slice"] >= t["slice_min"] and max(drops.values()) <= t["slice_drop"],
                 G_robust=mB["mce"] <= t["mce"] and mB["max_rdi"] <= t["max_rdi"],
                 G_calib=mB["ece"] <= t["ece"], G_latency=lat["p99"] <= t["p99_ms"])
    print("게이트 재현", gates, "JSON", G["gates"], G["decision"])
    # 현재 운영 모델 A를 같은 절대 기준으로 재면
    print("A 절대 기준: slice", mA["min_slice"] >= t["slice_min"], "robust", mA["mce"] <= t["mce"] and mA["max_rdi"] <= t["max_rdi"],
          "calib", mA["ece"] <= t["ece"])
    print("57강 대조: A max_rdi", R["robust"]["vdcnn"]["max_rdi"], R["robust"]["vdcnn"]["worst"], "mCE", R["robust"]["vdcnn"]["mce"])
    print("55강 대조: A 연무·계절 ECE", R["calib"]["shift"]["ece"], "드롭아웃 모델", R["calib"]["dropout_model"])
    rule = [r for r in R["prx"]["rules"] if r["metric"] == "max_rdi"][0]
    print("51강 규칙", rule["id"], rule["priority"], rule["range"], rule["action"], "→ B 0.5405 걸림:", mB["max_rdi"] >= rule["range"][0])
    print("지연시간(ms)", lat, "p99/p50 %.2f max/p50 %.2f p99/mean %.2f" % (lat["p99"] / lat["p50"], lat["max"] / lat["p50"], lat["p99"] / lat["mean"]))
    print("2,000회 중 p99 위에 놓이는 횟수 약", round(2000 * 0.01))
    print("25 ms → FPS", 1000 / 25, "/ 25 ms × 1.15 =", 25 * 1.15)


# ================================================================ 그림
def arrow(s, x1, y1, x2, y2, cls="s-fg", fcls="f-fg", width=1.5, dash=None, head=7):
    """직선 화살표 (marker id를 쓰지 않으려고 화살촉을 삼각형 경로로 그림)"""
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
    s.line(x1, y1, bx, by, cls, width, dash)
    px, py = -math.sin(ang) * head * 0.5, math.cos(ang) * head * 0.5
    s.path(f"M {x2:.1f} {y2:.1f} L {bx + px:.1f} {by + py:.1f} L {bx - px:.1f} {by - py:.1f} Z", fcls, 0)


def fig_flip():
    s = Svg(720, 226, "기존 모델 A와 새 모델 B의 오차 상속 행렬. 시험 자료에서는 플립 오차 10건, 신규 해결 4건이고, 연무·계절 자료에서는 플립 오차 26건, 신규 해결 93건")
    cw, ch = 110, 58
    for x0, sp, title in ((112, "test", "시험 900장"), (476, "shift", "연무·계절 900장")):
        o = OPS[sp]
        s.text(x0 + cw, 22, title, size=14, weight="bold")
        s.text(x0 + cw / 2, 50, "B 맞힘", "f-mu", 12); s.text(x0 + cw * 1.5, 50, "B 틀림", "f-mu", 12)
        s.text(x0 - 10, 62 + ch / 2 + 4, "A 맞힘", "f-mu", 12, "end"); s.text(x0 - 10, 62 + ch * 1.5 + 4, "A 틀림", "f-mu", 12, "end")
        cells = [(0, 0, o["N11"], "성공 보존", "s-mu f-sf"), (1, 0, o["N10"], "플립 오차", "s-bd f-bds"),
                 (0, 1, o["N01"], "신규 해결", "s-ac f-acs"), (1, 1, o["N00"], "둘 다 틀림", "s-mu f-sf")]
        for cx, cy, v, lab, cls in cells:
            x, y = x0 + cx * cw, 62 + cy * ch
            s.rect(x, y, cw, ch, cls, 1.2)
            s.text(x + cw / 2, y + 27, f"{v}", "f-bd" if cx == 1 and cy == 0 else "f-fg", 18, weight="bold")
            s.text(x + cw / 2, y + 46, lab, "f-mu", 11)
        net = o["N01"] - o["N10"]
        s.text(x0 + cw, 204, f"순증 {net:+d} · 회귀율 {o['RR_pct']:.2f}%".replace("-", "−"), size=13)
    s.save(os.path.join(OUT, "61-flip.svg"))


def fig_gate():
    mA, mB, t = G["metrics"]["A"], G["metrics"]["B"], G["thresholds"]
    lat = G["latency_ms_B"]; g = G["gates"]
    drop = max(G["slice_drop_A_to_B"].values())
    boxes = [("G_macro", "정확도(시험)", [f"{mB['acc']:.3f} < A {mA['acc']:.3f}"], g["G_macro"]),
             ("G_slice", "최저 슬라이스", [f"{mB['min_slice']:.3f} < {t['slice_min']:.2f}", f"최대 하락 {drop:.3f}"], g["G_slice"]),
             ("G_robust", "mCE · max RDI", [f"{mB['mce']:.1f} ≤ {t['mce']:.0f}", f"{mB['max_rdi']:.2f} > {t['max_rdi']:.2f}"], g["G_robust"]),
             ("G_calib", "ECE(시험)", [f"{mB['ece']:.3f} ≤ {t['ece']:.2f}"], g["G_calib"]),
             ("G_latency", "p99 지연", [f"{lat['p99']:.3f} ≤ {t['p99_ms']:.1f} ms"], g["G_latency"])]
    W, bw, gap, x0 = 760, 134, 12, 21
    s = Svg(W, 330, "새 모델 B의 5대 게이트 판정. 거시 성능·최저 슬라이스·강건성 게이트가 탈락하고 보정·지연 게이트는 통과해, 논리곱 결과는 No-Go")
    s.rect(W / 2 - 90, 8, 180, 34, "s-fg f-acs", 1.3, 6)
    s.text(W / 2, 30, "새 모델 B (드롭아웃)", size=13, weight="bold")
    ytop, bh = 78, 128
    for i, (name, metric, lines, ok) in enumerate(boxes):
        x = x0 + i * (bw + gap); cx = x + bw / 2
        s.line(W / 2, 42, W / 2, 58, "s-mu", 1.2); s.line(min(W / 2, cx), 58, max(W / 2, cx), 58, "s-mu", 1.2)
        arrow(s, cx, 58, cx, ytop, "s-mu", "f-mu", 1.2, head=6)
        s.rect(x, ytop, bw, bh, "s-ok f-sf" if ok else "s-bd f-bds", 1.4, 6)
        s.text(cx, ytop + 22, name, size=13, weight="bold")
        s.text(cx, ytop + 42, metric, "f-mu", 11)
        for j, ln in enumerate(lines):
            s.text(cx, ytop + 64 + j * 18, ln, size=12)
        s.text(cx, ytop + 114, "통과" if ok else "탈락", "f-ok" if ok else "f-bd", 14, weight="bold")
        arrow(s, cx, ytop + bh, W / 2 + (cx - W / 2) * 0.12, 250, "s-ok" if ok else "s-bd", "f-ok" if ok else "f-bd", 1.3, head=6)
    s.add(f'<circle cx="{W / 2:.1f}" cy="262.0" r="13" class="s-fg f-sf" stroke-width="1.3"/>')
    s.text(W / 2, 267, "∧", size=15, weight="bold")
    arrow(s, W / 2, 275, W / 2, 290, "s-fg", "f-fg", 1.4, head=6)
    s.rect(W / 2 - 60, 290, 120, 32, "s-bd f-bds", 1.5, 6)
    s.text(W / 2, 311, G["decision"], "f-bd", 15, weight="bold")
    s.text(W / 2 + 80, 266, "하나라도 탈락이면 0", "f-mu", 11, "start")
    s.save(os.path.join(OUT, "61-gate.svg"))


def fig_canary():
    s = Svg(720, 250, "섀도우 배포는 입력을 복제해 새 모델에도 넣지만 결과는 비교·기록만 하고, 카나리 배포는 입력 일부를 새 모델이 실제로 처리해 사용자에게 보냄")
    for px, title, canary in ((0, "섀도우", False), (370, "카나리", True)):
        s.text(px + 175, 22, title, size=14, weight="bold")
        s.rect(px + 12, 102, 62, 36, "s-fg f-sf", 1.3, 5); s.text(px + 43, 125, "입력", size=12)
        s.rect(px + 128, 50, 92, 36, "s-fg f-sf", 1.3, 5); s.text(px + 174, 73, "운영 모델 A", size=12)
        s.rect(px + 128, 156, 92, 36, "s-ac f-acs", 1.3, 5); s.text(px + 174, 179, "새 모델 B", size=12)
        if not canary:
            arrow(s, px + 74, 112, px + 128, 72); s.text(px + 92, 84, "전부", "f-mu", 11)
            arrow(s, px + 74, 128, px + 128, 172, "s-ac", "f-ac", 1.5, "5 3"); s.text(px + 90, 168, "복제", "f-ac", 11)
            s.rect(px + 262, 50, 76, 36, "s-fg f-sf", 1.3, 5); s.text(px + 300, 73, "사용자", size=12)
            arrow(s, px + 220, 68, px + 262, 68)
            s.rect(px + 262, 156, 76, 36, "s-mu f-bg", 1.3, 5); s.text(px + 300, 179, "비교·기록", size=12)
            arrow(s, px + 220, 174, px + 262, 174, "s-ac", "f-ac", 1.5, "5 3")
            s.text(px + 175, 222, "B 결과는 밖으로 나가지 않음", "f-mu", 12)
        else:
            arrow(s, px + 74, 112, px + 128, 72); s.text(px + 92, 84, "90%", "f-mu", 11)
            arrow(s, px + 74, 128, px + 128, 172, "s-ac", "f-ac"); s.text(px + 90, 168, "10%", "f-ac", 11)
            s.rect(px + 262, 102, 76, 36, "s-fg f-sf", 1.3, 5); s.text(px + 300, 125, "사용자", size=12)
            arrow(s, px + 220, 72, px + 262, 112); arrow(s, px + 220, 172, px + 262, 128, "s-ac", "f-ac")
            s.text(px + 175, 222, "B 결과 일부가 실제로 쓰임 · 감시 후 확대", "f-mu", 12)
    s.line(355, 40, 355, 200, "s-mu", 1, "3 4")
    s.save(os.path.join(OUT, "61-canary.svg"))


if __name__ == "__main__":
    check()
    fig_flip(); fig_gate(); fig_canary()
