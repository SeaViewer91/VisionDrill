"""51강 그림과 본문 수치: 4계층 진단 파이프라인 도식(SVG), 사례별 처방 발동 표(SVG)

- 본문 수치는 data/part8_runs.json의 prx(계산: data/run_part8.py의 exp_prx)와, 그 지표가 온 실험
  (tide 52강, nmsdiag 53강, slice 54강, calib 55강, repr 56강, robust 57강)에서 가져옴
- 여기서는 prx의 규칙 표(rules)로 규칙 엔진을 다시 돌려 JSON의 low·high 결과와 같은지 확인하고,
  범위를 원문보다 낮게 잡을 때의 예(라벨이 온전한 6부 기본 결과를 점수 기준 0.5로 잰 고신뢰 배경 오탐 비율)를 확인함
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "fig")
sys.path.insert(0, HERE)
from svglib import Svg

R = json.load(open(os.path.join(DATA, "part8_runs.json"), encoding="utf-8"))
P = R["prx"]
RULES = P["rules"]


def run(metrics, level):
    hits = []
    for r in RULES:
        v = metrics.get(r["metric"])
        if v is None:
            continue
        thr = r["range"][0] if level == "low" else r["range"][1]
        if (v >= thr) if r["op"] == ">=" else (v < thr):
            hits.append(r["id"])
    order = {"P0": 0, "P1": 1, "P2": 2}
    pr = {r["id"]: r["priority"] for r in RULES}
    return sorted(hits, key=lambda i: order[pr[i]])


# ---------------------------------------------------------------- 1. JSON 사례 재현
CASES = [k for k in P if k not in ("rules", "schema_example", "note")]
for k in CASES:
    m = P[k]["metrics"]
    for lv in ("low", "high"):
        mine = run(m, lv)
        js = [h["id"] for h in P[k][lv]]
        assert sorted(mine) == sorted(js), (k, lv, mine, js)
    print(k, m, "low:", run(m, "low"), "high:", run(m, "high"))

# 지표의 출처 값 확인
t, sl, c, rp, rb, nd = R["tide"], R["slice"], R["calib"], R["repr"], R["robust"], R["nmsdiag"]
hc = t["high_conf_bkg_ratio"]
print("고신뢰 배경 오탐 비율(점수 기준별):", hc["by_threshold"], "오탐 수 clean/drop20", hc["n_fp_clean"], hc["n_fp_drop20"])
print("6부 기본 mAP50", t["base_map50"], "오라클 ΔAP50", t["delta_ap50_oracle"])
print("밀집 계류 kill_ratio(NMS 0.5)", nd["dense_mooring"]["nms_0.5"]["kill_ratio"], "AP50", nd["dense_mooring"]["nms_0.5"]["ap50"],
      "Soft-NMS AP50", nd["dense_mooring"]["soft_nms_ap50"])
print("중심 우세 비율", t["loc"]["center_dominant_frac"], "Loc 오차 수", t["loc"]["n"])
print("분류기 시험 정확도", c["test"]["acc"], "ECE", c["test"]["ece"], "| 연무·계절", c["shift"]["acc"], "ECE", c["shift"]["ece"],
      "온도 스케일링 뒤 ECE", c["shift"]["ece_T"], "T", c["temperature"])
print("CUR", rp["base"]["cur"], "유효 랭크", rp["base"]["eff_rank"], "= /64", round(rp["base"]["eff_rank"] / 64, 4))
print("max RDI", rb["vdcnn"]["max_rdi"], rb["vdcnn"]["worst"], R["robust"]["nuisances"][rb["vdcnn"]["worst"]])
print("시험 최악 슬라이스", sl["test"]["worst"][0])
print("연무·계절 최악 슬라이스", sl["shift"]["worst"][0])

# ---------------------------------------------------------------- 2. 범위를 원문보다 낮게 잡으면(본문 5절)
# 라벨이 온전한 6부 기본 결과를 점수 기준 0.5로 잰 고신뢰 배경 오탐 비율은 0.057. 범위(0.10~0.15) 안에서는 발동하지 않지만,
# 원문보다 낮은 0.05로 잡으면 결함이 없는데도 P0가 나감(오경보)
extra = dict(P["detector_6부"]["metrics"], missing_gt_ratio=hc["by_threshold"]["0.5"]["clean"])
print("라벨 온전(점수 0.5 기준)", extra, "low:", run(extra, "low"), "high:", run(extra, "high"))
assert "RX-DATA-01" not in run(extra, "low")
print("기준 0.05로 낮추면 발동?", extra["missing_gt_ratio"] >= 0.05)
print("라벨 20% 누락을 노트 기준(점수 0.85)으로 재면", hc["by_threshold"]["0.85"]["drop20"], "→ low:",
      run(dict(P["detector_6부"]["metrics"], missing_gt_ratio=hc["by_threshold"]["0.85"]["drop20"]), "low"))
print("위치 오차 중 이미 짝이 있는 정답 둘레(덜 지워진 두 번째 상자)", t["loc"]["on_matched_gt"], "/", t["loc"]["n"])
print("mCE(VD-CNN)", rb["vdcnn"]["mce"])

# ---------------------------------------------------------------- 3. 그림 1: 4계층 파이프라인
s = Svg(760, 300, "4계층 진단 파이프라인: 입력, 진단 엔진 코어, 처방 규칙 엔진, 리포팅")
cols = [(10, 150), (196, 190), (422, 150), (608, 142)]   # (x, 폭)
heads = ["1 입력", "2 진단 엔진 코어", "3 처방 규칙 엔진", "4 리포팅"]
top, H = 18, 264
for (x, w), hd in zip(cols, heads):
    s.rect(x, top, w, H, cls="s-mu f-sf", width=1.2, rx=8)
    s.text(x + w / 2, top + 24, hd, size=14, weight="bold")
items = {
    0: [("예측", "NMS 전·후 상자·점수"), ("정답", "상자·마스크·클래스"), ("메타데이터", "촬영 조건·지역"), ("특성 맵(선택)", "중간 층·어텐션")],
    2: [("규칙과 대조", "지표 · 비교 · 기준"), ("등급 정렬", "P0 → P1 → P2"), ("처방 발행", "조치 · 근거 값")],
    3: [("JSON", "기계가 읽음"), ("요약 문서", "사람이 읽음")],
}
for ci, lst in items.items():
    x, w = cols[ci]
    bh, gap = 46, 10
    y0 = top + 40
    for i, (a, b) in enumerate(lst):
        y = y0 + i * (bh + gap)
        s.rect(x + 10, y, w - 20, bh, cls="s-mu f-bg", width=1, rx=5)
        s.text(x + w / 2, y + 19, a, size=13, weight="bold")
        s.text(x + w / 2, y + 37, b, cls="f-mu", size=11)
x, w = cols[1]
mods = [("A 오차 분해", "52·53강"), ("B 슬라이스·라벨", "54강"), ("C 표현·신뢰도", "55·56강")]
for i, (a, b) in enumerate(mods):
    y = top + 40 + i * 66
    s.rect(x + 10, y, w - 20, 56, cls="s-ac f-acs", width=1.4, rx=5)
    s.text(x + w / 2, y + 23, a, size=13, weight="bold")
    s.text(x + w / 2, y + 43, b, cls="f-mu", size=11)
s.text(x + w / 2, top + 40 + 3 * 66 + 14, "(57~60강 모듈로 확장)", cls="f-mu", size=11)
for (xa, wa), (xb, _) in zip(cols[:-1], cols[1:]):
    y = top + H / 2
    x1, x2 = xa + wa + 3, xb - 3
    s.line(x1, y, x2 - 6, y, cls="s-fg", width=2)
    s.path(f"M{x2 - 9:.1f},{y - 5:.1f} L{x2:.1f},{y:.1f} L{x2 - 9:.1f},{y + 5:.1f} Z", cls="s-fg f-fg", width=1)
x, w = cols[3]
s.text(x + w / 2, top + 40 + 2 * 56 + 22, "→ 배포 게이트(61강)", cls="f-ac", size=12)
s.save(os.path.join(OUT, "51-pipeline.svg"))

# ---------------------------------------------------------------- 4. 그림 2: 사례 × 규칙 발동 표
rule_cols = [("RX-DATA-01", "라벨 누락"), ("RX-DATA-02", "최악 슬라이스"), ("RX-MODEL-01", "NMS 억제"),
             ("RX-MODEL-02", "중심 어긋남"), ("RX-MODEL-03", "CUR"), ("RX-MODEL-04", "max RDI"), ("RX-INF-01", "ECE")]
rows = [("탐지 · 6부 기본", "점수 ≥ 0.85 기준", P["detector_6부"]["metrics"]),
        ("탐지 · 라벨 20% 누락", "점수 ≥ 0.5 기준", P["detector_라벨20%누락(점수기준0.5)"]["metrics"]),
        ("탐지 · 밀집 계류", "NMS IoU 0.5", P["detector_밀집계류"]["metrics"]),
        ("분류 · 시험", "900장", P["classifier_시험"]["metrics"]),
        ("분류 · 연무·계절 시험", "900장", P["classifier_연무·계절"]["metrics"])]
rmap = {r["id"]: r for r in RULES}
LW, CW, RH, HY = 168, 78, 46, 62
W = LW + CW * len(rule_cols) + 10
Hh = HY + RH * len(rows) + 58
s = Svg(W, Hh, "사례 다섯 개에서 규칙 일곱 개가 임계 범위의 두 끝에서 발동하는지 나타낸 표")
for j, (rid, nm) in enumerate(rule_cols):
    cx = LW + CW * j + CW / 2
    s.text(cx, 22, rmap[rid]["priority"], cls="f-mu", size=12, weight="bold")
    s.text(cx, 40, nm, size=12)
    rg = rmap[rid]["range"]
    s.text(cx, 56, ("≥ " if rmap[rid]["op"] == ">=" else "< ") + f"{rg[0]:.2f}~{rg[1]:.2f}", cls="f-mu", size=10.5)
for i, (a, b, m) in enumerate(rows):
    y = HY + RH * i
    s.text(8, y + 20, a, size=12.5, anchor="start")
    s.text(8, y + 37, b, cls="f-mu", size=11, anchor="start")
    lo, hi = run(m, "low"), run(m, "high")
    for j, (rid, _) in enumerate(rule_cols):
        x = LW + CW * j
        v = m.get(rmap[rid]["metric"])
        if v is None:
            s.rect(x + 3, y + 4, CW - 6, RH - 8, cls="s-mu f-sf", width=0.8, rx=4)
            s.text(x + CW / 2, y + RH / 2 + 4, "–", cls="f-mu", size=12)
            continue
        both, one = rid in lo and rid in hi, (rid in lo) != (rid in hi)
        if both:
            s.rect(x + 3, y + 4, CW - 6, RH - 8, cls="s-ac f-ac", width=1, rx=4)
            s.text(x + CW / 2, y + RH / 2 + 5, f"{v:.3f}", cls="f-bg", size=12.5, weight="bold")
        elif one:
            s.rect(x + 3, y + 4, CW - 6, RH - 8, cls="s-bd f-bds", width=1.8, rx=4)
            s.text(x + CW / 2, y + 20, f"{v:.3f}", cls="f-bd", size=12.5, weight="bold")
            s.text(x + CW / 2, y + 35, "높은 끝만" if rid in hi else "낮은 끝만", cls="f-bd", size=10.5)
        else:
            s.rect(x + 3, y + 4, CW - 6, RH - 8, cls="s-mu f-bg", width=0.8, rx=4)
            s.text(x + CW / 2, y + RH / 2 + 5, f"{v:.3f}", cls="f-mu", size=12)
ly = HY + RH * len(rows) + 22
lg = [("s-ac f-ac", "두 끝 모두 발동"), ("s-bd f-bds", "한쪽 끝에서만"), ("s-mu f-bg", "발동 안 함"), ("s-mu f-sf", "지표 없음")]
lx = LW
for cls, nm in lg:
    s.rect(lx, ly - 11, 22, 14, cls=cls, width=1.2, rx=3)
    s.text(lx + 28, ly, nm, size=11.5, anchor="start")
    lx += 132
s.text(8, ly + 26, "열 머리 = 등급, 규칙, 발동 조건의 잠정 범위(낮은 끝~높은 끝). 칸 숫자 = 지표 값", cls="f-mu", size=11, anchor="start")
s.save(os.path.join(OUT, "51-rules.svg"))
print("saved")
