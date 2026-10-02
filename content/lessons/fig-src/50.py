"""50강 그림과 본문 수치: 배포 경로 도식(SVG), 슬라이딩 윈도우와 이음매 정확도(SVG), 장면 8 예측·조각 제거·폴리곤(PNG)

- 본문 수치 대부분은 data/part7_runs.json의 export·sliding·polygon(계산: data/run_part7.py)에서 가져옴
- JSON에 없는 것(내보낸 그래프의 연산자와 초깃값 수, int8 모델의 입력 축척·영점과 예시 값, 보정 범위를 벗어난 입력 비율,
  처리량 비율, 슬라이딩 방식별 추론 횟수, 폴리곤 총수·10화소 미만 조각 수·조각 제거로 바뀐 화소,
  조각마다 따로 단순화했을 때 생기는 틈·겹침 면적)은 여기서 계산해 출력함
- 지연시간 재측정(--bench): 같은 장비에서 다시 재면 얼마나 흔들리는지 확인용. 결과는 실행할 때마다 다름
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "fig")
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
from svglib import Svg

R = json.load(open(os.path.join(DATA, "part7_runs.json"), encoding="utf-8"))
E, SL, PG = R["export"], R["sliding"], R["polygon"]
TMP = os.path.join(DATA, ".part7_tmp")          # run_part7.py export가 남긴 ONNX 파일

# ---------------------------------------------------------------- 1. 내보내기·양자화 확인
import onnx
from onnx import numpy_helper

for f in ("vdcnn.onnx", "vdcnn_int8.onnx"):
    m = onnx.load(os.path.join(TMP, f))
    ops = sorted(set(n.op_type for n in m.graph.node))
    tot = {}
    for t in m.graph.initializer:
        a = numpy_helper.to_array(t)
        tot.setdefault(str(a.dtype), [0, 0])
        tot[str(a.dtype)][0] += a.size; tot[str(a.dtype)][1] += a.nbytes
    print(f, "파일", os.path.getsize(os.path.join(TMP, f)), "B, 연산자", ops, "초깃값(개수, 바이트)", tot)
print("VD-CNN 파라미터 24,342 - BN 감마·베타 2×(16+32+64) =", 24342 - 2 * (16 + 32 + 64), "(BN이 합성곱에 합쳐짐)")

mq = onnx.load(os.path.join(TMP, "vdcnn_int8.onnx"))
I = {t.name: numpy_helper.to_array(t) for t in mq.graph.initializer}
for n in mq.graph.node:
    if n.op_type == "QuantizeLinear" and n.input[0] in ("x", "/features/features.2/Relu_output_0"):
        print("활성값 양자화", n.input[0], "축척", float(I[n.input[1]]), "영점", int(I[n.input[2]]))
lo, hi = -1.9101923, 7.8402505                   # 보정 자료(학습 타일 500장)의 입력 최솟값·최댓값(아래에서 다시 확인)
s = (hi - lo) / 255
z = round(-128 - lo / s)
q = round(1.0 / s) + z
print(f"입력 축척 s = {hi - lo:.3f}/255 = {s:.4f}, 영점 z = {z}, x = 1.00 → q = {q} → 되돌린 값 {(q - z) * s:.4f}, 최대 반올림 오차 s/2 = {s / 2:.4f}")

import part5_common as P5C
D = P5C.data()
xc = D["x_train"][:500].numpy()
print("보정 자료 범위", xc.min(), xc.max())
for k in ("x_test", "x_shift"):
    v = D[k].numpy()
    print(k, "보정 범위 밖 입력 비율", float(((v < lo) | (v > hi)).mean()), "최댓값", float(v.max()))
o = E["onnx_int8_static"]
print("연무 시험 맞힌 타일 int8·fp32", round(o["shift_acc"] * 900), round(o["shift_acc_fp32"] * 900))
L = E["latency_cpu_1thread"]
for k in ("pytorch_fp32", "onnxruntime_fp32", "onnxruntime_int8"):
    a, b = L["batch1"][k], L["batch64"][k]
    print(f"{k}: 배치1 {a['ms']} ms {a['tiles_per_s']}/s, 배치64 {b['ms']} ms {b['tiles_per_s']}/s, 처리량 {b['tiles_per_s'] / a['tiles_per_s']:.2f}배, 호출당 시간 {b['ms'] / a['ms']:.1f}배")
print("배치1 PyTorch/ORT fp32", round(L["batch1"]["pytorch_fp32"]["ms"] / L["batch1"]["onnxruntime_fp32"]["ms"], 1),
      "배치64 int8/fp32", round(L["batch64"]["onnxruntime_int8"]["ms"] / L["batch64"]["onnxruntime_fp32"]["ms"], 3))

if "--bench" in sys.argv:                          # 같은 장비에서 다시 재 보기(실행할 때마다 다름)
    import torch, onnxruntime as ort
    from part5_common import TinyCNN
    torch.set_num_threads(1)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s32 = ort.InferenceSession(os.path.join(TMP, "vdcnn.onnx"), so, providers=["CPUExecutionProvider"])
    sq = ort.InferenceSession(os.path.join(TMP, "vdcnn_int8.onnx"), so, providers=["CPUExecutionProvider"])
    m = TinyCNN(); m.load_state_dict(torch.load(os.path.join(DATA, ".part5_base.pt"))); m.eval()
    x = D["x_test"].numpy()

    def bench(fn, n):
        fn(); t = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
        return np.median(t) * 1000
    for rep in range(3):
        for b in (1, 64):
            xb = x[:b]; tb = torch.tensor(xb)
            with torch.no_grad():
                tp = bench(lambda: m(tb), 200 if b == 1 else 50)
            print("재측정", rep, "배치", b, "ms PyTorch·ORT fp32·int8", round(tp, 3),
                  round(bench(lambda: s32.run(None, {"x": xb}), 200 if b == 1 else 50), 3),
                  round(bench(lambda: sq.run(None, {"x": xb}), 200 if b == 1 else 50), 3))

# ---------------------------------------------------------------- 2. 슬라이딩 윈도우·폴리곤 확인
import run_part7 as RP
from make_part7 import scenes, GEOTRANSFORM
from part7_common import to_tensor
import rasterio.features as RF
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.ops import unary_union

n_pos = len(range(0, 256 - 32 + 1, 16))
print("추론 횟수: 이어 붙임", (256 // 32) ** 2, "겹쳐(간격 16)", n_pos ** 2, "통째로 1 / 이음매 화소 비율", SL["seam_frac"])
for kk in ("tiles", "overlap_avg", "overlap_center", "whole"):
    print(kk, SL[kk])

S = scenes()
K = 8
UN = RP.unet()
xs = to_tensor(S[K]["x"][None])[0]
pred = RP.infer_scene(UN, xs, "overlap_avg").argmax(0).numpy().astype(np.uint8)
pred_t = RP.infer_scene(UN, xs, "tiles").argmax(0).numpy().astype(np.uint8)
gt = S[K]["m"]
GT = GEOTRANSFORM(K)
T = Affine.from_gdal(*GT)
print("지오트랜스폼", GT, "열 100·행 50 중심", T * (100.5, 50.5), "모서리", T * (100, 50))
print("화소 정확도 겹쳐 평균", round(float((pred == gt).mean()), 4), "(JSON", PG["pixel_acc_raw"], ")")
sv = RF.sieve(pred, size=10, connectivity=8)
for name, a in (("정답", gt.astype(np.uint8)), ("예측", pred), ("조각 제거", sv)):
    gs = [shape(g) for g, _ in RF.shapes(a, transform=T, connectivity=8)]
    print(name, "폴리곤", len(gs), "10화소 미만", sum(g.area < 1000 for g in gs))
print("조각 제거로 바뀐 화소", int((sv != pred).sum()))
simp = [(shape(g).simplify(10.0, preserve_topology=True), int(v)) for g, v in RF.shapes(sv, transform=T, connectivity=8)]
tot = sum(g.area for g, _ in simp); uni = unary_union([g for g, _ in simp]).area
print(f"따로 단순화: 면적 합 {tot / 1e4:.2f} ha, 합집합 {uni / 1e4:.2f} ha, 겹침 {(tot - uni) / 1e4:.2f} ha, 틈 {(256 * 256 * 100 - uni) / 1e4:.2f} ha")
CL = ["산림", "농경지", "시가지", "수계", "나지", "갯벌"]
for c in range(6):
    a, b, d = PG["gt"][str(c)], PG["pred_sieve10"][str(c)], PG["pred_sieve10_simplify10m"][str(c)]
    print(f"{CL[c]}: 정답 {a['area_ha']} ha / 예측 {d['area_ha']} ha ({(d['area_ha'] / a['area_ha'] - 1) * 100:+.0f}%), "
          f"개수 {a['n']}/{PG['pred_raw'][str(c)]['n']}/{b['n']}, 꼭짓점 {b['vertices']}→{d['vertices']} ({d['vertices'] / b['vertices']:.2f}), "
          f"단순화 면적 변화 {d['area_ha'] - b['area_ha']:+.2f} ha")

# ---------------------------------------------------------------- 3. 배포 경로 도식(SVG)
s = Svg(800, 250, "PyTorch 모델을 ONNX로 내보내고, 추론 엔진(onnxruntime 또는 TensorRT)으로 장면을 슬라이딩 윈도우로 추론해 "
                  "확률 래스터를 만든 뒤, 지오트랜스폼을 붙여 조각 제거·단순화를 거쳐 폴리곤으로 바꿈. 산출물마다 메타데이터를 붙임")
BOX = [("PyTorch", "학습 코드·가중치"), ("ONNX", "계산 그래프"), ("추론 엔진", "onnxruntime·TensorRT"),
       ("래스터 결과", "클래스·확률 격자"), ("폴리곤", "벡터 다각형")]
BW, BH, GAP, Y0 = 140, 62, 24, 40
X0 = (800 - (5 * BW + 4 * GAP)) / 2
for i, (t1, t2) in enumerate(BOX):
    x = X0 + i * (BW + GAP)
    s.rect(x, Y0, BW, BH, "s-fg f-acs" if i in (1, 2) else "s-fg f-sf", 1.5, 6)
    s.text(x + BW / 2, Y0 + 26, t1, size=14, weight="bold")
    s.text(x + BW / 2, Y0 + 46, t2, "f-mu", size=11)
    if i < 4:
        xa = x + BW + 3
        s.line(xa, Y0 + BH / 2, xa + GAP - 8, Y0 + BH / 2, "s-fg", 1.5)
        s.path(f"M{xa + GAP - 6:.1f},{Y0 + BH / 2:.1f} l-7,-4 l0,8 z", "s-fg f-fg", 1)
STEP = ["내보내기", "양자화(선택)", "슬라이딩 윈도우", "지오트랜스폼"]
STEP2 = ["출력 차이 확인", "정확도 확인", "이어 붙이기", "조각 제거·단순화"]
for i in range(4):
    xc = X0 + i * (BW + GAP) + BW + GAP / 2
    s.text(xc, Y0 + BH + 28, STEP[i], "f-ac", size=12)
    s.text(xc, Y0 + BH + 45, STEP2[i], "f-mu", size=11)
s.rect(X0, 196, 5 * BW + 4 * GAP, 34, "s-mu f-bds", 1, 6)
s.text(400, 218, "메타데이터: 좌표계 · 지오트랜스폼 · 모델 이름·버전 · 전처리 · 클래스 코드 · 후처리 값", "f-fg", size=12)
s.save(os.path.join(OUT, "50-pipeline.svg"))

# ---------------------------------------------------------------- 4. 슬라이딩 윈도우와 이음매 정확도(SVG)
s = Svg(780, 300, "왼쪽: 32화소 타일을 겹침 없이 이어 붙이면 타일 경계(이음매)에서 예측이 끊김. 간격 16으로 겹쳐 놓은 창은 이음매를 가운데에 둠. "
                  "오른쪽: 방식별 이음매 근처와 나머지 화소의 정확도")
# 왼쪽 도식: 64×64 화소 구역(2×2 타일), 1화소 = 3.6 단위
PX, LX, LY = 3.6, 40, 40
side = 64 * PX
for r in range(2):
    for c in range(2):
        s.rect(LX + c * 32 * PX, LY + r * 32 * PX, 32 * PX, 32 * PX, "s-fg f-sf", 1.5)
s.rect(LX + 30 * PX, LY, 4 * PX, side, "s-bd f-bds", 0.8)       # 이음매 띠 ±2화소
s.rect(LX, LY + 30 * PX, side, 4 * PX, "s-bd f-bds", 0.8)
s.rect(LX + 24 * PX, LY + 24 * PX, 16 * PX, 16 * PX, "s-ac f-acs", 0.8)   # 겹친 창의 가운데 16×16
s.add(f'<rect x="{LX + 16 * PX:.1f}" y="{LY + 16 * PX:.1f}" width="{32 * PX:.1f}" height="{32 * PX:.1f}" class="s-ac" fill="none" stroke-width="2.2" stroke-dasharray="6 4"/>')
s.text(LX + 16 * PX, LY - 10, "타일 32", "f-fg", size=12)
s.text(LX + side + 8, LY + 16 * PX + 4, "겹친 창", "f-ac", size=12, anchor="start")
s.text(LX + side + 8, LY + 16 * PX + 20, "(간격 16)", "f-ac", size=11, anchor="start")
s.text(LX + side + 8, LY + 32 * PX + 4, "이음매", "f-bd", size=12, anchor="start")
s.text(LX + side + 8, LY + 32 * PX + 20, "±2화소", "f-bd", size=11, anchor="start")
s.text(LX + side / 2, LY + side + 24, "창 가운데가 이음매를 덮음", "f-mu", size=11)
# 오른쪽 점 그림: y = 정확도 0.70~0.82
RX0, RX1, RY0, RY1 = 410, 760, 40, 240
vmin, vmax = 0.70, 0.82
yv = lambda v: RY1 - (v - vmin) / (vmax - vmin) * (RY1 - RY0)
for v in (0.70, 0.74, 0.78, 0.82):
    s.line(RX0, yv(v), RX1, yv(v), "s-mu", 0.6, "2 3")
    s.text(RX0 - 6, yv(v) + 4, f"{v:.2f}", "f-mu", size=11, anchor="end")
MODES = [("tiles", "이어 붙임"), ("overlap_avg", "겹쳐 평균"), ("overlap_center", "가운데 우선"), ("whole", "통째로")]
cw = (RX1 - RX0) / 4
for i, (k, lab) in enumerate(MODES):
    xc = RX0 + cw * (i + 0.5)
    a, b = SL[k]["seam_acc"], SL[k]["non_seam_acc"]
    s.line(xc, yv(a), xc, yv(b), "s-mu", 1.5)
    s.circle(xc, yv(b), 5.5, "f-ac")
    s.circle(xc, yv(a), 5.5, "f-bd")
    s.text(xc, RY1 + 22, lab, "f-fg", size=12)
s.text(RX0 + cw * 0.5 + 10, yv(SL["tiles"]["seam_acc"]) + 4, f'{SL["tiles"]["seam_acc"]:.3f}', "f-bd", size=11, anchor="start")
s.text(RX0 + cw * 0.5 + 10, yv(SL["tiles"]["non_seam_acc"]) + 4, f'{SL["tiles"]["non_seam_acc"]:.3f}', "f-ac", size=11, anchor="start")
s.circle(RX0 + 8, 22, 5, "f-bd"); s.text(RX0 + 18, 26, "이음매 ±2화소", "f-fg", size=11, anchor="start")
s.circle(RX0 + 128, 22, 5, "f-ac"); s.text(RX0 + 138, 26, "나머지", "f-fg", size=11, anchor="start")
s.text(RX1, 26, "화소 정확도", "f-mu", size=11, anchor="end")
s.save(os.path.join(OUT, "50-sliding.svg"))

# ---------------------------------------------------------------- 5. 장면 8: 영상·정답·예측·조각 제거·폴리곤(PNG, 글자 없음)
import cv2
from PIL import Image

PAL = np.array([[27, 110, 50], [176, 214, 92], [214, 60, 60], [40, 95, 210], [200, 160, 100], [150, 130, 175]], np.uint8)
fc = S[K]["x"][[3, 2, 1]]                        # 폴스컬러(근적외·적·녹)
plo, phi = np.percentile(fc, 1, axis=(1, 2)), np.percentile(fc, 99, axis=(1, 2))
rgb = (np.clip((fc - plo[:, None, None]) / (phi - plo)[:, None, None], 0, 1) ** 0.8 * 255).round().astype(np.uint8).transpose(1, 2, 0)
Z = 2                                            # 2배 확대(외곽선을 또렷하게)
up = lambda a: a.repeat(Z, 0).repeat(Z, 1)
inv = ~T
ol = up(rgb).copy()
for g, v in simp:
    polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
    for p in polys:
        for ring in [p.exterior] + list(p.interiors):
            pts = np.array([inv * xy for xy in ring.coords]) * Z
            cv2.polylines(ol, [np.round(pts).astype(np.int32)], True, (255, 255, 0), 1, cv2.LINE_AA)
panels = [[up(rgb), up(PAL[gt])], [up(PAL[pred_t]), up(PAL[pred])], [up(PAL[sv]), ol]]   # 3행 2열
N, G = 256 * Z, 10
canvas = np.zeros((3 * N + 2 * G, 2 * N + G, 4), np.uint8)
for r, row in enumerate(panels):
    for c, pnl in enumerate(row):
        canvas[r * (N + G):r * (N + G) + N, c * (N + G):c * (N + G) + N, :3] = pnl
        canvas[r * (N + G):r * (N + G) + N, c * (N + G):c * (N + G) + N, 3] = 255
im = Image.fromarray(canvas, "RGBA").quantize(colors=128, method=Image.Quantize.FASTOCTREE)
p = os.path.join(OUT, "50-scene.png")
im.save(p, optimize=True)
print("PNG", im.size, os.path.getsize(p) // 1024, "KB")
