"""43강 그림과 확인 계산: 7부 장면 예시(PNG), 나누는 방법 다섯 가지 도식(SVG), 방법별 검증 vs 시험 정확도(SVG)

수치 출처
- 학습 결과(검증·시험 정확도, 장면 단위 교차검증, 타일링 표): data/part7_runs.json의 leak·leakcv·tiling (data/run_part7.py)
- 여기서 새로 계산하는 것(학습 없음, 몇 초): 분할별로 검증 타일 화소가 학습 타일에 들어 있는 비율,
  6부 큰 장면 객체의 HBB 긴 변 분포, 타일 시작 위치, 처리 화소 배수 어림
"""
import json, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from svglib import Svg
from make_part7 import scenes, tile_scene
from make_part6 import big_scene

OUT = os.path.join(HERE, "..", "fig")
R = json.load(open(os.path.join(HERE, "..", "data", "part7_runs.json"), encoding="utf-8"))
S = scenes()

# ---------------------------------------------------------------- 확인 계산 1: 분할별 화소 공유
print("장면당 타일 수: 간격 16", len(tile_scene(S[0], 32, 16)["y"]), "/ 간격 32", len(tile_scene(S[0], 32, 32)["y"]))
rng = np.random.default_rng(0)          # run_part7.exp_leak과 같은 순서로 순열을 뽑음 (A 다음 B)


def flat(ts):
    return (np.concatenate([np.full(len(t["y"]), k) for k, t in enumerate(ts)]),
            np.concatenate([t["r0"] for t in ts]), np.concatenate([t["c0"] for t in ts]))


def shared(sc, r0, c0, tr, va):
    """검증 타일마다 화소 가운데 학습 타일에도 들어 있는 비율"""
    cov = np.zeros((8, 256, 256), bool)
    for i in tr:
        cov[sc[i], r0[i]:r0[i] + 32, c0[i]:c0[i] + 32] = True
    return np.array([cov[sc[i], r0[i]:r0[i] + 32, c0[i]:c0[i] + 32].mean() for i in va])


sc, r0, c0 = flat([tile_scene(S[k], 32, 16) for k in range(8)])
p = rng.permutation(len(sc)); cut = int(0.8 * len(sc))
f = shared(sc, r0, c0, p[:cut], p[cut:])
print(f"A 검증 {len(f)}장: 학습 타일과 겹치는 타일 {(f > 0).mean():.3f}, 화소 공유 {f.mean():.3f}, 화소 전부 공유 {(f == 1).mean():.3f}")
sc, r0, c0 = flat([tile_scene(S[k], 32, 32) for k in range(8)])
p = rng.permutation(len(sc)); cut = int(0.8 * len(sc))
trs = set(p[:cut].tolist()); key = {(sc[i], r0[i], c0[i]): i for i in range(len(sc))}
nb = [any(key.get((sc[i], r0[i] + a, c0[i] + b)) in trs for a, b in ((32, 0), (-32, 0), (0, 32), (0, -32))) for i in p[cut:]]
print(f"B 검증 {len(nb)}장: 상하좌우 이웃에 학습 타일이 있는 비율 {np.mean(nb):.3f}")
sc, r0, c0 = flat([tile_scene(S[k], 32, 16) for k in range(8)])
tr = np.where(c0 + 16 < 192)[0]; va = np.where(c0 + 16 >= 192)[0]
f = shared(sc, r0, c0, tr, va)
print(f"C 완충 없음 검증 {len(va)}장: 학습과 화소를 나누는 타일 {(f > 0).mean():.3f}, 화소 공유 {f.mean():.3f}")
tr = np.where(c0 + 32 <= 176)[0]; va = np.where(c0 >= 208)[0]
f = shared(sc, r0, c0, tr, va)
print(f"C 완충 검증 {len(va)}장: 화소 공유 {f.mean():.3f}, 학습 {len(tr)}장, "
      f"학습 화소 마지막 열 {int((c0[tr] + 31).max())}, 검증 화소 첫 열 {int(c0[va].min())}")
# 띠 너비와 비교할 자기상관 거리: 장면 0~7에서 가로로 d화소 떨어진 두 화소의 B8 상관(장면 평균)
for d in (16, 32, 48, 64):
    print(f"B8 가로 상관 d={d}: {np.mean([np.corrcoef(S[k]['x'][3][:, :-d].ravel(), S[k]['x'][3][:, d:].ravel())[0, 1] for k in range(8)]):.2f}")

L = R["leak"]; CV = R["leakcv"]
names = [("A_random_overlap", "A"), ("B_random_nooverlap", "B"), ("C_block_nobuffer", "C"), ("C_block_buffer", "C+"), ("D_scene", "D")]
base = np.mean([f["val_acc_best"] for f in CV["folds"]]) - np.mean([f["test_acc_best"] for f in CV["folds"]])
print(f"장면 단위 CV: 검증 {[f['val_acc_best'] for f in CV['folds']]} 평균 {CV['val_mean']} sd {CV['val_sd']}, "
      f"시험 {[f['test_acc_best'] for f in CV['folds']]} 평균 {CV['test_mean']}, 차이 {base:.4f}")
for k, lab in names:
    v, t = L[k]["val_acc_best"], L[k]["test_acc_best"]
    print(f"{lab}: 학습 {L[k]['n_train']} 검증 {L[k]['n_val']} best {L[k]['best_epoch']} 검증 {v} 시험 {t} 차이 {v - t:.4f} 기준선 초과 {v - t - base:+.4f}")
print("시험 타일 하나 =", round(1 / L["n_test"], 4))

# ---------------------------------------------------------------- 확인 계산 2: 6부 큰 장면 타일링
B = big_scene()
hb = np.array([o["hbb"] for o in B["objs"]]); side = np.maximum(hb[:, 2] - hb[:, 0], hb[:, 3] - hb[:, 1])
print("객체", len(side), "HBB 긴 변 >64:", int((side > 64).sum()), ">128:", int((side > 128).sum()), ">512:", int((side > 512).sum()),
      "가장 긴 객체 길이", R["tiling"]["longest_obj_px"])
for size in (256, 512, 1024):
    for ov in (0, 64, 128):
        st = list(range(0, 2048 - size + 1, size - ov))
        if st[-1] != 2048 - size:
            st.append(2048 - size)
        full = np.zeros(len(side), bool)
        for y0 in st:
            for x0 in st:
                full |= (hb[:, 0] >= x0) & (hb[:, 1] >= y0) & (hb[:, 2] <= x0 + size) & (hb[:, 3] <= y0 + size)
        T = R["tiling"][f"{size}_ov{ov}"]
        print(size, ov, "시작", st, "타일", T["n_tiles"], "화소배수", T["pixels_rel"], "어림", round((size / (size - ov)) ** 2, 2),
              "온전히 안 들어감", T["objs_never_whole"], "(그중 중첩보다 짧은 것", int(((~full) & (side <= ov)).sum()) if ov else "-", ")",
              "어딘가 잘림", T["objs_cut_somewhere"], "배경", T["background_tiles"], T["background_frac"])

# ---------------------------------------------------------------- 1. 장면 예시 (PNG)
# 위 줄: 폴스컬러(B8, B4, B3 → R, G, B), 아래 줄: 화소 라벨. 왼쪽부터 장면 0, 3, 6(개발용), 8(시험용)
PAL = np.array([[34, 110, 50], [200, 190, 90], [190, 70, 70], [40, 90, 170], [215, 170, 120], [120, 105, 95]], np.uint8)
show = [0, 3, 6, 8]
fc = np.stack([S[k]["x"][[3, 2, 1]] for k in show])
lo = np.percentile(fc, 1, axis=(0, 2, 3)); hi = np.percentile(fc, 99, axis=(0, 2, 3))   # 네 장면 공통 늘이기(촬영 조건 차이가 보이게)
N, G = 256, 10
canvas = np.zeros((2 * N + G, len(show) * N + (len(show) - 1) * G, 4), np.uint8)
for j, k in enumerate(show):
    v = np.clip((fc[j] - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1) ** 0.8
    c = j * (N + G)
    canvas[:N, c:c + N, :3] = np.round(v * 255).astype(np.uint8).transpose(1, 2, 0)
    canvas[N + G:, c:c + N, :3] = PAL[S[k]["m"]]
    canvas[:N, c:c + N, 3] = 255; canvas[N + G:, c:c + N, 3] = 255
Image.fromarray(canvas, "RGBA").quantize(colors=256, method=Image.Quantize.FASTOCTREE).save(os.path.join(OUT, "43-scenes.png"), optimize=True)
print("43-scenes.png", os.path.getsize(os.path.join(OUT, "43-scenes.png")) // 1024, "KB", canvas.shape)

# ---------------------------------------------------------------- 2. 나누는 방법 도식 (SVG)
def orect(s, x, y, w, h, cls, width):
    """채움 없는 테두리 사각형"""
    s.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" class="{cls}" stroke-width="{width}" fill="none"/>')


s = Svg(700, 214, "타일 데이터셋을 나누는 방법 다섯 가지에서 학습·검증 타일이 장면 위에 놓이는 모습")
W, TOP, X0, GAP = 120, 46, 18, 16
sc_ = W / 256                                     # 장면 화소 → 그림 좌표
titles = ["A 겹침·무작위", "B 무작위", "C 구역", "C+ 완충", "D 장면 단위"]
pr = np.random.default_rng(3)
for k, title in enumerate(titles):
    ox = X0 + k * (W + GAP)
    s.text(ox + W / 2, TOP - 14, title, "f-fg", 13, weight="600")
    if k == 0:      # 타일 중심 점(간격 16 → 그림에서는 간격을 줄여 7×7로), 그 위에 겹친 타일 한 묶음
        s.rect(ox, TOP, W, W, "s-mu f-sf", 1)
        for i in range(7):
            for jj in range(7):
                s.circle(ox + 15 + 15 * i, TOP + 15 + 15 * jj, 3.2, "f-bd" if pr.random() < 0.2 else "f-ac")
        vx, vy = ox + 45, TOP + 45                # 검증 타일 하나(가운데 (60,60))와 겹치는 학습 타일들
        for dx, dy in ((-15, -15), (15, 0), (0, 15)):
            orect(s, vx + dx, vy + dy, 30, 30, "s-ac", 1.6)
        orect(s, vx, vy, 30, 30, "s-bd", 2.4)
        s.circle(vx + 15, vy + 15, 3.2, "f-bd")
    elif k == 1:    # 안 겹친 타일 4×4(그림용으로 줄임)를 무작위 배정
        for i in range(4):
            for jj in range(4):
                s.rect(ox + 30 * i + 1, TOP + 30 * jj + 1, 28, 28, ("f-bd" if pr.random() < 0.25 else "f-ac"), 0)
        orect(s, ox, TOP, W, W, "s-mu", 1)
    elif k == 2:    # 타일 중심 x 192 기준. 경계 타일(176~208)은 검증, 그 왼쪽 학습 타일(160~192)과 화소를 나눔
        s.rect(ox, TOP, 192 * sc_, W, "s-mu f-acs", 1)
        s.rect(ox + 192 * sc_, TOP, (256 - 192) * sc_, W, "s-mu f-bds", 1)
        y = TOP + 44
        orect(s, ox + 160 * sc_, y, 32 * sc_, 32 * sc_, "s-ac", 2)
        orect(s, ox + 176 * sc_, y + 4, 32 * sc_, 32 * sc_, "s-bd", 2.4)
    elif k == 3:    # 176~208 띠를 비움
        s.rect(ox, TOP, 176 * sc_, W, "s-mu f-acs", 1)
        s.rect(ox + 176 * sc_, TOP, 32 * sc_, W, "s-mu f-sf", 1)
        s.rect(ox + 208 * sc_, TOP, 48 * sc_, W, "s-mu f-bds", 1)
        for yy in np.arange(TOP + 6, TOP + W, 12):
            s.line(ox + 176 * sc_ + 3, yy + 6, ox + 208 * sc_ - 3, yy, "s-mu", 1)
    else:           # 장면 10장: 0~5 학습, 6~7 검증, 8~9 시험
        cw, cg = 26, 5
        for n in range(10):
            col, row = n % 4, n // 4
            x = ox + col * (cw + cg) + (cw + cg) * (1 if n >= 8 else 0)
            yy = TOP + row * (cw + cg + (6 if n >= 8 else 0))
            cls = "f-ac" if n < 6 else ("f-bd" if n < 8 else "f-ok")
            s.rect(x, yy, cw, cw, cls, 0)
            s.text(x + cw / 2, yy + cw / 2 + 4, str(n), "f-bg", 11)
# 범례
ly = TOP + W + 34
items = [("f-ac", "학습"), ("f-bd", "검증"), ("f-sf", "비운 띠"), ("f-ok", "시험(장면 8·9)")]
lx = 150
for cls, lab in items:
    s.rect(lx, ly - 11, 14, 14, "s-mu " + cls, 1)
    s.text(lx + 22, ly + 1, lab, "f-mu", 12, anchor="start")
    lx += 40 + 13 * len(lab)
s.save(os.path.join(OUT, "43-splits.svg"))

# ---------------------------------------------------------------- 3. 방법별 검증 vs 시험 정확도 (SVG)
rows = [(lab, L[k]["val_acc_best"], L[k]["test_acc_best"], None) for k, lab in names]
fv = [f["val_acc_best"] for f in CV["folds"]]; ft = [f["test_acc_best"] for f in CV["folds"]]
rows.append(("CV", float(np.mean(fv)), float(np.mean(ft)), (min(fv), max(fv), min(ft), max(ft))))   # 반올림 전 겹별 값의 평균
s = Svg(640, 300, "나누는 방법별 검증 정확도와 시험 정확도. 겹친 타일 무작위 분할의 차이가 가장 큼")
PL, PR, PT, RH = 120, 600, 30, 38
lo_, hi_ = 0.78, 0.96
X = lambda a: PL + (a - lo_) / (hi_ - lo_) * (PR - PL)
for t in np.arange(0.78, 0.961, 0.02):
    s.line(X(t), PT - 6, X(t), PT + RH * len(rows) - 10, "s-mu", 0.6, dash="2 3")
    s.text(X(t), PT + RH * len(rows) + 6, f"{t:.2f}", "f-mu", 11)
s.text((PL + PR) / 2, PT + RH * len(rows) + 28, "정확도", "f-mu", 12)
labels = {"A": "A 겹침·무작위", "B": "B 무작위", "C": "C 구역", "C+": "C+ 완충", "D": "D 장면 단위", "CV": "장면 4겹 평균"}
for i, (lab, v, t, rg) in enumerate(rows):
    y = PT + RH * i + 8
    s.text(PL - 12, y + 4, labels[lab], "f-fg", 12, anchor="end", weight="600" if lab in ("A", "CV") else None)
    if rg:
        s.line(X(rg[0]), y - 11, X(rg[1]), y - 11, "s-bd", 2)
        s.line(X(rg[2]), y + 11, X(rg[3]), y + 11, "s-ac", 2)
    s.line(X(t), y, X(v), y, "s-mu", 2)
    s.circle(X(v), y, 5.5, "f-bd")
    s.circle(X(t), y, 5.5, "f-ac")
    s.text(X(v) + 10, y + 4, f"+{v - t:.3f}", "f-mu", 11, anchor="start")
ly = 14
s.circle(PL + 4, ly - 4, 5.5, "f-bd"); s.text(PL + 14, ly, "검증(best)", "f-mu", 12, anchor="start")
s.circle(PL + 110, ly - 4, 5.5, "f-ac"); s.text(PL + 120, ly, "시험(장면 8·9)", "f-mu", 12, anchor="start")
s.line(PL + 240, ly - 7, PL + 262, ly - 7, "s-bd", 2); s.line(PL + 240, ly - 1, PL + 262, ly - 1, "s-ac", 2)
s.text(PL + 268, ly, "4겹의 범위", "f-mu", 12, anchor="start")
s.save(os.path.join(OUT, "43-leak.svg"))
print("done")
