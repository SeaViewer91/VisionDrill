"""7부 본문 수치를 만드는 실험 모음. 결과는 part7_runs.json. CPU 1스레드로 전체 40분 안팎.

    python3 run_part7.py              # 전체
    python3 run_part7.py aug seeds    # 일부만 (다른 결과는 유지)

자료: 5부 타일 자료(make_part5), 7부 장면 10장(make_part7), 6부 항만 자료(make_part6). 모델: 5부 VD-CNN·U-Net.
학습 기본: Adam 0.001, 배치 64, 시드 0. fit()이 끝나면 모델에는 검증 손실이 가장 낮았던 에폭(best)의 가중치가 들어 있음.

실험 이름과 쓰는 강
- leak     : 타일 나누는 방법 4가지(겹친 타일 무작위, 안 겹친 타일 무작위, 장면 안 구역 분할(완충 유무), 장면 단위)의 검증 vs 시험 (43강)
- leakcv   : 장면 단위 4겹 교차검증(검증 장면 둘씩 돌아가며) (43강)
- tiling   : 6부 큰 장면(2,048)에서 타일 크기·중첩별 타일 수, 잘리는 객체, 배경 타일 비율 (43강)
- formats  : 같은 객체의 YOLO·COCO·VOC·YOLO-OBB·GeoJSON 좌표 (44강)
- noise    : 학습 라벨 잡음(대칭 10·20·40%, 갯벌→수계 40%)과 시험 정확도 (44강)
- missing  : 정답 라벨이 일부 빠진 채 평가하면 같은 탐지의 AP가 어떻게 보이나 (44강)
- aug      : 학습 1,000장에서 증강(없음·기하·광학·기하+광학·밴드 섞기·믹스업)과 시험·회전 시험·연무 시험 정확도 (45강)
- lrfind   : 학습률 범위 테스트 (46강)
- sched    : SGD 모멘텀에서 학습률 스케줄(고정·계단·코사인·원사이클·워밍업+코사인) (46강)
- warmup   : 정규화 없는 VD-CNN, 큰 학습률에서 워밍업 유무 (46강)
- bscale   : 배치 32·256과 선형 스케일링 규칙 (46강)
- accum    : 기울기 누적이 큰 배치와 같은가(배치 정규화 vs 그룹 정규화) (47강)
- bf16     : CPU bf16 자동 혼합 정밀도 학습 vs fp32, 수 표현 범위 (47강)
- memory   : ResNet-50·VD-CNN의 학습 메모리 어림(파라미터·기울기·Adam 상태·활성값) (47강)
- loader   : 데이터로더 작업자 수에 따른 에폭 시간 (47강)
- tl       : 연무·계절 도메인 소량 라벨(60·300장)로 처음부터 vs 선형 탐침 vs 일부 동결 vs 전체 미세조정, 배치 정규화 통계 갱신(AdaBN) (48강)
- rgb4     : RGB 3밴드로 사전학습한 가중치를 4밴드 입력에 옮기는 방법 (48강)
- seeds    : 같은 설정을 시드 5개로, 배치 정규화 제거 절제 실험 (49강)
- export   : ONNX 내보내기, 출력 차이, 지연시간·처리량, int8 양자화 (50강)
- sliding  : 장면 전체 추론: 타일 이어 붙이기 vs 겹쳐 평균 vs 통째로, 이음매 정확도 (50강)
- polygon  : 지오트랜스폼과 폴리곤화, 작은 조각 제거·단순화, 면적 (50강)
"""
import json, os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import part5_common as P5C
import make_part5 as P5
from part5_common import TinyCNN, TinyUNet, evaluate, n_params, miou
from part7_common import fit, to_tensor, norm_stats, aug_geo
from make_part7 import scenes, tile_scene, GEOTRANSFORM

OUT = os.environ.get("PART7_OUT") or os.path.join(HERE, "part7_runs.json")
R = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
D = P5C.data()
EV = {"test": (D["x_test"], D["y_test"]), "shift": (D["x_shift"], D["y_shift"])}


def r4(x):
    return None if x is None else round(float(x), 4)


def save():
    json.dump(R, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def slim(res):
    """결과에서 큰 기록은 줄여 저장"""
    h = res["hist"]
    out = {k: v for k, v in res.items() if k != "hist"}
    out["hist"] = {k: h[k] for k in ("train_loss", "val_loss", "val_acc", "lr") if k in h}
    out["sec_total"] = round(sum(h["sec"]), 1)
    return out


# ------------------------------------------------------------------ 43 데이터셋 나누기
def exp_leak():
    S = scenes()
    def stack(parts):
        x = np.concatenate([p["x"] for p in parts]); y = np.concatenate([p["y"] for p in parts])
        return to_tensor(x), torch.tensor(y)
    test = stack([tile_scene(S[k], 32, 32) for k in (8, 9)])
    rng = np.random.default_rng(0)
    out = {}
    def run(name, tr, va, note):
        torch.manual_seed(0)
        m = TinyCNN()
        res = fit(m, tr[0], tr[1], va[0], va[1], {"test": test}, epochs=30)
        out[name] = dict(note=note, n_train=len(tr[1]), n_val=len(va[1]), best_epoch=res["best_epoch"],
                         val_acc_best=res["best"]["val"], test_acc_best=res["best"]["test"],
                         val_acc_last=res["last"]["val"], test_acc_last=res["last"]["test"],
                         hist_val_acc=res["hist"]["val_acc"])
    # A. 겹친 타일(간격 16) 무작위 8:2
    allt = [tile_scene(S[k], 32, 16) for k in range(8)]
    x = np.concatenate([t["x"] for t in allt]); y = np.concatenate([t["y"] for t in allt])
    p = rng.permutation(len(y)); cut = int(0.8 * len(y))
    run("A_random_overlap", (to_tensor(x[p[:cut]]), torch.tensor(y[p[:cut]])), (to_tensor(x[p[cut:]]), torch.tensor(y[p[cut:]])),
        "장면 0~7을 간격 16(50% 겹침)으로 자른 타일을 무작위 8:2")
    # B. 안 겹친 타일(간격 32) 무작위 8:2
    allt = [tile_scene(S[k], 32, 32) for k in range(8)]
    x = np.concatenate([t["x"] for t in allt]); y = np.concatenate([t["y"] for t in allt])
    p = rng.permutation(len(y)); cut = int(0.8 * len(y))
    run("B_random_nooverlap", (to_tensor(x[p[:cut]]), torch.tensor(y[p[:cut]])), (to_tensor(x[p[cut:]]), torch.tensor(y[p[cut:]])),
        "간격 32(겹침 없음) 타일을 무작위 8:2")
    # C. 장면마다 오른쪽 구역을 검증으로(겹친 타일). 완충 없음: 타일 중심이 x=192 기준 / 완충: 경계 띠 [176, 208)에 걸친 타일 제외
    def region(buffer):
        tr, va = [], []
        for k in range(8):
            t = tile_scene(S[k], 32, 16)
            cx = t["c0"] + 16
            if buffer:
                trm = t["c0"] + 32 <= 176; vam = t["c0"] >= 208
            else:
                trm = cx < 192; vam = cx >= 192
            tr.append({k2: v[trm] for k2, v in t.items()}); va.append({k2: v[vam] for k2, v in t.items()})
        return stack(tr), stack(va)
    tr, va = region(False)
    run("C_block_nobuffer", tr, va, "장면마다 x 192 오른쪽을 검증(겹친 타일, 경계에 걸친 타일이 양쪽에 섞임)")
    tr, va = region(True)
    run("C_block_buffer", tr, va, "장면마다 x 176~208 띠(32화소)를 비우고 왼쪽 학습·오른쪽 검증")
    # D. 장면 단위: 0~5 학습(겹친 타일), 6~7 검증(안 겹친 타일)
    tr = stack([tile_scene(S[k], 32, 16) for k in range(6)]); va = stack([tile_scene(S[k], 32, 32) for k in (6, 7)])
    run("D_scene", tr, va, "장면 0~5 학습, 장면 6~7 검증")
    out["test_note"] = "시험은 모두 장면 8~9(안 겹친 타일 128장)"
    out["n_test"] = len(test[1])
    R["leak"] = out


def exp_leakcv():
    """장면 단위 4겹 교차검증: 장면 0~7을 둘씩 묶어 차례로 검증에 씀(나머지 6장면 학습)"""
    S = scenes()
    def stack(parts):
        x = np.concatenate([p["x"] for p in parts]); y = np.concatenate([p["y"] for p in parts])
        return to_tensor(x), torch.tensor(y)
    test = stack([tile_scene(S[k], 32, 32) for k in (8, 9)])
    folds = []
    for va_k in ((0, 1), (2, 3), (4, 5), (6, 7)):
        tr = stack([tile_scene(S[k], 32, 16) for k in range(8) if k not in va_k]); va = stack([tile_scene(S[k], 32, 32) for k in va_k])
        torch.manual_seed(0)
        res = fit(TinyCNN(), tr[0], tr[1], va[0], va[1], {"test": test}, epochs=30)
        folds.append(dict(val_scenes=list(va_k), val_acc_best=res["best"]["val"], test_acc_best=res["best"]["test"], best_epoch=res["best_epoch"]))
    v = np.array([f["val_acc_best"] for f in folds]); t = np.array([f["test_acc_best"] for f in folds])
    R["leakcv"] = dict(folds=folds, val_mean=r4(v.mean()), val_sd=r4(v.std(ddof=1)), test_mean=r4(t.mean()), test_sd=r4(t.std(ddof=1)),
                       note="장면 단위 4겹. 겹마다 학습 6장면(간격 16 타일), 검증 2장면(간격 32), 시험은 늘 장면 8~9")


def exp_tiling():
    from make_part6 import big_scene, obb_to_hbb
    B = big_scene()
    S, objs = B["size"], B["objs"]
    hb = np.array([o["hbb"] for o in objs])
    out = {}
    for size in (256, 512, 1024):
        for ov in (0, 64, 128):
            if ov >= size:
                continue
            step = size - ov
            st = list(range(0, S - size + 1, step))
            if st[-1] != S - size:
                st.append(S - size)
            n_t = len(st) ** 2
            full = np.zeros(len(objs), bool); cut_any = np.zeros(len(objs), bool)
            empty = 0
            for y0 in st:
                for x0 in st:
                    inside = (hb[:, 0] >= x0) & (hb[:, 1] >= y0) & (hb[:, 2] <= x0 + size) & (hb[:, 3] <= y0 + size)
                    touch = (hb[:, 2] > x0) & (hb[:, 0] < x0 + size) & (hb[:, 3] > y0) & (hb[:, 1] < y0 + size)
                    full |= inside; cut_any |= touch & ~inside
                    empty += int(not touch.any())
            out[f"{size}_ov{ov}"] = dict(n_tiles=n_t, pixels_rel=round(n_t * size * size / S ** 2, 3),
                                         objs_never_whole=int((~full).sum()), objs_cut_somewhere=int(cut_any.sum()),
                                         background_tiles=empty, background_frac=r4(empty / n_t))
    out["n_objs"] = len(objs)
    out["longest_obj_px"] = round(float(max(o["w"] for o in objs)), 1)
    R["tiling"] = out


# ------------------------------------------------------------------ 44 라벨
def exp_formats():
    from make_part6 import tiles, obb_corners
    T = tiles()
    t = T[8]                                            # 외해 타일 9번
    o = max([o for o in t["objs"] if o["cls"] == 1], key=lambda o: o["w"])
    W = H = 512
    x1, y1, x2, y2 = o["hbb"]
    gt0 = (420000.0, 0.5, 0.0, 3880000.0, 0.0, -0.5)     # 가상 지오트랜스폼(0.5 m, EPSG:32652)
    corners = obb_corners(o["cx"], o["cy"], o["w"], o["h"], o["theta"])
    geo = [[gt0[0] + gt0[1] * c[0], gt0[3] + gt0[5] * c[1]] for c in corners]
    out = dict(tile=t["image_id"], obj_obb=[round(v, 2) for v in [o["cx"], o["cy"], o["w"], o["h"], o["theta"]]],
               hbb_x1y1x2y2=[round(v, 2) for v in o["hbb"]],
               yolo=[1, round((x1 + x2) / 2 / W, 6), round((y1 + y2) / 2 / H, 6), round((x2 - x1) / W, 6), round((y2 - y1) / H, 6)],
               coco_bbox=[round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
               voc_xmin_ymin_xmax_ymax=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
               yolo_obb=[1] + [round(v, 6) for c in corners for v in (c[0] / W, c[1] / H)],
               geojson_polygon=[[round(a, 2), round(b, 2)] for a, b in geo + [geo[0]]],
               geotransform=gt0, crs="EPSG:32652",
               note="YOLO 클래스 번호는 0부터(소형선박=1). VOC 원래 자료는 1부터 세는 정수 화소 좌표를 씀(변환 때 ±1 주의). GeoJSON 규격(RFC 7946)은 원래 WGS84 경위도라 투영 좌표는 도구 관례로만 씀")
    R["formats"] = out


def noisy_labels(y, rate, rng, mode="sym"):
    y = y.clone()
    if mode == "sym":
        flip = torch.tensor(rng.random(len(y)) < rate)
        new = torch.tensor(rng.integers(0, 5, len(y)))
        new = new + (new >= y).long()                    # 원래 클래스가 아닌 다른 클래스로
        y[flip] = new[flip]
    else:                                                # 갯벌(5) → 수계(3)
        flip = (y == 5) & torch.tensor(rng.random(len(y)) < rate)
        y[flip] = 3
    return y, int(flip.sum())


def exp_noise():
    out = {}
    for name, rate, mode in [("clean", 0.0, "sym"), ("sym10", 0.1, "sym"), ("sym20", 0.2, "sym"), ("sym40", 0.4, "sym"),
                             ("tidal_to_water40", 0.4, "pair")]:
        rng = np.random.default_rng(1)
        y, nflip = noisy_labels(D["y_train"], rate, rng, mode)
        torch.manual_seed(0)
        m = TinyCNN()
        res = fit(m, D["x_train"], y, D["x_val"], D["y_val"], EV, epochs=30)
        with torch.no_grad():
            m.eval(); pred = m(D["x_test"]).argmax(1)
        rec = [r4(((pred == c) & (D["y_test"] == c)).sum() / (D["y_test"] == c).sum()) for c in range(6)]
        out[name] = dict(n_flipped=nflip, best_epoch=res["best_epoch"], test_last=res["last"]["test"], test_best=res["best"]["test"],
                         recall_best=rec, train_loss_last=res["hist"]["train_loss"][-1])
    out["note"] = "검증 라벨은 깨끗한 그대로. 대칭 잡음 = 정한 비율만큼 다른 다섯 클래스 중 하나로 무작위로 바꿈"
    R["noise"] = out


def exp_missing():
    from make_part6 import tiles, detect
    from part6_eval import nms_dets, coco_eval
    T = tiles()
    base = [(t["objs"], nms_dets(detect(t["objs"], t["clutter"], 1.0, seed=t["image_id"]), 0.5)) for t in T]
    out = {}
    for frac in (0.0, 0.1, 0.2, 0.3):
        rng = np.random.default_rng(5)
        imgs = [([o for o in g if rng.random() >= frac], d) for g, d in base]
        e = coco_eval(imgs)
        out[f"drop{int(frac * 100)}"] = dict(n_gt=sum(len(g) for g, _ in imgs), ap50=e["ap50"], ap=e["ap"],
                                           per_class_ap50={c: e["per_class"][c][0] for c in range(3)})
    out["note"] = "같은 탐지 결과(6부 기본 탐지 결과)를, 정답 라벨을 무작위로 일부 지운 채 평가. 지워진 객체를 맞힌 탐지가 오탐으로 셈"
    R["missing"] = out


# ------------------------------------------------------------------ 45 증강
def exp_aug():
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(D["y_train"]), generator=g)[:1000]
    xtr, ytr = D["x_train"][idx], D["y_train"][idx]
    rot = (torch.rot90(D["x_test"], 1, (2, 3)), D["y_test"])
    ev = dict(EV, test_rot90=rot)
    out = {}
    for name, aug, mix in [("none", None, 0), ("geo", "geo", 0), ("photo", "photo", 0), ("geo_photo", "geo_photo", 0),
                           ("band_shuffle", "band_shuffle", 0), ("geo_mixup0.2", "geo", 0.2)]:
        torch.manual_seed(0)
        res = fit(TinyCNN(), xtr, ytr, D["x_val"], D["y_val"], ev, epochs=60, aug=aug, mixup=mix)
        out[name] = slim(res)
    out["note"] = "학습 타일 1,000장(5부 학습 자료에서 무작위), 60에폭. test_rot90 = 시험 타일을 90° 돌린 것, shift = 연무·계절 타일"
    R["aug"] = out


# ------------------------------------------------------------------ 46 학습률
def exp_lrfind():
    torch.manual_seed(0)
    m = TinyCNN()
    o = torch.optim.SGD(m.parameters(), lr=1e-5, momentum=0.9)
    g = torch.Generator().manual_seed(0)
    lrs, losses, sm = [], [], 0.0
    n_it = 200
    m.train()
    for i in range(n_it):
        lr = 1e-5 * (10 / 1e-5) ** (i / (n_it - 1))
        for pg in o.param_groups:
            pg["lr"] = lr
        b = torch.randint(0, len(D["y_train"]), (64,), generator=g)
        loss = F.cross_entropy(m(D["x_train"][b]), D["y_train"][b])
        o.zero_grad(); loss.backward(); o.step()
        v = loss.item()
        if not math.isfinite(v) or v > 20:
            lrs.append(float(f"{lr:.4g}")); losses.append(None); break
        sm = 0.9 * sm + 0.1 * v
        lrs.append(float(f"{lr:.4g}")); losses.append(round(sm / (1 - 0.9 ** (i + 1)), 4))
    L = np.array([x if x is not None else np.nan for x in losses])
    i_min = int(np.nanargmin(L))
    grad = np.gradient(L[: i_min + 1], np.log10(lrs[: i_min + 1]))
    i_steep = 20 + int(np.nanargmin(grad[20:]))          # 처음 20걸음(이동 평균이 자리 잡기 전)은 뺌
    R["lrfind"] = dict(lrs=lrs, smoothed_loss=losses, lr_at_min_loss=lrs[i_min], lr_steepest=lrs[i_steep],
                       min_loss=r4(L[i_min]), diverged_at=lrs[len(losses) - 1] if losses[-1] is None else None,
                       note="SGD 모멘텀 0.9, 배치 64, 200걸음 동안 학습률을 1e-5에서 10까지 지수로 키움. 손실은 지수 이동 평균(0.9, 편향 보정)")


def exp_sched():
    out = {}
    for name, kw in [("const_0.05", dict(sched="const")), ("step_0.05", dict(sched="step", step_epochs=(20, 25))),
                     ("cos_0.05", dict(sched="cos")), ("onecycle_max0.1", dict(sched="onecycle", max_lr=0.1)),
                     ("warm2_cos_0.05", dict(sched="cos", warmup_epochs=2))]:
        torch.manual_seed(0)
        res = fit(TinyCNN(), D["x_train"], D["y_train"], D["x_val"], D["y_val"], EV, epochs=30, opt="momentum", lr=0.05, **kw)
        out[name] = slim(res)
    out["note"] = "VD-CNN, SGD 모멘텀 0.9, 배치 64, 30에폭(에폭당 66걸음). 계단은 20·25에폭에 ×0.1. 원사이클은 최고 0.1(시작 0.004)"
    R["sched"] = out


def exp_warmup():
    out = {}
    for name, w in [("nowarm", 0), ("warm3", 3)]:
        torch.manual_seed(0)
        res = fit(TinyCNN(norm="none"), D["x_train"], D["y_train"], D["x_val"], D["y_val"], EV, epochs=20, opt="adam", lr=0.01,
                  sched="cos", warmup_epochs=w)
        out[name] = slim(res)
    out["note"] = "정규화 없는 VD-CNN, Adam 0.01(기본의 10배), 코사인, 20에폭"
    R["warmup"] = out


def exp_bscale():
    out = {}
    for name, bs, lr, w in [("bs32_lr0.025", 32, 0.025, 0), ("bs256_lr0.025", 256, 0.025, 0), ("bs256_lr0.2", 256, 0.2, 0),
                            ("bs256_lr0.2_warm2", 256, 0.2, 2)]:
        torch.manual_seed(0)
        res = fit(TinyCNN(), D["x_train"], D["y_train"], D["x_val"], D["y_val"], EV, epochs=30, bs=bs, opt="momentum", lr=lr,
                  sched="cos", warmup_epochs=w)
        out[name] = slim(res)
        out[name]["steps"] = math.ceil(4200 / bs) * 30
    out["note"] = "SGD 모멘텀 0.9, 코사인, 30에폭 고정. 배치를 8배로 키우면 걸음 수가 8분의 1"
    R["bscale"] = out


# ------------------------------------------------------------------ 47 배치·자원
def grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def exp_accum():
    out = {}
    x, y = D["x_train"][:64], D["y_train"][:64]
    for norm in ("bn", "gn"):
        torch.manual_seed(0); m1 = TinyCNN(norm=norm); m1.train()
        m2 = TinyCNN(norm=norm); m2.load_state_dict(m1.state_dict()); m2.train()
        m1.zero_grad(); F.cross_entropy(m1(x), y).backward(); g1 = grads(m1)
        m2.zero_grad()
        for i in range(0, 64, 16):
            (F.cross_entropy(m2(x[i:i + 16]), y[i:i + 16]) / 4).backward()
        g2 = grads(m2)
        out[f"grad_diff_{norm}"] = dict(max_abs=float(f"{(g1 - g2).abs().max():.3e}"), rel=float(f"{((g1 - g2).norm() / g1.norm()):.3e}"))
    for name, norm, bs, acc in [("bn_bs64", "bn", 64, 1), ("bn_bs16x4", "bn", 16, 4), ("gn_bs64", "gn", 64, 1), ("gn_bs16x4", "gn", 16, 4)]:
        torch.manual_seed(0)
        res = fit(TinyCNN(norm=norm), D["x_train"], D["y_train"], D["x_val"], D["y_val"], EV, epochs=10, bs=bs, accum=acc)
        out[name] = slim(res)
    R["accum"] = out


def exp_bf16():
    out = {}
    for name, amp in (("fp32", None), ("bf16", "bf16")):
        torch.manual_seed(0)
        res = fit(TinyCNN(), D["x_train"], D["y_train"], D["x_val"], D["y_val"], EV, epochs=30, amp=amp)
        out[name] = slim(res)
    fi16, fib = torch.finfo(torch.float16), torch.finfo(torch.bfloat16)
    fi32 = torch.finfo(torch.float32)
    out["finfo"] = {n: dict(max=float(f"{f.max:.4g}"), min_normal=float(f"{f.tiny:.4g}"), eps=float(f"{f.eps:.4g}"), bits=f.bits)
                    for n, f in (("fp16", fi16), ("bf16", fib), ("fp32", fi32))}
    out["fp16_min_subnormal"] = float(f"{2 ** -24:.4g}")
    g = torch.tensor([3e-8, 1e-8, 1e-4, 70000.0])
    out["cast_examples"] = dict(values=g.tolist(), fp16=[float(v) for v in g.half().float()], bf16=[float(v) for v in g.bfloat16().float()],
                                fp16_scaled_1024=[float(v) for v in (g * 1024).half().float() / 1024])
    out["note"] = "CPU에서 torch.autocast(bfloat16). CPU 시간은 GPU와 성질이 달라 속도 비교로 쓰지 않음"
    R["bf16"] = out


def exp_memory():
    import torchvision.models as M
    out = {}
    def act_bytes(model, inp):
        tot = [0]
        hooks = [mod.register_forward_hook(lambda m, i, o: tot.__setitem__(0, tot[0] + (o.numel() * o.element_size() if torch.is_tensor(o) else 0)))
                 for mod in model.modules() if len(list(mod.children())) == 0]
        model.eval()
        with torch.no_grad():
            model(inp)
        for h in hooks:
            h.remove()
        return tot[0]
    for name, model, inp in [("resnet50_512", M.resnet50(weights=None), torch.zeros(1, 3, 512, 512)),
                             ("resnet50_224", M.resnet50(weights=None), torch.zeros(1, 3, 224, 224)),
                             ("vdcnn_32", TinyCNN(), torch.zeros(1, 4, 32, 32))]:
        p = n_params(model)
        a = act_bytes(model, inp)
        out[name] = dict(params=p, weights_MB=round(p * 4 / 2 ** 20, 1), grads_MB=round(p * 4 / 2 ** 20, 1),
                         adam_states_MB=round(p * 8 / 2 ** 20, 1), act_per_image_MB_fp32=round(a / 2 ** 20, 1),
                         act_batch16_GB_fp32=round(a * 16 / 2 ** 30, 2), act_batch16_GB_half=round(a * 8 / 2 ** 30, 2))
    out["note"] = "활성값은 모든 말단 층 출력 크기의 합(순전파 한 번)으로 어림. 실제로 역전파를 위해 저장되는 텐서와 같지 않고(제자리 ReLU, 프레임워크 작업 공간 등) 대략의 크기만 봄"
    R["memory"] = out


class _SlowDS(torch.utils.data.Dataset):
    """타일 하나를 읽고(압축 해제를 흉내 낸 계산) 증강하는 데이터셋"""
    def __init__(self, x):
        self.x = x.numpy()
    def __len__(self):
        return 1024
    def __getitem__(self, i):
        a = self.x[i % len(self.x)]
        for _ in range(40):                              # 디코딩 비용 흉내
            a = np.ascontiguousarray(np.rot90(a, 1, (1, 2)))
            _ = np.fft.rfft2(a)
        return torch.tensor(a.copy())


def exp_loader():
    out = {}
    ds = _SlowDS(D["x_train"][:256])
    for nw in (0, 1, 2):
        dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=nw, shuffle=False)
        t = time.time()
        for b in dl:
            pass
        out[f"workers{nw}"] = round(time.time() - t, 2)
    out["note"] = "타일 1,024개를 배치 64로 한 번 읽는 시간(초). 이 실행 환경은 CPU 코어 2개라 작업자를 늘려도 2배 안팎이 한계"
    out["cpu_count"] = os.cpu_count()
    R["loader"] = out


# ------------------------------------------------------------------ 48 전이학습
def target_tiles(n_per_class, seed):
    """연무·계절 도메인(5부 test_shift와 같은 변환)의 학습용 타일. 학습 지역 0~20에서 새로 뽑음"""
    rng = np.random.default_rng(seed)
    gains = {r: np.exp(np.random.default_rng(1000 + r).normal(0, 0.12, 4)).astype(np.float32) for r in range(21)}
    X, Y = [], []
    for c in range(6):
        for _ in range(n_per_class):
            r = int(rng.integers(0, 21))
            x, m = P5._tile(rng, c, gains[r])
            x = x.copy()
            veg = (m == 0) | (m == 1)
            x[3][veg] *= 0.78
            x += np.array([0.035, 0.022, 0.010, 0.004], np.float32)[:, None, None]
            X.append(x); Y.append(np.bincount(m.ravel(), minlength=6).argmax())
    p = rng.permutation(len(X))
    return to_tensor(np.stack(X)[p]), torch.tensor(np.array(Y)[p])


def base_model():
    m = TinyCNN()
    m.load_state_dict(torch.load(os.path.join(HERE, ".part5_base.pt")))
    return m


def exp_tl():
    xv, yv = target_tiles(25, 77)                       # 대상 도메인 검증 150장
    test = {"shift_test": (D["x_shift"], D["y_shift"]), "orig_test": (D["x_test"], D["y_test"])}
    out = {}
    m = base_model()
    out["source_only"] = {k: round(evaluate(m, x, y)[1], 4) for k, (x, y) in test.items()}
    # AdaBN: 라벨 없이 대상 도메인 영상으로 배치 정규화 러닝 통계만 다시 잼
    for n_unl in (60, 600):
        m = base_model()
        xu, _ = target_tiles(n_unl // 6, 99)
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.reset_running_stats(); mod.momentum = None    # 누적 평균
        m.train()
        with torch.no_grad():
            for i in range(0, len(xu), 60):
                m(xu[i:i + 60])
        out[f"adabn_{n_unl}"] = {k: round(evaluate(m, x, y)[1], 4) for k, (x, y) in test.items()}
    for n in (10, 50):
        xt, yt = target_tiles(n, 55)
        res_n = {}
        for name in ("scratch", "linear_probe", "freeze_block1", "finetune_all", "finetune_all_adabn"):
            torch.manual_seed(0)
            if name == "scratch":
                m = TinyCNN(); lr = 1e-3
            else:
                m = base_model(); lr = 1e-4 if name.startswith("finetune") else 1e-3
            if name == "finetune_all_adabn":
                for mod in m.modules():
                    if isinstance(mod, nn.BatchNorm2d):
                        mod.reset_running_stats(); mod.momentum = None
                m.train()
                with torch.no_grad():
                    m(xt)
                for mod in m.modules():
                    if isinstance(mod, nn.BatchNorm2d):
                        mod.momentum = 0.1
            if name == "linear_probe":
                for p in m.features.parameters():
                    p.requires_grad = False
            if name == "freeze_block1":
                for p in list(m.features.parameters())[:4]:     # 첫 합성곱 + 첫 배치 정규화
                    p.requires_grad = False
            if name == "linear_probe":
                class Frozen(nn.Module):                       # 특징 추출부는 추론 모드로 고정(러닝 통계도 그대로)
                    def __init__(s, mm):
                        super().__init__(); s.mm = mm
                    def forward(s, x):
                        s.mm.features.eval()
                        return s.mm.head(s.mm.features(x))
                mm = Frozen(m)
            else:
                mm = m
            trainable = sum(p.numel() for p in mm.parameters() if p.requires_grad)
            res = fit(mm, xt, yt, xv, yv, test, epochs=60 if n == 10 else 40, bs=32, lr=lr)
            res_n[name] = dict(trainable_params=trainable, best_epoch=res["best_epoch"], **{f"{k}_best": v for k, v in res["best"].items()},
                               **{f"{k}_last": v for k, v in res["last"].items()})
        out[f"n{n * 6}"] = res_n
    out["note"] = ("원래(원천) 도메인 = 5부 학습 자료로 학습한 VD-CNN(best). 대상 도메인 = 연무·계절 타일. 라벨 있는 대상 타일 60·300장, "
                   "검증 150장(대상), 시험 = 5부 test_shift(900장, 대상)와 원래 시험(900장). 미세조정 학습률 1e-4, 처음부터·선형 탐침 1e-3, 배치 32")
    R["tl"] = out


def exp_rgb4():
    out = {}
    # 1) 원천 도메인에서 RGB 3밴드(B2 B3 B4)만으로 사전학습
    class CNN3(TinyCNN):
        pass
    torch.manual_seed(0)
    m3 = TinyCNN(); m3.features[0] = nn.Conv2d(3, 16, 3, padding=1)
    res = fit(m3, D["x_train"][:, :3], D["y_train"], D["x_val"][:, :3], D["y_val"],
              {"test": (D["x_test"][:, :3], D["y_test"])}, epochs=30)
    out["rgb_pretrain_test"] = res["best"]["test"]
    xt, yt = target_tiles(50, 55)
    xv, yv = target_tiles(25, 77)
    ev = {"shift_test": (D["x_shift"], D["y_shift"])}
    W3 = m3.features[0].weight.data.clone(); b3 = m3.features[0].bias.data.clone()
    for name in ("scratch4", "nir_zero", "nir_mean", "nir_random"):
        torch.manual_seed(0)
        m = TinyCNN()
        if name != "scratch4":
            sd = {k: v for k, v in m3.state_dict().items() if not k.startswith("features.0.")}
            m.load_state_dict(sd, strict=False)
            w = torch.zeros(16, 4, 3, 3)
            w[:, :3] = W3
            if name == "nir_mean":
                w[:, 3] = W3.mean(1)
            elif name == "nir_random":
                w[:, 3] = torch.randn(16, 3, 3) * W3.std()
            m.features[0].weight.data = w; m.features[0].bias.data = b3.clone()
        res = fit(m, xt, yt, xv, yv, ev, epochs=40, bs=32, lr=1e-3 if name == "scratch4" else 3e-4)
        out[name] = dict(best_epoch=res["best_epoch"], shift_test_best=res["best"]["shift_test"], shift_test_last=res["last"]["shift_test"])
    out["note"] = "원천 도메인에서 B2·B3·B4만으로 VD-CNN을 사전학습한 뒤, 4밴드 첫 합성곱으로 옮겨 대상 도메인 300장으로 미세조정(학습률 3e-4). 근적외 채널 가중치 초기화만 다름"
    R["rgb4"] = out


# ------------------------------------------------------------------ 49 재현성
def exp_seeds():
    out = {}
    for cfg, norm in (("bn", "bn"), ("no_bn", "none")):
        rows = []
        for s in range(5):
            torch.manual_seed(s)
            res = fit(TinyCNN(norm=norm), D["x_train"], D["y_train"], D["x_val"], D["y_val"], EV, epochs=30, seed=s)
            rows.append(dict(seed=s, best_epoch=res["best_epoch"], test_last=res["last"]["test"], test_best=res["best"]["test"],
                             shift_best=res["best"]["shift"]))
        tl = np.array([r["test_last"] for r in rows]); tb = np.array([r["test_best"] for r in rows])
        out[cfg] = dict(runs=rows, last_mean=r4(tl.mean()), last_sd=r4(tl.std(ddof=1)), best_mean=r4(tb.mean()), best_sd=r4(tb.std(ddof=1)),
                        best_min=r4(tb.min()), best_max=r4(tb.max()))
    # 같은 시드 두 번 → 같은가
    a = []
    for _ in range(2):
        torch.manual_seed(0)
        res = fit(TinyCNN(), D["x_train"], D["y_train"], D["x_val"], D["y_val"], EV, epochs=3, seed=0)
        a.append(res["hist"]["train_loss"])
    out["same_seed_identical"] = a[0] == a[1]
    out["same_seed_losses"] = a
    out["note"] = "시드가 정하는 것: 가중치 초기값, 학습 순서(섞기), 증강 난수. 이 실험의 학습 자료는 같고 시드만 바꿈"
    R["seeds"] = out


# ------------------------------------------------------------------ 50 내보내기·배포
def exp_export():
    import onnx, onnxruntime as ort
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
    tmp = os.path.join(HERE, ".part7_tmp"); os.makedirs(tmp, exist_ok=True)
    out = {}
    m = base_model(); m.eval()
    p32 = os.path.join(tmp, "vdcnn.onnx")
    torch.onnx.export(m, torch.zeros(1, 4, 32, 32), p32, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s32 = ort.InferenceSession(p32, so, providers=["CPUExecutionProvider"])
    x = D["x_test"].numpy()
    with torch.no_grad():
        ref = m(D["x_test"]).numpy()
    o32 = s32.run(None, {"x": x})[0]
    out["onnx_fp32"] = dict(size_KB=round(os.path.getsize(p32) / 1024, 1), max_abs_diff=float(f"{np.abs(o32 - ref).max():.3e}"),
                            acc=r4((o32.argmax(1) == D["y_test"].numpy()).mean()), same_argmax=r4((o32.argmax(1) == ref.argmax(1)).mean()))

    class Reader(CalibrationDataReader):
        def __init__(s):
            s.it = iter([{"x": D["x_train"][i:i + 50].numpy()} for i in range(0, 500, 50)])
        def get_next(s):
            return next(s.it, None)
    pq = os.path.join(tmp, "vdcnn_int8.onnx")
    quantize_static(p32, pq, Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    sq = ort.InferenceSession(pq, so, providers=["CPUExecutionProvider"])
    oq = sq.run(None, {"x": x})[0]
    out["onnx_int8_static"] = dict(size_KB=round(os.path.getsize(pq) / 1024, 1), acc=r4((oq.argmax(1) == D["y_test"].numpy()).mean()),
                                   same_argmax_vs_fp32=r4((oq.argmax(1) == ref.argmax(1)).mean()),
                                   shift_acc=r4((sq.run(None, {"x": D["x_shift"].numpy()})[0].argmax(1) == D["y_shift"].numpy()).mean()),
                                   shift_acc_fp32=r4((s32.run(None, {"x": D["x_shift"].numpy()})[0].argmax(1) == D["y_shift"].numpy()).mean()),
                                   calib="학습 타일 500장")
    # 지연시간·처리량
    def bench(fn, n_rep):
        fn(); t = []
        for _ in range(n_rep):
            t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
        return float(np.median(t))
    lat = {}
    for bsz in (1, 64):
        xb = x[:bsz]
        tb = torch.tensor(xb)
        with torch.no_grad():
            t_pt = bench(lambda: m(tb), 50 if bsz == 1 else 20)
        t_32 = bench(lambda: s32.run(None, {"x": xb}), 50 if bsz == 1 else 20)
        t_q = bench(lambda: sq.run(None, {"x": xb}), 50 if bsz == 1 else 20)
        lat[f"batch{bsz}"] = {k: dict(ms=round(v * 1000, 3), tiles_per_s=round(bsz / v, 1)) for k, v in
                             (("pytorch_fp32", t_pt), ("onnxruntime_fp32", t_32), ("onnxruntime_int8", t_q))}
    out["latency_cpu_1thread"] = lat
    out["note"] = "CPU 1스레드, 중앙값. 숫자는 이 실행 환경 기준이라 장비가 바뀌면 달라짐. TensorRT는 GPU가 없어 실험하지 않음"
    out["versions"] = dict(onnx=onnx.__version__, onnxruntime=ort.__version__, opset=17)
    R["export"] = out


def unet():
    """7부 장면 0~5의 타일(32×32, 간격 16)로 학습한 U-Net(스킵 있음). 없으면 학습해 .part7_unet.pt에 저장(git 제외)"""
    p = os.path.join(HERE, ".part7_unet.pt")
    m = TinyUNet(skip=True)
    if os.path.exists(p):
        m.load_state_dict(torch.load(p))
    else:
        S = scenes()
        tr = [tile_scene(S[k], 32, 16) for k in range(6)]; va = [tile_scene(S[k], 32, 32) for k in (6, 7)]
        xt = to_tensor(np.concatenate([t["x"] for t in tr])); mt = torch.tensor(np.concatenate([t["m"] for t in tr]).astype(np.int64))
        xv = to_tensor(np.concatenate([t["x"] for t in va])); mv = torch.tensor(np.concatenate([t["m"] for t in va]).astype(np.int64))
        torch.manual_seed(0)
        res = fit(m, xt, mt, xv, mv, epochs=20, aug="geo")
        R.setdefault("sliding", {})["unet_train"] = dict(n_tiles=len(mt), best_epoch=res["best_epoch"], val_pixel_acc_best=res["best"]["val"],
                                                         note="장면 0~5 타일(간격 16) 학습, 장면 6~7 검증, 기하 증강, Adam 0.001, 20에폭")
        torch.save(m.state_dict(), p)
    m.eval()
    return m


def unet_part5():
    m = TinyUNet(skip=True)
    m.load_state_dict(torch.load(os.path.join(HERE, ".part5_unet_skip.pt"))); m.eval()
    return m


def infer_scene(m, x, mode):
    """x (4,H,W) 표준화 텐서 → 로짓 (6,H,W)"""
    H = x.shape[1]
    with torch.no_grad():
        if mode == "whole":
            return m(x[None])[0]
        if mode == "tiles":
            out = torch.zeros(6, H, H)
            for r in range(0, H, 32):
                for c in range(0, H, 32):
                    out[:, r:r + 32, c:c + 32] = m(x[None, :, r:r + 32, c:c + 32])[0]
            return out
        if mode in ("overlap_avg", "overlap_center"):
            out = torch.zeros(6, H, H); cnt = torch.zeros(1, H, H)
            st = list(range(0, H - 32 + 1, 16))
            w = torch.ones(1, 32, 32)
            if mode == "overlap_center":                  # 가장자리 8화소는 무게를 낮춤(가운데 우선)
                w = torch.full((1, 32, 32), 0.05); w[:, 8:24, 8:24] = 1.0
            for r in st:
                for c in st:
                    out[:, r:r + 32, c:c + 32] += m(x[None, :, r:r + 32, c:c + 32])[0].softmax(0) * w
                    cnt[:, r:r + 32, c:c + 32] += w
            return out / cnt


def exp_sliding():
    S = scenes()
    m = unet()
    keep = R.get("sliding", {}).get("unet_train")
    seam = np.zeros((256, 256), bool)
    for k in range(32, 256, 32):
        seam[:, k - 2:k + 2] = True; seam[k - 2:k + 2, :] = True
    out = {}
    for mode in ("tiles", "overlap_avg", "overlap_center", "whole"):
        accs, seams, inner, mi = [], [], [], []
        for k in (8, 9):
            x = to_tensor(S[k]["x"][None])[0]
            pred = infer_scene(m, x, mode).argmax(0).numpy()
            y = S[k]["m"]
            ok = pred == y
            accs.append(ok.mean()); seams.append(ok[seam].mean()); inner.append(ok[~seam].mean())
            mi.append(miou(torch.tensor(pred), torch.tensor(y.astype(np.int64)))[0])
        out[mode] = dict(pixel_acc=r4(np.mean(accs)), seam_acc=r4(np.mean(seams)), non_seam_acc=r4(np.mean(inner)), miou=r4(np.mean(mi)))
    out["seam_frac"] = r4(seam.mean())
    # 참고: 5부 타일로 학습한 U-Net을 그대로 쓰면(자료 생성 방식이 조금 다름)
    m5 = unet_part5()
    a5 = [((infer_scene(m5, to_tensor(S[k]["x"][None])[0], "overlap_avg").argmax(0).numpy()) == S[k]["m"]).mean() for k in (8, 9)]
    out["part5_unet_overlap_avg_pixel_acc"] = r4(np.mean(a5))
    out["unet_train"] = keep
    out["note"] = ("7부 장면 0~5로 학습한 U-Net(스킵 있음, best)을 시험 장면 8·9(256×256)에 적용. tiles = 32 타일 겹침 없이 이어 붙임, "
                   "overlap_avg = 간격 16으로 겹쳐 확률 평균, overlap_center = 겹치되 가장자리 8화소 무게 0.05, whole = 장면을 통째로 넣음. "
                   "seam = 타일 경계에서 2화소 안")
    R["sliding"] = out


def exp_polygon():
    import rasterio.features as RF
    from rasterio.transform import Affine
    from shapely.geometry import shape
    S = scenes()
    k = 8
    x = to_tensor(S[k]["x"][None])[0]
    pred = infer_scene(unet(), x, "overlap_avg").argmax(0).numpy().astype(np.uint8)
    gt = S[k]["m"].astype(np.uint8)
    T = Affine.from_gdal(*GEOTRANSFORM(k))
    out = dict(geotransform=GEOTRANSFORM(k), crs="EPSG:32652",
               pixel_to_map_example=dict(pixel_col_row=[100, 50], center_map_xy=[GEOTRANSFORM(k)[0] + 100.5 * 10, GEOTRANSFORM(k)[3] - 50.5 * 10]))
    def polys(a, sieve=0, simplify=0.0):
        if sieve:
            a = RF.sieve(a, size=sieve, connectivity=8)
        res = {c: dict(n=0, area_ha=0.0, vertices=0) for c in range(6)}
        for geom, val in RF.shapes(a, transform=T, connectivity=8):
            g = shape(geom)
            if simplify:
                g = g.simplify(simplify, preserve_topology=True)
            c = int(val)
            res[c]["n"] += 1; res[c]["area_ha"] += g.area / 1e4
            res[c]["vertices"] += len(g.exterior.coords) + sum(len(i.coords) for i in g.interiors)
        for c in res:
            res[c]["area_ha"] = round(res[c]["area_ha"], 2)
        return res, a
    out["gt"], _ = polys(gt)
    out["pred_raw"], _ = polys(pred)
    out["pred_sieve10"], sv = polys(pred, sieve=10)
    out["pred_sieve10_simplify10m"], _ = polys(pred, sieve=10, simplify=10.0)
    out["pixel_acc_raw"] = r4((pred == gt).mean()); out["pixel_acc_sieve10"] = r4((sv == gt).mean())
    out["note"] = "장면 8, U-Net 겹쳐 평균 추론 결과를 rasterio.features.shapes(8-연결)로 폴리곤화. sieve10 = 10화소(0.1 ha) 미만 덩어리를 이웃에 합침, simplify10m = 더글라스-포이커 허용 10 m"
    R["polygon"] = out


EXPS = {k[4:]: v for k, v in list(globals().items()) if k.startswith("exp_")}

if __name__ == "__main__":
    names = sys.argv[1:] or list(EXPS)
    for nme in names:
        t0 = time.time()
        EXPS[nme]()
        R.setdefault("_meta", {})[nme] = round(time.time() - t0, 1)
        save()
        print(nme, "done", round(time.time() - t0, 1), "s", flush=True)
