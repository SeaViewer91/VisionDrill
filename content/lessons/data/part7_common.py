"""7부 공통: 유연한 학습 루프(증강, 학습률 스케줄·워밍업, 기울기 누적, bf16 자동 혼합 정밀도, 믹스업).

모델은 5부 것(part5_common의 TinyCNN = VD-CNN, TinyUNet)을 그대로 씀. 입력 표준화도 5부 학습 자료의 밴드 평균·표준편차.
재현: torch 2.14 CPU, 스레드 1개, 시드 고정(part5_common을 불러오면 설정됨).
"""
import math, time
import numpy as np
import torch
import torch.nn.functional as F

import part5_common as P5C
from part5_common import TinyCNN, TinyUNet, evaluate, n_params, make_opt

_STATS = None


def norm_stats():
    global _STATS
    if _STATS is None:
        D = P5C.data()
        _STATS = (D["mu"].reshape(1, 4, 1, 1), D["sd"].reshape(1, 4, 1, 1))
    return _STATS


def to_tensor(x):
    mu, sd = norm_stats()
    return torch.tensor((x - mu) / sd, dtype=torch.float32)


# ------------------------------------------------------------------ 증강 (표준화 전 반사율이 아니라 표준화 뒤 텐서에 적용)
def aug_geo(x, g):
    """좌우·상하 뒤집기와 90° 회전(위에서 내려다본 영상이라 방향이 의미 없다고 봄)"""
    if torch.rand(1, generator=g) < 0.5:
        x = x.flip(3)
    if torch.rand(1, generator=g) < 0.5:
        x = x.flip(2)
    k = int(torch.randint(0, 4, (1,), generator=g))
    return torch.rot90(x, k, (2, 3))


def aug_photo(x, g):
    """밴드별 밝기 이득 ±15%와 청·녹 밴드에 더해지는 산란광(연무 흉내), 잡음. 표준화 공간에서 근사"""
    mu, sd = norm_stats()
    mu, sd = torch.tensor(mu), torch.tensor(sd)
    refl = x * sd + mu
    n = len(x)
    gain = torch.exp(torch.randn(n, 4, 1, 1, generator=g) * 0.08).clamp(0.85, 1.15)
    haze = torch.rand(n, 1, 1, 1, generator=g) * torch.tensor([0.035, 0.022, 0.010, 0.004]).view(1, 4, 1, 1)
    veg = torch.where(torch.rand(n, 1, 1, 1, generator=g) < 0.5, 1.0, 0.85)       # 계절(근적외 감소) 흉내: 절반에만
    nir = torch.ones(n, 4, 1, 1); nir[:, 3:4] = veg
    refl = refl * gain * nir + haze + torch.randn(refl.shape, generator=g) * 0.004
    return (refl - mu) / sd


def aug_geo_photo(x, g):
    return aug_photo(aug_geo(x, g), g)


def aug_band_shuffle(x, g):
    """원격탐사에서 하면 안 되는 증강의 예: 밴드 순서를 무작위로 섞음(분광 특성이 깨짐)"""
    x = aug_geo(x, g)
    out = x.clone()
    for i in range(len(x)):
        if torch.rand(1, generator=g) < 0.5:
            out[i] = x[i][torch.randperm(4, generator=g)]
    return out


AUGS = {None: None, "geo": aug_geo, "photo": aug_photo, "geo_photo": aug_geo_photo, "band_shuffle": aug_band_shuffle}


# ------------------------------------------------------------------ 학습률 스케줄
def lr_at(it, total, base, sched, warmup=0, step_at=(), max_lr=None):
    if warmup and it < warmup:
        return base * (it + 1) / warmup
    t = (it - warmup) / max(1, total - warmup)
    if sched in (None, "const"):
        return base
    if sched == "step":
        e = sum(1 for s in step_at if it >= s)
        return base * (0.1 ** e)
    if sched == "cos":
        return base * 0.5 * (1 + math.cos(math.pi * t))
    if sched == "onecycle":       # torch OneCycleLR와 같은 모양(코사인, 30% 지점 최고, 시작 max/25, 끝 max/1e4)
        m = max_lr or base
        p = 0.3
        f = it / max(1, total - 1)
        if f < p:
            a, b, u = m / 25, m, f / p
        else:
            a, b, u = m, m / 25 / 1e4, (f - p) / (1 - p)
        return b + (a - b) * 0.5 * (1 + math.cos(math.pi * u))
    raise ValueError(sched)


# ------------------------------------------------------------------ 학습
def fit(model, xtr, ytr, xval, yval, evals=None, epochs=30, bs=64, opt="adam", lr=1e-3, wd=0.0, sched=None,
        warmup_epochs=0, step_epochs=(), max_lr=None, accum=1, amp=None, aug=None, mixup=0.0, seed=0,
        seg=False, record_iter=False, label_smoothing=0.0):
    """반환: hist(에폭별 train_loss·val_loss·val_acc·lr, 초), last·best(검증 손실 최소 에폭)의 evals 정확도"""
    import os
    if os.environ.get("PART7_SMOKE"):
        epochs = 1
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    o = make_opt(opt, model.parameters(), lr, wd)
    steps_per_epoch = math.ceil(len(xtr) / (bs * accum))
    total = steps_per_epoch * epochs
    warm = warmup_epochs * steps_per_epoch
    step_at = [e * steps_per_epoch for e in step_epochs]
    augf = AUGS[aug] if isinstance(aug, (str, type(None))) else aug
    hist = {k: [] for k in ["train_loss", "val_loss", "val_acc", "lr", "sec"]}
    if record_iter:
        hist["iter_loss"], hist["iter_lr"] = [], []
    best = (float("inf"), None, -1)
    it = 0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(xtr), generator=g)
        tl, n = 0.0, 0
        o.zero_grad()
        micro = 0
        for i in range(0, len(xtr), bs):
            b = perm[i:i + bs]
            x, y = xtr[b], ytr[b]
            if augf is not None:
                x = augf(x, g)
            cur = lr_at(it, total, lr, sched, warm, step_at, max_lr)
            for pg in o.param_groups:
                pg["lr"] = cur
            if mixup > 0:
                lam = float(np.random.default_rng(seed * 100000 + it).beta(mixup, mixup))
                j = torch.randperm(len(x), generator=g)
                x = lam * x + (1 - lam) * x[j]
            ctx = torch.autocast("cpu", dtype=torch.bfloat16) if amp == "bf16" else torch.autocast("cpu", enabled=False)
            with ctx:
                out = model(x)
                if mixup > 0:
                    loss = lam * F.cross_entropy(out.float(), y) + (1 - lam) * F.cross_entropy(out.float(), y[j])
                else:
                    loss = F.cross_entropy(out.float(), y, label_smoothing=label_smoothing)
            (loss / accum).backward()
            micro += 1
            if micro % accum == 0 or i + bs >= len(xtr):
                o.step(); o.zero_grad(); it += 1
            if not torch.isfinite(loss):
                tl = float("nan"); break
            tl += loss.item() * len(b); n += len(b)
            if record_iter:
                hist["iter_loss"].append(round(loss.item(), 4)); hist["iter_lr"].append(cur)
        vl, va, _ = evaluate(model, xval, yval)
        hist["train_loss"].append(round(tl / max(n, 1), 4) if tl == tl else None)
        hist["val_loss"].append(round(vl, 4)); hist["val_acc"].append(round(va, 4))
        hist["lr"].append(float(f"{cur:.3g}")); hist["sec"].append(round(time.time() - t0, 2))
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in model.state_dict().items()}, ep + 1)
        if tl != tl:
            break
    res = {"hist": hist, "params": n_params(model), "best_epoch": best[2]}
    for tag, state in (("last", None), ("best", best[1])):
        if state is not None:
            model.load_state_dict(state)
        res[tag] = {name: round(evaluate(model, x, y)[1], 4) for name, (x, y) in (evals or {}).items()}
        res[tag]["val"] = round(evaluate(model, xval, yval)[1], 4)
    return res
