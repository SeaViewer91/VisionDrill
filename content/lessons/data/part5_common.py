"""5부 공통: 모델 정의와 학습 루프 (PyTorch, CPU).

자료는 make_part5.load(). 본문 수치는 run_part5.py가 이 파일로 학습해 part5_runs.json에 저장한 값임.
재현: torch 2.14 CPU, 스레드 1개, 시드 고정. 다른 환경에서는 소수 셋째 자리 정도가 달라질 수 있음.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from make_part5 import load

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def data(normalize=True):
    """학습 자료의 밴드별 평균·표준편차로 표준화한 텐서 dict"""
    D = load()
    mu = D["x_train"].mean(axis=(0, 2, 3), keepdims=True)
    sd = D["x_train"].std(axis=(0, 2, 3), keepdims=True)
    out = {"mu": mu.ravel(), "sd": sd.ravel()}
    for k in ["train", "val", "test", "shift"]:
        x = D[f"x_{k}"]
        if normalize:
            x = (x - mu) / sd
        out[f"x_{k}"] = torch.tensor(x, dtype=torch.float32)
        out[f"y_{k}"] = torch.tensor(D[f"y_{k}"])
        out[f"m_{k}"] = torch.tensor(D[f"m_{k}"].astype(np.int64))
    return out


# ------------------------------------------------------------------ 모델
class MeanMLP(nn.Module):
    """타일의 밴드 평균 4개만 보는 다층 퍼셉트론 (무늬를 못 봄)"""
    def __init__(self, hidden=(32,), act="relu"):
        super().__init__()
        layers, d = [], 4
        for h in hidden:
            layers += [nn.Linear(d, h), ACT[act]()]
            d = h
        layers.append(nn.Linear(d, 6))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.mean(dim=(2, 3)))


class FlatMLP(nn.Module):
    """타일 화소 4×32×32 = 4,096개를 한 줄로 펴서 넣는 다층 퍼셉트론"""
    def __init__(self, hidden=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(4 * 32 * 32, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 6))

    def forward(self, x):
        return self.net(x)


ACT = {"relu": nn.ReLU, "leaky": lambda: nn.LeakyReLU(0.01), "gelu": nn.GELU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}


def _norm(kind, c):
    if kind == "bn":
        return nn.BatchNorm2d(c)
    if kind == "gn":
        return nn.GroupNorm(4, c)
    if kind == "ln":
        return nn.GroupNorm(1, c)   # 채널·공간 전체를 한 묶음으로 정규화(합성곱 층에서의 레이어 정규화)
    return nn.Identity()


class TinyCNN(nn.Module):
    """5부 기본 모델 (본문에서 'VD-CNN'이라 부름)
    입력 4×32×32
    conv3×3(4→16) → 정규화 → ReLU → 최대 풀링 2   : 16×16×16
    conv3×3(16→32) → 정규화 → ReLU → 최대 풀링 2  : 32×8×8
    conv3×3(32→64) → 정규화 → ReLU               : 64×8×8
    전역 평균 풀링 → 드롭아웃 → 선형(64→6)          : 로짓 6개
    """
    def __init__(self, norm="bn", dropout=0.0, width=16):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(4, w, 3, padding=1), _norm(norm, w), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1), _norm(norm, 2 * w), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * w, 4 * w, 3, padding=1), _norm(norm, 4 * w), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * w, 6))

    def forward(self, x):
        return self.head(self.features(x))


class DeepNet(nn.Module):
    """깊이 실험용: 3×3 합성곱 층 depth개(폭 16, 정규화 없음). residual=True면 두 층마다 잔차 연결"""
    def __init__(self, depth=20, residual=False, act="relu"):
        super().__init__()
        self.stem = nn.Conv2d(4, 16, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(16, 16, 3, padding=1) for _ in range(depth - 1)])
        self.residual = residual
        self.act = ACT[act]()
        self.fc = nn.Linear(16, 6)

    def forward(self, x):
        h = self.act(self.stem(x))
        i = 0
        while i < len(self.convs):
            if self.residual and i + 1 < len(self.convs):
                r = self.convs[i + 1](self.act(self.convs[i](h)))
                h = self.act(h + r)
                i += 2
            else:
                h = self.act(self.convs[i](h))
                i += 1
        return self.fc(h.mean(dim=(2, 3)))


class TinyViT(nn.Module):
    """작은 비전 트랜스포머: 패치 4×4 → 토큰 64개, 차원 64, 인코더 4층, 헤드 4개, 분류 토큰"""
    def __init__(self, patch=4, dim=64, depth=4, heads=4, pos=True):
        super().__init__()
        n = (32 // patch) ** 2
        self.embed = nn.Conv2d(4, dim, patch, stride=patch)    # 패치 임베딩(패치마다 같은 선형 사상)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02) if pos else None
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 6)

    def forward(self, x):
        t = self.embed(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
        if self.pos is not None:
            t = t + self.pos
        return self.fc(self.norm(self.enc(t))[:, 0])


class TinyUNet(nn.Module):
    """작은 U-Net(분할용): 인코더 32→16→8, 디코더에서 업샘플 + 같은 해상도 특징을 이어 붙임(스킵 연결)"""
    def __init__(self, w=16, skip=True):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.skip = skip
        self.e1, self.e2, self.e3 = block(4, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.d2 = block(4 * w if skip else 2 * w, 2 * w)
        self.u1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.d1 = block(2 * w if skip else w, w)
        self.out = nn.Conv2d(w, 6, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.u2(e3)
        d2 = self.d2(torch.cat([d2, e2], 1) if self.skip else d2)
        d1 = self.u1(d2)
        d1 = self.d1(torch.cat([d1, e1], 1) if self.skip else d1)
        return self.out(d1)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# ------------------------------------------------------------------ 학습
def make_opt(name, params, lr, wd=0.0, momentum=0.9):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, x, y, seg=False, bs=300):
    model.eval()
    logits = torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])
    loss = F.cross_entropy(logits, y).item()
    pred = logits.argmax(1)
    return loss, (pred == y).float().mean().item(), pred


def miou(pred, y, k=6):
    ious = []
    for c in range(k):
        inter = ((pred == c) & (y == c)).sum().item()
        union = ((pred == c) | (y == c)).sum().item()
        ious.append(inter / union if union else float("nan"))
    return float(np.nanmean(ious)), ious


def train(model, opt="adam", lr=1e-3, wd=0.0, epochs=30, bs=64, seed=0, label_smoothing=0.0,
          n_train=None, clip=None, seg=False, log_grad=False, D=None):
    """에폭마다 학습·검증 손실과 정확도를 기록. 마지막 에폭 모델과 검증 손실이 가장 낮았던 에폭의 모델을 모두 평가"""
    import os
    if os.environ.get("PART5_SMOKE"):
        epochs, n_train = 1, 300
    torch.manual_seed(seed)
    D = D or data()
    xtr, ytr = D["x_train"], (D["m_train"] if seg else D["y_train"])
    if n_train:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(xtr), generator=g)[:n_train]
        xtr, ytr = xtr[idx], ytr[idx]
    yv = D["m_val"] if seg else D["y_val"]
    o = make_opt(opt, model.parameters(), lr, wd)
    g = torch.Generator().manual_seed(seed)
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "grad_norm", "sec"]}
    best = (float("inf"), None, -1)
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(xtr), generator=g)
        tl, tc, tn, gn = 0.0, 0, 0, []
        for i in range(0, len(xtr), bs):
            b = perm[i:i + bs]
            out = model(xtr[b])
            loss = F.cross_entropy(out, ytr[b], label_smoothing=label_smoothing)
            o.zero_grad()
            loss.backward()
            if log_grad:   # 자르기 전 전체 기울기 크기(L2 노름)
                gn.append(float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))))
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            o.step()
            if not torch.isfinite(loss):
                tl = float("nan")
                break
            tl += loss.item() * len(b)
            tc += (out.argmax(1) == ytr[b]).sum().item()
            tn += ytr[b].numel()
        vl, va, _ = evaluate(model, D["x_val"], yv)
        hist["train_loss"].append(round(tl / len(xtr), 4) if tl == tl else float("nan"))
        hist["train_acc"].append(round(tc / max(tn, 1), 4))
        hist["val_loss"].append(round(vl, 4))
        hist["val_acc"].append(round(va, 4))
        hist["grad_norm"].append(round(float(np.mean(gn)), 4) if gn else None)
        hist["sec"].append(round(time.time() - t0, 2))
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in model.state_dict().items()}, ep + 1)
        if tl != tl:
            break
    res = {"hist": hist, "params": n_params(model)}
    yt, ys = (D["m_test"], D["m_shift"]) if seg else (D["y_test"], D["y_shift"])
    for tag, state in [("last", None), ("best", best[1])]:
        if state is not None:
            model.load_state_dict(state)
        _, ta, pt = evaluate(model, D["x_test"], yt)
        _, sa, ps = evaluate(model, D["x_shift"], ys)
        r = {"test_acc": round(ta, 4), "shift_acc": round(sa, 4)}
        if seg:
            r["test_miou"] = round(miou(pt, yt)[0], 4)
            r["test_iou_per_class"] = [round(v, 4) for v in miou(pt, yt)[1]]
        else:
            cm = np.zeros((6, 6), int)
            for a, b in zip(yt.numpy(), pt.numpy()):
                cm[a, b] += 1
            r["test_cm"] = cm.tolist()
        res[tag] = r
    res["best_epoch"] = best[2]
    return res
