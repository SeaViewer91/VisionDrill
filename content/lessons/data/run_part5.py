"""5부 본문 수치를 만드는 실험 모음. 결과는 part5_runs.json에 모음(실험 이름 → 결과).

    python3 run_part5.py            # 전체 (CPU 1스레드로 30분 안팎)
    python3 run_part5.py base opt   # 일부만 (이미 있는 다른 결과는 유지)

실험 이름과 쓰는 강
- base    : 기본 모델 VD-CNN(TinyCNN, 배치 정규화) Adam 1e-3, 배치 64, 30에폭 (27~33강 공통 기준)
- mlp     : 밴드 평균 4개 입력 MLP(은닉층 0·1·2개), 화소 펼침 MLP (27강)
- softmax : 기본 모델의 시험 타일 로짓·확률 예시 (27강)
- opt     : SGD·모멘텀·Adam·AdamW 비교, 너무 큰 학습률 (28강)
- grad    : 깊은 망(20층)의 층별 기울기 크기: 시그모이드·ReLU·잔차 (28·31강)
- clip    : 정규화 없는 망 + 큰 학습률에서 기울기 클리핑 유무 (28강)
- small   : 학습 타일 600개로 줄였을 때 과적합과 규제(드롭아웃·가중치 감쇠·라벨 스무딩·배치 정규화) (29강)
- reg     : 화소 펼침 MLP의 과적합과 규제(드롭아웃·가중치 감쇠·라벨 스무딩), 조기 종료 (29강)
- mode    : 학습 모드로 추론하면 생기는 일(배치 정규화) (29강)
- smallbs : 배치 4에서 배치 정규화 vs 그룹 정규화 (29강)
- dw      : 깊이별 분리 합성곱 판 VD-CNN (30강)
- deep    : 20층 평범한 망 vs 잔차 연결 망 학습 곡선 (31강)
- tv      : torchvision 백본·탐지·분할 모델 파라미터 수 (31·33강, 가중치 내려받지 않음)
- vit     : 작은 ViT vs VD-CNN, 전체 자료와 1,000개 자료 (32강)
- unet    : 작은 U-Net 분할, 스킵 연결 유무 (33강)
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from part5_common import (data, train, evaluate, n_params, TinyCNN, MeanMLP, FlatMLP, DeepNet, TinyViT,
                          TinyUNet, miou)

OUT = os.environ.get("PART5_OUT") or os.path.join(os.path.dirname(__file__), "part5_runs.json")
R = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
D = data()


def save():
    json.dump(R, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def base_model():
    torch.manual_seed(0)
    return TinyCNN()


def exp_base():
    m = base_model()
    R["base"] = train(m, "adam", 1e-3, epochs=30)
    # train()이 끝나면 모델에 best(검증 손실 최소 에폭) 가중치가 들어 있음 → softmax·mode 실험은 best 모델 기준
    torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), ".part5_base.pt"))   # git 제외


def exp_mlp():
    out = {}
    for name, hidden in [("h0", ()), ("h32", (32,)), ("h32x32", (32, 32))]:
        torch.manual_seed(0)
        out[f"mean_{name}"] = train(MeanMLP(hidden), "adam", 1e-2, epochs=30)
    torch.manual_seed(0)
    out["flat"] = train(FlatMLP(), "adam", 1e-3, epochs=30)
    R["mlp"] = out


def exp_softmax():
    m = base_model()
    m.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), ".part5_base.pt")))
    m.eval()
    with torch.no_grad():
        logits = m(D["x_test"])
    p = logits.softmax(1)
    conf, pred = p.max(1)
    y = D["y_test"]
    mixed = torch.tensor([len(torch.unique(mm)) > 1 for mm in D["m_test"]])
    ex = {}
    # 확신하는 맞은 예, 두 클래스 사이에서 망설인 예, 확신하며 틀린 예
    ok = torch.where((pred == y) & (y == 2))[0]
    i_conf = int(ok[conf[ok].argmax()]) if len(ok) else 0
    top2 = p.topk(2, 1).values
    amb = torch.where((top2[:, 0] - top2[:, 1] < 0.15) & mixed)[0]
    i_amb = int(amb[0]) if len(amb) else int((top2[:, 0] - top2[:, 1]).argmin())
    wrong = torch.where((pred != y) & (conf > 0.9))[0]
    i_wrong = int(wrong[0]) if len(wrong) else -1
    for tag, i in [("confident", i_conf), ("ambiguous", i_amb), ("wrong_confident", i_wrong)]:
        if i < 0:
            continue
        ex[tag] = {"index": i, "label": int(y[i]), "mask_frac": np.bincount(D["m_test"][i].ravel().numpy(), minlength=6).round(0).tolist(),
                   "logits": [round(float(v), 3) for v in logits[i]], "probs": [round(float(v), 4) for v in p[i]]}
    ex["mean_conf_correct"] = round(float(conf[pred == y].mean()), 4)
    ex["mean_conf_wrong"] = round(float(conf[pred != y].mean()), 4)
    ex["n_wrong_conf_gt_0.9"] = int(len(wrong))
    R["softmax"] = ex


def exp_opt():
    out = {}
    for name, opt, lr, wd in [("sgd_0.01", "sgd", 0.01, 0), ("momentum_0.01", "momentum", 0.01, 0),
                              ("adam_1e-3", "adam", 1e-3, 0), ("adamw_1e-3_wd0.05", "adamw", 1e-3, 0.05),
                              ("adam_0.1", "adam", 0.1, 0), ("sgd_0.1", "sgd", 0.1, 0)]:
        torch.manual_seed(0)
        out[name] = train(TinyCNN(), opt, lr, wd, epochs=30)
    R["opt"] = out


def layer_grads(model, n=256):
    """첫 배치 하나에서 합성곱 층별 가중치 기울기 크기(L2 노름)"""
    model.train()
    x, y = D["x_train"][:n], D["y_train"][:n]
    loss = F.cross_entropy(model(x), y)
    model.zero_grad()
    loss.backward()
    return [float(f"{float(c.weight.grad.norm()):.3e}") for c in [model.stem, *model.convs]]


def exp_grad():
    out = {}
    for name, act, res in [("sigmoid", "sigmoid", False), ("relu", "relu", False), ("relu_residual", "relu", True)]:
        torch.manual_seed(0)
        out[name] = layer_grads(DeepNet(20, res, act))
    R["grad"] = out


def exp_clip():
    out = {}
    for name, clip in [("noclip", None), ("clip1", 1.0)]:
        torch.manual_seed(0)
        out[name] = train(TinyCNN(norm="none"), "momentum", 0.1, epochs=15, clip=clip, log_grad=True)
    R["clip"] = out


def exp_small():
    out = {}
    cfgs = [("bn", dict(norm="bn"), {}), ("none", dict(norm="none"), {}),
            ("bn_dropout0.3", dict(norm="bn", dropout=0.3), {}),
            ("bn_adamw_wd0.05", dict(norm="bn"), dict(opt="adamw", wd=0.05)),
            ("bn_ls0.1", dict(norm="bn"), dict(label_smoothing=0.1))]
    for name, mk, tk in cfgs:
        torch.manual_seed(0)
        kw = dict(opt="adam", lr=1e-3, epochs=80, n_train=600)
        kw.update(tk)
        out[name] = train(TinyCNN(**mk), **kw)
    R["small"] = out


def exp_reg():
    """과적합이 뚜렷한 화소 펼침 MLP(54만 파라미터)에 규제를 하나씩 넣어 비교 (40에폭)"""
    out = {}
    for name, mk, tk in [("flat", {}, {}), ("flat_dropout0.5", dict(dropout=0.5), {}),
                         ("flat_adamw_wd0.1", {}, dict(opt="adamw", wd=0.1)),
                         ("flat_adamw_wd1.0", {}, dict(opt="adamw", wd=1.0)),
                         ("flat_ls0.1", {}, dict(label_smoothing=0.1))]:
        torch.manual_seed(0)
        kw = dict(opt="adam", lr=1e-3, epochs=40)
        kw.update(tk)
        out[name] = train(FlatMLP(**mk), **kw)
    R["reg"] = out


def exp_mode():
    m = base_model()
    m.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), ".part5_base.pt")))
    x, y = D["x_test"], D["y_test"]
    out = {}
    m.eval()
    out["eval_mode"] = round(evaluate(m, x, y)[1], 4)
    # 학습 모드(배치 통계 사용)로 한 클래스만 모인 배치를 추론
    with torch.no_grad():
        m.train()
        correct = 0
        for c in range(6):
            idx = torch.where(y == c)[0]
            for i in range(0, len(idx), 30):
                b = idx[i:i + 30]
                correct += (m(x[b]).argmax(1) == y[b]).sum().item()
        out["train_mode_single_class_batches"] = round(correct / len(y), 4)
        m.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), ".part5_base.pt")))  # 러닝 통계 복원
        m.train()
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(y), generator=g)
        correct = 0
        for i in range(0, len(y), 64):
            b = perm[i:i + 64]
            correct += (m(x[b]).argmax(1) == y[b]).sum().item()
        out["train_mode_mixed_batches64"] = round(correct / len(y), 4)
    R["mode"] = out


def exp_smallbs():
    out = {}
    for name, norm in [("bn_bs4", "bn"), ("gn_bs4", "gn"), ("bn_bs64", "bn"), ("gn_bs64", "gn")]:
        torch.manual_seed(0)
        bs = 4 if "bs4" in name else 64
        out[name] = train(TinyCNN(norm=norm), "adam", 1e-3 if bs == 64 else 2.5e-4, epochs=10, bs=bs)
    R["smallbs"] = out


class DWCNN(nn.Module):
    """VD-CNN의 2·3번째 3×3 합성곱을 깊이별(3×3, 채널마다 따로) + 점별(1×1) 합성곱으로 바꾼 판"""
    def __init__(self):
        super().__init__()
        def dw(i, o):
            return nn.Sequential(nn.Conv2d(i, i, 3, padding=1, groups=i), nn.Conv2d(i, o, 1))
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            dw(16, 32), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            dw(32, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 6))

    def forward(self, x):
        return self.head(self.features(x))


def exp_dw():
    torch.manual_seed(0)
    m = DWCNN()
    r = train(m, "adam", 1e-3, epochs=30)
    # 층별 파라미터 수와 곱셈-누산(MAC) 수
    def macs(model):
        tot, hooks = [0], []
        def hk(mod, inp, outp):
            if isinstance(mod, nn.Conv2d):
                tot[0] += outp.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
            elif isinstance(mod, nn.Linear):
                tot[0] += mod.in_features * mod.out_features
        for mod in model.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                hooks.append(mod.register_forward_hook(hk))
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 4, 32, 32))
        for h in hooks:
            h.remove()
        return tot[0]
    torch.manual_seed(0)
    r["macs"] = macs(m)
    r["base_macs"] = macs(TinyCNN())
    r["base_params"] = n_params(TinyCNN())
    R["dw"] = r


def exp_deep():
    out = {}
    for name, depth, res in [("plain8", 8, False), ("plain20", 20, False), ("res20", 20, True)]:
        torch.manual_seed(0)
        out[name] = train(DeepNet(depth, res), "adam", 1e-3, epochs=20)
    R["deep"] = out


def exp_tv():
    import torchvision.models as M
    import torchvision.models.detection as Dm
    import torchvision.models.segmentation as S
    out = {}
    for name in ["vgg16", "resnet18", "resnet50", "densenet121", "mobilenet_v2", "mobilenet_v3_large",
                 "efficientnet_b0", "efficientnet_b3", "convnext_tiny", "vit_b_16", "swin_t"]:
        m = getattr(M, name)(weights=None)
        out[name] = n_params(m)
    for name, fn in [("fasterrcnn_resnet50_fpn", Dm.fasterrcnn_resnet50_fpn), ("maskrcnn_resnet50_fpn", Dm.maskrcnn_resnet50_fpn),
                     ("retinanet_resnet50_fpn", Dm.retinanet_resnet50_fpn), ("fcos_resnet50_fpn", Dm.fcos_resnet50_fpn)]:
        out[name] = n_params(fn(weights=None, weights_backbone=None, num_classes=91))
    out["deeplabv3_resnet50"] = n_params(S.deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=21, aux_loss=False))
    out["_note"] = "torchvision 0.29 기준, 가중치 없이 구조만 만들어 센 값. 탐지는 COCO 91클래스, 분할은 21클래스 헤드"
    R["tv"] = out


def exp_vit():
    out = {}
    torch.manual_seed(0)
    out["vit_full"] = train(TinyViT(), "adamw", 1e-3, 0.05, epochs=30)
    torch.manual_seed(0)
    out["vit_full_nopos"] = train(TinyViT(pos=False), "adamw", 1e-3, 0.05, epochs=30)
    torch.manual_seed(0)
    out["vit_n1000"] = train(TinyViT(), "adamw", 1e-3, 0.05, epochs=60, n_train=1000)
    torch.manual_seed(0)
    out["cnn_n1000"] = train(TinyCNN(), "adam", 1e-3, epochs=60, n_train=1000)
    R["vit"] = out


def exp_unet():
    """분할: 스킵 연결 유무. 경계 정확도 = 참 라벨 경계에서 1화소 안쪽 화소만 모은 정확도(best 모델 기준)"""
    from scipy import ndimage as ndi
    m_test = D["m_test"].numpy()
    edge = np.zeros_like(m_test, bool)
    for i, mm in enumerate(m_test):
        e = (mm != ndi.maximum_filter(mm, 3)) | (mm != ndi.minimum_filter(mm, 3))
        edge[i] = e
    edge_t = torch.tensor(edge)
    out = {}
    for name, skip in [("skip", True), ("noskip", False)]:
        torch.manual_seed(0)
        m = TinyUNet(skip=skip)
        r = train(m, "adam", 1e-3, epochs=20, seg=True)
        _, _, pred = evaluate(m, D["x_test"], D["m_test"])            # best 가중치가 들어 있음
        r["best"]["edge_acc"] = round(float((pred == D["m_test"])[edge_t].float().mean()), 4)
        r["best"]["interior_acc"] = round(float((pred == D["m_test"])[~edge_t].float().mean()), 4)
        r["edge_pixel_frac"] = round(float(edge.mean()), 4)
        torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), f".part5_unet_{name}.pt"))  # git 제외
        out[name] = r
    R["unet"] = out


EXPS = {k[4:]: v for k, v in globals().items() if k.startswith("exp_")}

if __name__ == "__main__":
    names = sys.argv[1:] or list(EXPS)
    for n in names:
        t = time.time()
        EXPS[n]()
        R.setdefault("_meta", {})[n] = {"sec": round(time.time() - t, 1), "torch": torch.__version__}
        save()
        print(n, "done", round(time.time() - t, 1), "s", flush=True)
