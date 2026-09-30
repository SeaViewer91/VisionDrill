"""5부(딥러닝) 공통 예제 자료: 가상 Sentinel-2 10 m 타일 분류(겸 분할) 자료.

실제 지역·영상 자료가 아님. 파일로 저장하지 않고 이 스크립트가 매번 똑같이 만들어 냄(시드 고정, 몇 초).
    from make_part5 import load
    D = load()   # dict

- 타일: 32×32 화소(10 m → 320 m × 320 m), 밴드 4개 B2(청) B3(녹) B4(적) B8(근적외), 지표 반사율 float32
- 클래스 6개(3부와 같음): 0 산림, 1 농경지, 2 시가지, 3 수계, 4 나지, 5 갯벌
- 라벨: 타일 라벨 y(타일에서 가장 넓은 클래스), 화소 라벨 mask(32×32, 분할 예제용)
- 약 40%의 타일은 다른 클래스가 섞여 있음(경계 타일). 섞인 면적은 최대 49%. 약 10%에는 얇은 구름 조각이 있음
- 지역(region) 30곳에서 뽑음. 지역마다 촬영 조건(밴드별 이득)이 조금 다름
  분할: 지역 0~20 학습(train), 21~25 검증(val), 26~29 시험(test) → 같은 지역 타일이 두 집합에 섞이지 않음
- test_shift: 시험(test) 타일과 똑같은 타일에(같은 시드) 옅은 연무(청·녹 밴드에 더해지는 산란광)와
  다른 계절(식생 근적외 반사 감소)을 넣은 타일. 분포 이동 예제용
- 헷갈리게 만든 곳: 시가지↔나지(둘 다 밝음, 무늬로 구분), 휴경 농경지↔나지, 갯벌↔수계(물에 잠긴 갯벌),
  산림↔농경지(생육 좋은 작물)

반환 dict: x_train (N,4,32,32) float32, y_train (N,), m_train (N,32,32) uint8, r_train (N,) 지역 번호,
           x_val, y_val, m_val, r_val, x_test, y_test, m_test, r_test, x_shift, y_shift, m_shift, r_shift,
           classes(이름 목록), bands
"""
import os
import numpy as np
from scipy import ndimage as ndi

CLASSES = ["산림", "농경지", "시가지", "수계", "나지", "갯벌"]
BANDS = ["B2", "B3", "B4", "B8"]
S = 32
MEAN = np.array([  # B2 B3 B4 B8 (3부 클래스 평균과 같은 값)
    [0.030, 0.050, 0.030, 0.330],   # 산림
    [0.050, 0.080, 0.065, 0.280],   # 농경지
    [0.095, 0.105, 0.115, 0.210],   # 시가지
    [0.060, 0.050, 0.030, 0.020],   # 수계
    [0.110, 0.130, 0.160, 0.240],   # 나지
    [0.070, 0.075, 0.075, 0.100],   # 갯벌
], np.float32)
N_PER = {"train": 700, "val": 150, "test": 150, "shift": 150}   # 클래스당 타일 수
REGIONS = {"train": range(0, 21), "val": range(21, 26), "test": range(26, 30), "shift": range(26, 30)}


def _smooth(rng, sigma, shape=(S, S)):
    z = ndi.gaussian_filter(rng.normal(0, 1, shape), sigma, mode="wrap")
    return (z - z.mean()) / (z.std() + 1e-8)


def _texture(rng, c):
    """클래스 c의 화소별 반사율 (4,S,S). 클래스마다 공간 무늬가 다름"""
    m = MEAN[c].copy() * np.exp(rng.normal(0, 0.20, 4)).astype(np.float32)   # 타일마다 상태 차이
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    if c == 0:    # 산림: 수관 얼룩(작은 덩어리)과 그림자
        tex = 0.18 * _smooth(rng, 1.2)
        img = m[:, None, None] * np.exp(tex)[None]
    elif c == 1:  # 농경지: 방향이 있는 직사각 필지, 필지마다 작물 상태가 다름(일부 휴경 → 나지에 가까움)
        th = rng.uniform(0, np.pi)
        u = xx * np.cos(th) + yy * np.sin(th)
        v = -xx * np.sin(th) + yy * np.cos(th)
        w1, w2 = rng.uniform(6, 12), rng.uniform(8, 16)
        pid = (np.floor(u / w1) * 101 + np.floor(v / w2)).astype(int)
        uniq, inv = np.unique(pid, return_inverse=True)
        state = rng.uniform(0, 1, len(uniq))
        fallow = rng.random(len(uniq)) < (0.45 if rng.random() < 0.3 else 0.12)
        img = np.zeros((4, S, S), np.float32)
        for k in range(len(uniq)):
            sel = (inv.reshape(S, S) == k)
            if fallow[k]:
                spec = 0.55 * MEAN[4] + 0.45 * m
            else:
                spec = m * np.array([1, 1, 1, 0.7 + 0.5 * state[k]], np.float32)
            img[:, sel] = spec[:, None]
        edge = (np.abs(np.sin(np.pi * u / w1)) < 0.12) | (np.abs(np.sin(np.pi * v / w2)) < 0.12)   # 논둑·농로
        img[:, edge] = (0.5 * img[:, edge] + 0.5 * MEAN[4][:, None])
        img *= np.exp(0.05 * _smooth(rng, 0.8))[None]
    elif c == 2:  # 시가지: 격자 도로 + 밝고 어두운 지붕 + 일부 가로수
        img = np.empty((4, S, S), np.float32)
        img[:] = m[:, None, None]
        step = int(rng.integers(6, 10))
        off = rng.integers(0, step, 2)
        for i in range(-1, S // step + 1):
            for j in range(-1, S // step + 1):
                y0, x0 = i * step + off[0] + 1, j * step + off[1] + 1
                roof = m * rng.choice([0.6, 0.9, 1.3, 1.6])
                img[:, max(y0, 0):max(min(y0 + step - 2, S), 0), max(x0, 0):max(min(x0 + step - 2, S), 0)] = roof[:, None, None]
        tree = _smooth(rng, 1.5) > 1.3
        img[:, tree] = MEAN[0][:, None] * 0.8 + img[:, tree] * 0.2
        img *= np.exp(0.06 * _smooth(rng, 0.6))[None]
    elif c == 3:  # 수계: 매끈함, 탁도에 따른 완만한 기울기
        g = 0.25 * (xx / S - 0.5) * rng.normal(0, 1) + 0.08 * _smooth(rng, 4)
        img = m[:, None, None] * np.exp(g)[None]
    elif c == 4:  # 나지: 밝고 거친 무늬, 차량 궤적 한두 줄
        img = m[:, None, None] * np.exp(0.14 * _smooth(rng, 0.9) + 0.10 * _smooth(rng, 3))[None]
        for _ in range(int(rng.integers(0, 3))):
            th, r0 = rng.uniform(0, np.pi), rng.uniform(-10, 10)
            d = np.abs((xx - S / 2) * np.sin(th) - (yy - S / 2) * np.cos(th) - r0)
            img[:, d < 0.8] *= 1.18
    else:         # 갯벌: 나뭇가지 모양 조수로(물), 젖은 곳은 어두움
        img = m[:, None, None] * np.exp(0.10 * _smooth(rng, 2.5))[None]
        ch = np.abs(_smooth(rng, 2.2)) < 0.12
        ch = ndi.binary_dilation(ch, iterations=int(rng.integers(0, 2)))
        img[:, ch] = (0.3 * img[:, ch] + 0.7 * MEAN[3][:, None])
    return img.astype(np.float32)


def _tile(rng, c, gain):
    """주 클래스 c 타일 하나: (반사율 4×S×S, 화소 라벨 S×S)"""
    img = _texture(rng, c)
    mask = np.full((S, S), c, np.uint8)
    if rng.random() < 0.4:                            # 경계 타일: 다른 클래스가 한쪽에서 들어옴
        # 실제로 자주 붙어 있는 클래스를 이웃으로 고름
        NEI = {0: [1, 4], 1: [0, 2, 4], 2: [1, 4, 0], 3: [5, 1, 2], 4: [2, 1], 5: [3]}
        c2 = int(rng.choice(NEI[c]))
        frac = rng.uniform(0.1, 0.49)
        th = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        proj = (xx - S / 2) * np.cos(th) + (yy - S / 2) * np.sin(th) + 1.5 * _smooth(rng, 3)
        sel = proj > np.quantile(proj, 1 - frac)
        img2 = _texture(rng, c2)
        img[:, sel] = img2[:, sel]
        mask[sel] = c2
    if c == 5 and rng.random() < 0.35:                # 물에 잠긴 갯벌: 수계와 헷갈림
        img = 0.55 * img + 0.45 * MEAN[3][:, None, None]
    img = img * gain[:, None, None]
    if rng.random() < 0.10:                           # 얇은 구름 조각
        cloud = np.clip(_smooth(rng, 3) - 0.6, 0, None)
        img = img + 0.12 * cloud[None] / (cloud.max() + 1e-8)
    img = img + rng.normal(0, 0.008, img.shape)       # 센서 잡음
    return np.clip(img, 0.001, 0.8).astype(np.float32), mask


def _split(name, seed):
    rng = np.random.default_rng(seed)
    regs = list(REGIONS[name])
    gains = {r: np.exp(np.random.default_rng(1000 + r).normal(0, 0.12, 4)).astype(np.float32) for r in regs}
    X, Y, M, R = [], [], [], []
    for c in range(len(CLASSES)):
        for _ in range(N_PER[name]):
            r = int(rng.choice(regs))
            x, m = _tile(rng, c, gains[r])
            if name == "shift":
                x = x.copy()
                veg = (m == 0) | (m == 1)
                x[3][veg] *= 0.78                          # 다른 계절: 식생 근적외 반사 감소
                x += np.array([0.035, 0.022, 0.010, 0.004], np.float32)[:, None, None]  # 연무 산란광
            X.append(x); Y.append(np.bincount(m.ravel(), minlength=6).argmax()); M.append(m); R.append(r)
    idx = rng.permutation(len(X))
    return (np.stack(X)[idx], np.array(Y, np.int64)[idx], np.stack(M)[idx], np.array(R)[idx])


_CACHE = None


def load():
    global _CACHE
    if _CACHE is None:
        d = {"classes": CLASSES, "bands": BANDS}
        for name, seed in [("train", 51), ("val", 52), ("test", 53), ("shift", 53)]:
            d[f"x_{name}"], d[f"y_{name}"], d[f"m_{name}"], d[f"r_{name}"] = _split(name, seed)
        _CACHE = d
    return _CACHE


if __name__ == "__main__":
    D = load()
    for k in ["train", "val", "test", "shift"]:
        x, y = D[f"x_{k}"], D[f"y_{k}"]
        print(k, x.shape, np.bincount(y, minlength=6), "mixed tiles",
              round(float(np.mean([len(np.unique(m)) > 1 for m in D[f'm_{k}']])), 3))
