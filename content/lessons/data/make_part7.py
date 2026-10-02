"""7부(학습 실무) 공통 예제 자료: 가상 Sentinel-2 10 m 장면 10장(256×256 화소 = 2.56 km 사방)과 그 타일.

실제 지역·영상 자료가 아님. 파일로 저장하지 않고 시드 고정으로 다시 만듦(몇 초).
5부 타일 자료와 같은 클래스·밴드·무늬 생성기(make_part5)를 써서, 5부 모델(VD-CNN, U-Net)을 그대로 쓸 수 있음.

- 장면: 256×256 화소, 밴드 4개(B2 B3 B4 B8, 반사율), 화소 라벨(클래스 6개: 산림 농경지 시가지 수계 나지 갯벌)
- 장면마다 토지피복 배치(덩어리 지도)와 촬영 조건(밴드별 이득)이 다름. 같은 장면 안의 이웃 화소는 서로 닮음(공간 자기상관)
- 장면 0~7: 개발용(학습·검증으로 나눠 씀), 장면 8~9: 끝까지 손대지 않는 시험용
- 지오트랜스폼: 장면 k의 왼쪽 위 모서리 = (동 352,000 + 3,000·k m, 북 3,890,000 m), 화소 10 m, EPSG:32652(UTM 52N) — 가상 좌표

    from make_part7 import scenes, tile_scene, GEOTRANSFORM
    S = scenes()                 # list of dict(x (4,256,256) float32, m (256,256) uint8, k)
    T = tile_scene(S[0], size=32, stride=16)   # dict(x (N,4,32,32), y (N,), m (N,32,32), r0, c0)
"""
import numpy as np
from scipy import ndimage as ndi

import make_part5 as P5

N_SCENES = 10
SIZE = 256
PIX = 10.0


def GEOTRANSFORM(k):
    """GDAL 순서 (x0, 화소 폭, 0, y0, 0, -화소 높이)"""
    return (352000.0 + 3000.0 * k, PIX, 0.0, 3890000.0, 0.0, -PIX)


def _class_map(rng):
    """덩어리 지도: 클래스마다 부드러운 잡음장을 만들고 가장 큰 값의 클래스를 고름 + 해안(왼쪽 물·갯벌)"""
    f = np.stack([ndi.gaussian_filter(rng.normal(0, 1, (SIZE, SIZE)), rng.uniform(10, 18)) for _ in range(6)])
    f /= f.std(axis=(1, 2), keepdims=True)
    f += rng.normal(0, 0.3, (6, 1, 1))
    m = f.argmax(0).astype(np.uint8)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    coast = rng.uniform(20, 70) + 15 * np.sin(yy / rng.uniform(20, 40))
    m[xx < coast] = 3                                   # 바다
    m[(xx >= coast) & (xx < coast + rng.uniform(8, 20))] = 5   # 갯벌 띠
    m = ndi.median_filter(m, 5)
    return m


def _scene(k):
    rng = np.random.default_rng(700 + k)
    m = _class_map(rng)
    gain = np.exp(rng.normal(0, 0.12, 4)).astype(np.float32)
    # 덩어리(같은 클래스의 연결 영역)마다 평균 반사율 하나: 같은 필지·숲은 블록이 달라도 밝기가 이어지게
    pm = np.zeros((4, SIZE, SIZE), np.float32)
    for cl in range(6):
        lab, n = ndi.label(m == cl)
        for i in range(1, n + 1):
            pm[:, lab == i] = (P5.MEAN[cl] * np.exp(rng.normal(0, 0.15, 4)))[:, None]
    x = np.zeros((4, SIZE, SIZE), np.float32)
    B = P5.S                                            # 32화소 블록마다 클래스별 무늬를 만들어 덩어리 평균에 맞춰 채움
    for r in range(0, SIZE, B):
        for c in range(0, SIZE, B):
            blk = m[r:r + B, c:c + B]
            for cl in np.unique(blk):
                tex = P5._texture(rng, int(cl))
                tex = tex / tex.mean(axis=(1, 2), keepdims=True)
                sel = blk == cl
                x[:, r:r + B, c:c + B][:, sel] = tex[:, sel] * pm[:, r:r + B, c:c + B][:, sel]
    x = ndi.gaussian_filter(x, (0, 0.6, 0.6))           # 블록 경계 이음매를 조금 누그러뜨림
    x = x * gain[:, None, None] + rng.normal(0, 0.008, x.shape)
    return dict(k=k, x=np.clip(x, 0.001, 0.8).astype(np.float32), m=m)


_CACHE = None


def scenes():
    global _CACHE
    if _CACHE is None:
        _CACHE = [_scene(k) for k in range(N_SCENES)]
    return _CACHE


def tile_scene(sc, size=32, stride=32, r_range=None, c_range=None):
    """장면을 size×size 타일로 자름. 타일 라벨 = 가장 넓은 클래스. r_range/c_range로 자를 영역 제한(시작점 기준)"""
    X, Y, M, R, C = [], [], [], [], []
    H = sc["m"].shape[0]
    for r in range(0, H - size + 1, stride):
        if r_range and not (r_range[0] <= r and r + size <= r_range[1]):
            continue
        for c in range(0, H - size + 1, stride):
            if c_range and not (c_range[0] <= c and c + size <= c_range[1]):
                continue
            m = sc["m"][r:r + size, c:c + size]
            X.append(sc["x"][:, r:r + size, c:c + size]); M.append(m)
            Y.append(np.bincount(m.ravel(), minlength=6).argmax()); R.append(r); C.append(c)
    return dict(x=np.stack(X), y=np.array(Y, np.int64), m=np.stack(M), r0=np.array(R), c0=np.array(C))


if __name__ == "__main__":
    S = scenes()
    for s in S:
        print(s["k"], np.bincount(s["m"].ravel(), minlength=6).round(-2).tolist())
    T = tile_scene(S[0], 32, 16)
    print("tiles stride16 per scene", len(T["y"]))
