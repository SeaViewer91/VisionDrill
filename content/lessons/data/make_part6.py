"""6부(태스크와 평가) 공통 예제 자료: 가상 항만·연안 고해상도 영상(0.5 m)의 객체 탐지 정답과 '가상 탐지기' 출력.

실제 지역·영상·모델이 아님. 파일로 저장하지 않고 시드 고정으로 매번 똑같이 만듦(몇 초).
    from make_part6 import tiles, big_scene, sequence, detect

좌표 규약
- 영상 좌표: x 오른쪽, y 아래, 원점 왼쪽 위, 단위 화소(0.5 m)
- 회전 박스(OBB): (cx, cy, w, h, theta). w = 긴 변(길이), h = 짧은 변(폭), theta = 긴 변이 x축과 이루는 각(도).
  y가 아래로 커지므로 theta가 +이면 화면에서 시계 방향. 범위 [-90, 90) (긴 변 기준 90° 규약, 41강)
- 수평 박스(HBB): (x1, y1, x2, y2) = 회전 박스 네 꼭짓점을 감싸는 최소 직사각형
- 넓이(area): COCO 크기 구간(소·중·대)을 나눌 때 쓰는 값. 여기서는 HBB 넓이(w×h)로 둠

클래스: 0 선박(길이 40~140 m), 1 소형선박(6~20 m), 2 차량(4~5 m)

1) tiles(): 512×512 화소(256 m) 타일 24장. 항만 8장(부두·주차장·계류 선박), 외해 10장(흩어진 배·흰 물결),
   마리나 6장(작은 배가 촘촘). 일부 주차장은 45° 기울어져 있어 이웃 차량의 HBB가 많이 겹침
2) detect(objs, clutter, scale, seed): 가상 탐지기. 모델 입력에서 보이는 크기(화소)가 작을수록 놓치고,
   위치가 흔들리고, 점수가 낮음. NMS 전의 원시 출력(한 객체에 상자 여러 개)을 냄. 흰 물결·컨테이너 같은
   방해물에서 오탐이 나고, 선박↔소형선박은 크기 경계 근처에서 헷갈림
3) big_scene(): 2,048×2,048 화소(1 km) 큰 장면 한 장(슬라이스 추론 예제용, 37강)
4) sequence(): 외해 512×512에서 120프레임(0.2초 간격) 동안 움직이는 배 10척의 정답 궤적과 프레임별 탐지(42강)
"""
import numpy as np

GSD = 0.5
CLASSES = ["선박", "소형선박", "차량"]
TILE = 512


# ----------------------------------------------------------------- 기하
def obb_corners(cx, cy, w, h, th):
    t = np.deg2rad(th)
    u = np.array([np.cos(t), np.sin(t)]) * w / 2
    v = np.array([-np.sin(t), np.cos(t)]) * h / 2
    c = np.array([cx, cy])
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])


def obb_to_hbb(o):
    p = obb_corners(*o)
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])


def norm_angle(th):
    """[-90, 90)로 접기"""
    return (th + 90.0) % 180.0 - 90.0


# ----------------------------------------------------------------- 장면 만들기
def _ship(rng, x, y, th, cls):
    if cls == 0:
        L = rng.uniform(40, 140) / GSD
        W = L * rng.uniform(0.14, 0.19)
    elif cls == 1:
        L = rng.uniform(6, 20) / GSD
        W = L * rng.uniform(0.3, 0.4)
    else:
        L = rng.uniform(4.0, 5.0) / GSD
        W = rng.uniform(1.75, 1.95) / GSD
    return dict(cls=cls, cx=float(x), cy=float(y), w=float(L), h=float(W), theta=float(norm_angle(th)))


def _overlaps(o, objs, pad=2.0):
    a = obb_to_hbb((o["cx"], o["cy"], o["w"] + pad, o["h"] + pad, o["theta"]))
    for b in objs:
        if b["cls"] == 2 and o["cls"] == 2:
            continue
        bb = obb_to_hbb((b["cx"], b["cy"], b["w"] + pad, b["h"] + pad, b["theta"]))
        if a[0] < bb[2] and bb[0] < a[2] and a[1] < bb[3] and bb[1] < a[3]:
            return True
    return False


def _place(rng, objs, cls, region, th_fn, tries=200):
    x0, y0, x1, y1 = region
    for _ in range(tries):
        o = _ship(rng, rng.uniform(x0, x1), rng.uniform(y0, y1), th_fn(), cls)
        p = obb_corners(o["cx"], o["cy"], o["w"], o["h"], o["theta"])
        if p[:, 0].min() < x0 or p[:, 0].max() > x1 or p[:, 1].min() < y0 or p[:, 1].max() > y1:
            continue
        if not _overlaps(o, objs):
            objs.append(o)
            return o
    return None


def _parking(rng, objs, x0, y0, nrow, ncol, lot_th):
    """주차장: lot_th 방향으로 늘어선 차량 줄. 차 사이 간격 2.6 m, 줄 사이 7 m"""
    t = np.deg2rad(lot_th)
    along = np.array([np.cos(t), np.sin(t)])          # 줄 방향(차가 옆으로 나란히)
    across = np.array([-np.sin(t), np.cos(t)])        # 차 길이 방향
    for r in range(nrow):
        for c in range(ncol):
            if rng.random() < 0.18:                    # 빈자리
                continue
            p = np.array([x0, y0]) + along * c * (2.6 / GSD) + across * r * (7.0 / GSD)
            p = p + rng.normal(0, 0.6, 2)
            o = _ship(rng, p[0], p[1], lot_th + 90 + rng.normal(0, 3), 2)
            objs.append(o)


def _clutter(rng, kind, region, n):
    x0, y0, x1, y1 = region
    out = []
    for _ in range(n):
        if kind == "whitecap":   # 흰 물결: 작은 배와 헷갈림
            out.append(dict(kind=kind, cx=rng.uniform(x0, x1), cy=rng.uniform(y0, y1),
                            w=rng.uniform(6, 22), h=rng.uniform(3, 7), theta=rng.uniform(-90, 90)))
        else:                    # 컨테이너·부두 구조물: 차량·소형선박과 헷갈림
            out.append(dict(kind=kind, cx=rng.uniform(x0, x1), cy=rng.uniform(y0, y1),
                            w=rng.uniform(10, 26), h=rng.uniform(4, 6), theta=rng.choice([0, 90, 45])))
    return out


def _tile(rng, kind):
    objs, clutter = [], []
    S = TILE
    if kind == "harbor":          # 왼쪽 땅(주차장·부두), 오른쪽 바다(계류 선박)
        land = S * rng.uniform(0.38, 0.5)
        lot_th = rng.choice([0, 45, 90, 45])
        _parking(rng, objs, 30, 30, nrow=int(rng.integers(5, 10)), ncol=int(rng.integers(12, 20)), lot_th=lot_th)
        _parking(rng, objs, 40, S * 0.62, nrow=int(rng.integers(3, 7)), ncol=int(rng.integers(10, 16)), lot_th=0)
        objs[:] = [o for o in objs if 4 < o["cx"] < land - 6 and 4 < o["cy"] < S - 4]
        for _ in range(int(rng.integers(1, 3))):  # 부두에 붙은 선박(부두와 나란히)
            _place(rng, objs, 0, (land + 4, 10, S - 4, S - 10), lambda: 90 + rng.normal(0, 4))
        for _ in range(int(rng.integers(3, 8))):
            _place(rng, objs, 1, (land + 4, 4, S - 4, S - 4), lambda: rng.uniform(-90, 90))
        clutter += _clutter(rng, "container", (8, 8, land - 8, S - 8), int(rng.integers(6, 14)))
        clutter += _clutter(rng, "whitecap", (land + 8, 8, S - 8, S - 8), int(rng.integers(2, 6)))
    elif kind == "sea":
        for _ in range(int(rng.integers(0, 3))):
            _place(rng, objs, 0, (4, 4, S - 4, S - 4), lambda: rng.uniform(-90, 90))
        for _ in range(int(rng.integers(3, 10))):
            _place(rng, objs, 1, (4, 4, S - 4, S - 4), lambda: rng.uniform(-90, 90))
        clutter += _clutter(rng, "whitecap", (8, 8, S - 8, S - 8), int(rng.integers(8, 25)))
    else:                         # marina: 잔교 사이 작은 배가 촘촘
        th = rng.choice([0, 90])
        for k in range(4):
            for j in range(int(rng.integers(8, 14))):
                if rng.random() < 0.25:
                    continue
                L = rng.uniform(8, 16) / GSD
                if th == 0:
                    x, y = 40 + j * 34 + rng.normal(0, 2), 70 + k * 110 + rng.normal(0, 2)
                else:
                    x, y = 70 + k * 110 + rng.normal(0, 2), 40 + j * 34 + rng.normal(0, 2)
                o = dict(cls=1, cx=float(x), cy=float(y), w=float(L), h=float(L * rng.uniform(0.3, 0.38)),
                         theta=float(norm_angle(th + 90 + rng.normal(0, 5))))
                p = obb_corners(o["cx"], o["cy"], o["w"], o["h"], o["theta"])
                if p.min() > 2 and p.max() < S - 2 and not _overlaps(o, objs, pad=0.5):
                    objs.append(o)
        clutter += _clutter(rng, "whitecap", (8, 8, S - 8, S - 8), int(rng.integers(2, 6)))
    for i, o in enumerate(objs):
        o["obb"] = [o["cx"], o["cy"], o["w"], o["h"], o["theta"]]
        o["hbb"] = obb_to_hbb((o["cx"], o["cy"], o["w"], o["h"], o["theta"])).tolist()
        hb = o["hbb"]
        o["area"] = float((hb[2] - hb[0]) * (hb[3] - hb[1]))
    return objs, clutter


def tiles():
    """타일 24장: list of dict(image_id, kind, objs, clutter)"""
    rng = np.random.default_rng(61)
    kinds = ["harbor"] * 8 + ["sea"] * 10 + ["marina"] * 6
    out = []
    for i, k in enumerate(kinds):
        objs, cl = _tile(rng, k)
        out.append(dict(image_id=i + 1, kind=k, objs=objs, clutter=cl))
    return out


# ----------------------------------------------------------------- 가상 탐지기
def _sig(x):
    return 1 / (1 + np.exp(-x))


def detect(objs, clutter, scale=1.0, seed=0, view=None):
    """가상 탐지기의 NMS 전 원시 출력.

    scale: 모델 입력 화소 / 영상 화소 (타일을 그대로 넣으면 1, 2,048 장면을 640으로 줄이면 0.3125)
    view : (x0, y0, x1, y1) 모델이 보는 창(슬라이스). 창 밖으로 잘린 객체는 보이는 부분만 상자를 냄
    반환: list of dict(cls, score, obb(5), hbb(4), src) — src는 만든 원인(객체 번호 또는 'clutter'/'bg')
    """
    rng = np.random.default_rng(seed)
    out = []
    for gi, o in enumerate(objs):
        hb = np.array(o["hbb"])
        vis = 1.0
        if view is not None:
            x0, y0, x1, y1 = view
            ix = max(0.0, min(hb[2], x1) - max(hb[0], x0))
            iy = max(0.0, min(hb[3], y1) - max(hb[1], y0))
            area = (hb[2] - hb[0]) * (hb[3] - hb[1])
            vis = ix * iy / area if area > 0 else 0
            if vis <= 0.02:
                continue
        La = o["w"] * scale * (vis ** 0.5 if vis < 1 else 1)          # 모델 입력에서 보이는 길이(화소)
        p = 0.96 * _sig((La - 7.0) / 1.6)
        if o["cls"] == 2 and abs(((o["theta"] % 90) - 45)) < 20:      # 비스듬히 빽빽한 차량은 조금 더 놓침
            p *= 0.9
        p *= vis ** 0.3
        if rng.random() > p:
            continue
        base = float(np.clip(_sig((La - 6.0) / 4.0) * rng.beta(4, 2) * (vis ** 0.5), 0.02, 0.995))
        # 클래스 혼동: 선박↔소형선박은 길이 경계(약 30~40 m) 근처에서
        Lm = o["w"] * GSD
        cls = o["cls"]
        conf = 0.03
        if cls in (0, 1) and 22 < Lm < 50:
            conf = 0.3
        if rng.random() < conf:
            cls = 1 - cls if cls in (0, 1) else 1
        k = 1 + rng.poisson(3.0 if La > 12 else 1.5)
        for j in range(k):
            sj = 0.5 if j == 0 else 1.0
            sig_c = sj * (0.04 * o["w"] + 0.7 / scale)
            cx = o["cx"] + rng.normal(0, sig_c)
            cy = o["cy"] + rng.normal(0, sig_c)
            w = o["w"] * np.exp(rng.normal(0, sj * 0.08))
            h = o["h"] * np.exp(rng.normal(0, sj * 0.10)) + rng.normal(0, 0.3 / scale)
            th = o["theta"] + rng.normal(0, sj * (2.0 + 60.0 / max(La, 3)))
            ob = np.array([cx, cy, max(w, 1.0), max(h, 1.0), norm_angle(th)])
            if ob[3] > ob[2]:                                         # 짧은 변이 길어지면 규약대로 바꿈
                ob = np.array([ob[0], ob[1], ob[3], ob[2], norm_angle(ob[4] + 90)])
            hbb = obb_to_hbb(ob)
            if view is not None:                                     # 창 밖은 볼 수 없음
                hbb = np.array([max(hbb[0], view[0]), max(hbb[1], view[1]), min(hbb[2], view[2]), min(hbb[3], view[3])])
                if hbb[2] - hbb[0] < 1 or hbb[3] - hbb[1] < 1:
                    continue
            sc = base * (1.0 if j == 0 else rng.uniform(0.55, 0.95))
            bc = cls
            if j > 0 and rng.random() < conf:                         # 같은 객체에서 다른 클래스로 낸 상자도 섞임
                bc = (1 - cls if cls in (0, 1) else 1)
                sc *= 0.7
            out.append(dict(cls=int(bc), score=round(float(sc), 4), obb=[float(v) for v in ob],
                            hbb=[float(v) for v in hbb], src=gi))
    for c in clutter:
        if view is not None and not (view[0] <= c["cx"] < view[2] and view[1] <= c["cy"] < view[3]):
            continue
        q = 0.3 if c["kind"] == "whitecap" else 0.35
        La = c["w"] * scale
        q *= _sig((La - 6) / 2)
        if rng.random() > q:
            continue
        cls = 1 if c["kind"] == "whitecap" else int(rng.choice([2, 1]))
        base = float(rng.uniform(0.15, 0.65))
        for j in range(1 + rng.poisson(1.2)):
            ob = np.array([c["cx"] + rng.normal(0, 1.5), c["cy"] + rng.normal(0, 1.5), c["w"] * np.exp(rng.normal(0, 0.1)),
                           c["h"] * np.exp(rng.normal(0, 0.1)), norm_angle(c["theta"] + rng.normal(0, 5))])
            out.append(dict(cls=cls, score=round(base * (1 if j == 0 else rng.uniform(0.5, 0.95)), 4),
                            obb=[float(v) for v in ob], hbb=[float(v) for v in obb_to_hbb(ob)], src="clutter"))
    # 배경 잡음: 아주 낮은 점수
    x0, y0, x1, y1 = view if view is not None else (0, 0, TILE, TILE)
    for _ in range(rng.poisson(1.0 * (x1 - x0) * (y1 - y0) / TILE ** 2)):
        ob = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(8, 30), rng.uniform(3, 8), rng.uniform(-90, 90)])
        out.append(dict(cls=int(rng.integers(0, 3)), score=round(float(rng.uniform(0.01, 0.2)), 4),
                        obb=[float(v) for v in ob], hbb=[float(v) for v in obb_to_hbb(ob)], src="bg"))
    return out


# ----------------------------------------------------------------- 큰 장면
def big_scene():
    """2,048×2,048 화소. 왼쪽 위 항만(차량 많음), 나머지 바다(선박·소형선박). 큰 선박 일부는 512 타일보다 김"""
    rng = np.random.default_rng(62)
    S = 2048
    objs, clutter = [], []
    for (x0, y0, nr, nc, th) in [(60, 60, 8, 30, 0), (60, 420, 6, 24, 45), (100, 760, 6, 26, 90), (520, 120, 5, 20, 0)]:
        _parking(rng, objs, x0, y0, nr, nc, th)
    objs[:] = [o for o in objs if 4 < o["cx"] < 900 and 4 < o["cy"] < 1000]
    for _ in range(6):
        _place(rng, objs, 0, (950, 40, S - 40, S - 40), lambda: rng.uniform(-90, 90))
    for _ in range(4):        # 아주 큰 선박(길이 300~400 m 급 → 600~800 화소)
        for _t in range(200):
            o = _ship(rng, rng.uniform(1100, S - 300), rng.uniform(300, S - 300), rng.uniform(-90, 90), 0)
            o["w"] = rng.uniform(300, 400) / GSD
            o["h"] = o["w"] * 0.14
            p = obb_corners(o["cx"], o["cy"], o["w"], o["h"], o["theta"])
            if p.min() > 10 and p.max() < S - 10 and not _overlaps(o, objs):
                objs.append(o)
                break
    for _ in range(60):
        _place(rng, objs, 1, (950, 20, S - 20, S - 20), lambda: rng.uniform(-90, 90))
    for _ in range(25):
        _place(rng, objs, 1, (20, 1100, 940, S - 20), lambda: rng.uniform(-90, 90))
    clutter += _clutter(rng, "whitecap", (950, 20, S - 20, S - 20), 120)
    clutter += _clutter(rng, "container", (20, 20, 900, 1000), 60)
    for o in objs:
        o["obb"] = [o["cx"], o["cy"], o["w"], o["h"], o["theta"]]
        o["hbb"] = obb_to_hbb((o["cx"], o["cy"], o["w"], o["h"], o["theta"])).tolist()
        hb = o["hbb"]
        o["area"] = float((hb[2] - hb[0]) * (hb[3] - hb[1]))
    return dict(size=S, objs=objs, clutter=clutter)


# ----------------------------------------------------------------- 추적용 시퀀스
def sequence(n_frames=120, dt=0.2):
    """외해 512×512, 0.2초 간격(5 fps) 120프레임(24초). 배 10척(선박 2, 소형 8)이 등속에 가깝게 움직이고 두 쌍은 서로 엇갈림.
    프레임 50~65에는 햇빛 반사(글린트) 구역(x 200~330)이 있어 그 안의 배는 대부분 낮은 점수(0.15~0.45)로만 잡힘.
    반환: dict(gt: list of (frame, track_id, cls, cx, cy, w, h, theta), det: list of (frame, cls, score, x1, y1, x2, y2))
    """
    rng = np.random.default_rng(63)
    boats = []
    for i in range(10):
        cls = 0 if i < 2 else 1
        L = (rng.uniform(45, 70) if cls == 0 else rng.uniform(8, 16)) / GSD
        speed = (rng.uniform(4, 6) if cls == 0 else rng.uniform(5, 9)) / GSD * dt   # 화소/프레임
        if i in (2, 4):          # 엇갈리는 쌍 1: 3번↔4번 배, 2: 5번↔6번 배 (같은 지점을 비슷한 시각에 지남)
            th = rng.uniform(-30, 30)
        elif i in (3, 5):
            th = boats[i - 1]["th"] + rng.choice([-1, 1]) * rng.uniform(25, 45)
        else:
            th = rng.uniform(-180, 180)
        boats.append(dict(cls=cls, L=L, W=L * (0.16 if cls == 0 else 0.35), v=speed, th=th))
    gt, det = [], []
    meet = {2: (250, 260, 40), 3: (250, 260, 40), 4: (340, 150, 85), 5: (340, 150, 85)}
    for i, b in enumerate(boats):
        t = np.deg2rad(b["th"])
        vx, vy = b["v"] * np.cos(t), b["v"] * np.sin(t)
        if i in meet:
            mx, my, mf = meet[i]
            b["x0"], b["y0"] = mx - vx * mf, my - vy * mf
        else:
            b["x0"], b["y0"] = rng.uniform(80, 430), rng.uniform(80, 430)
        b["vx"], b["vy"] = vx, vy
    for f in range(n_frames):
        for i, b in enumerate(boats):
            wob = 2.0 * np.sin(f * dt / 2.0 + i)                     # 약간의 흔들림(등속이 아님)
            cx = b["x0"] + b["vx"] * f + wob * -np.sin(np.deg2rad(b["th"]))
            cy = b["y0"] + b["vy"] * f + wob * np.cos(np.deg2rad(b["th"]))
            o = (cx, cy, b["L"], b["W"], norm_angle(b["th"]))
            hb = np.clip(obb_to_hbb(o), 0, TILE)
            if hb[2] - hb[0] < 3 or hb[3] - hb[1] < 3:
                continue
            gt.append((f, i + 1, b["cls"], *[float(v) for v in o]))
            glint = 50 <= f <= 65 and 200 <= cx <= 330
            p = 0.85 if glint else 0.92
            if rng.random() < p:
                j = rng.normal(0, 1.0, 4) + rng.normal(0, 0.03, 4) * (hb[2] - hb[0])
                d = np.clip(hb + j, 0, TILE)
                sc = rng.uniform(0.15, 0.45) if (glint or rng.random() < 0.05) else rng.uniform(0.55, 0.95)
                det.append((f, b["cls"], round(float(sc), 3), *[float(v) for v in d]))
        for _ in range(rng.poisson(1.0)):                            # 흰 물결 오탐(대부분 낮은 점수)
            x, y = rng.uniform(0, TILE - 20), rng.uniform(0, TILE - 20)
            w, h = rng.uniform(8, 20), rng.uniform(4, 10)
            det.append((f, 1, round(float(rng.uniform(0.1, 0.6) ** 1.5), 3), x, y, x + w, y + h))
    return dict(gt=gt, det=det, n_frames=n_frames, dt=dt)


if __name__ == "__main__":
    T = tiles()
    n = np.zeros(3, int)
    for t in T:
        for o in t["objs"]:
            n[o["cls"]] += 1
    print("tiles", len(T), "objects per class", n.tolist(), "max per tile", max(len(t["objs"]) for t in T))
    a = np.array([o["area"] for t in T for o in t["objs"]])
    print("area small/medium/large", int((a < 32 ** 2).sum()), int(((a >= 32 ** 2) & (a < 96 ** 2)).sum()), int((a >= 96 ** 2).sum()))
    B = big_scene()
    print("big scene objects", len(B["objs"]), np.bincount([o["cls"] for o in B["objs"]]).tolist())
    Q = sequence()
    print("sequence gt", len(Q["gt"]), "det", len(Q["det"]))


# ----------------------------------------------------------------- 그림용 렌더링(평가에는 쓰지 않음)
def render(objs, clutter, kind="sea", size=TILE, seed=0, land_frac=None):
    """RGB uint8 영상. 바다·땅 바탕에 객체와 방해물을 그림(그림 예시용)"""
    import cv2
    from scipy import ndimage as ndi
    rng = np.random.default_rng(1000 + seed)
    img = np.zeros((size, size, 3), np.float32)
    sea = np.array([28, 52, 72], np.float32)
    n = ndi.gaussian_filter(rng.normal(0, 1, (size, size)), 6)
    n = n / (n.std() + 1e-6)
    img[:] = sea + 5 * n[..., None]
    if kind == "harbor" or land_frac:
        lx = int(size * (land_frac or max([o["cx"] + o["w"] for o in objs if o["cls"] == 2] + [size * 0.4]) / size))
        lx = min(max(lx, int(size * 0.35)), int(size * 0.55))
        img[:, :lx] = np.array([118, 116, 110], np.float32) + 4 * n[:, :lx, None]
        img[:, lx - 6:lx] = [160, 158, 150]
    for c in clutter:
        p = obb_corners(c["cx"], c["cy"], c["w"], c["h"], c["theta"])
        col = (215, 225, 230) if c["kind"] == "whitecap" else (150, 90, 60)
        if c["kind"] == "whitecap":
            cv2.ellipse(img, ((float(c["cx"]), float(c["cy"])), (float(c["w"]), float(c["h"])), float(c["theta"])), col, -1)
        else:
            cv2.fillPoly(img, [np.round(p * 16).astype(np.int32)], col, lineType=cv2.LINE_AA, shift=4)
    for o in objs:
        p = obb_corners(o["cx"], o["cy"], o["w"], o["h"], o["theta"])
        if o["cls"] == 2:
            col = [(200, 30, 30), (230, 230, 230), (40, 40, 45), (60, 90, 170), (170, 170, 175)][int(rng.integers(0, 5))]
        elif o["cls"] == 1:
            col = (238, 238, 232)
        else:
            col = (190, 188, 180)
        cv2.fillPoly(img, [np.round(p * 16).astype(np.int32)], col, lineType=cv2.LINE_AA, shift=4)
        if o["cls"] == 0:   # 갑판 선
            q = obb_corners(o["cx"], o["cy"], o["w"] * 0.7, o["h"] * 0.4, o["theta"])
            cv2.fillPoly(img, [np.round(q * 16).astype(np.int32)], (120, 60, 50), lineType=cv2.LINE_AA, shift=4)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
