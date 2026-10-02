"""부록 B 생성: 각 강의 '## 헷갈리는 쌍' 절을 그대로 모아 content/lessons/B.md를 다시 씀.

강 원고를 고친 뒤 이 스크립트를 다시 돌리면 부록 B가 따라옴(손으로 B.md의 쌍 목록을 고치지 않음).
    python3 content/lessons/fig-src/B.py
"""
import json, os, re, glob

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.dirname(HERE)
PARTS = {p["part"]: p for p in json.load(open(os.path.join(L, "parts.json"), encoding="utf-8"))["parts"]}

HEAD = """---
id: B
part: 9
title: 헷갈리는 용어 쌍 모음
chapters: []
version: 1
updated: 2026-10-02
status: draft
---

# 부록 B 헷갈리는 용어 쌍 모음

## 이 부록의 쓰임

각 강 끝의 "헷갈리는 쌍"을 부와 강 순서대로 한곳에 모았음. 문장은 강 원고에서 그대로 가져왔으므로 "이 강", "이 자료"는 그 쌍이 실린 강과 그 강의 예제 자료를 가리킴. 설명이 짧게 느껴지면 괄호 안 강 번호로 돌아가 본문을 읽음.

쓰는 법은 두 가지임.
- **지시를 받았는데 두 말이 섞여 들릴 때**: 그 말이 나오는 부를 찾아 쌍을 읽고, 지시한 사람이 어느 쪽을 뜻했는지 되물음. 되묻는 것이 틀리게 하는 것보다 빠름
- **복습할 때**: 쌍의 앞말만 보고 뒷말과의 차이를 먼저 말해 본 뒤 설명과 맞춰 봄

## 같은 말, 다른 뜻

책 전체에서 이름이 같거나 비슷해 가장 자주 섞이는 것만 따로 뽑았음. 자세한 설명은 괄호 안 강에 있음.

| 같은(비슷한) 말 | 뜻 1 | 뜻 2 |
|---|---|---|
| 정규화 | 규제(regularization): 과적합을 막는 벌점·드롭아웃 등(11·14·29강) | 정규화(normalization): 값의 척도를 맞춤(10강), 배치·레이어 정규화 층(29강) |
| 편향 | 평균 예측이 참값에서 비켜난 정도(14강), 뽑는 방식의 치우침(3강) | 신경망 층의 상수항 bias(27강) |
| 검증 | 검증 세트(validation): 모델·하이퍼파라미터를 고르는 데 쓰는 자료(15·43강) | 정확도 평가의 검증점: 다 만든 지도를 평가하는 표본(3·17강). 고르는 데 쓰지 않으므로 15강의 시험 세트 쪽 역할임 |
| 시험·검정(test) | 시험 세트(test set): 모든 선택이 끝난 뒤 한 번 재는 자료(15강) | 가설검정(hypothesis test): p값으로 차이가 우연인지 따짐(4강) |
| AP | 한 클래스의 PR 곡선 넓이(39강) | COCO가 클래스 평균인 mAP50-95를 그냥 부르는 이름(39강) |
| 정밀도 | 정밀도(precision): 양성이라 한 것 가운데 맞은 비율(17·38강) | 수치 정밀도: fp32·fp16·INT8처럼 숫자를 담는 형식(47·50강) |
| 해상도 | 원격탐사의 해상도: 공간(GSD)·분광·방사·시간 네 가지(20강) | 딥러닝의 "해상도 640": 입력 영상의 화소 수(20강). 확대·샤프닝으로는 공간해상도가 늘지 않음(20·22강) |
| 합성곱 | 신호처리의 커널을 뒤집는 합성곱(22강) | 딥러닝 합성곱 층, 실제로는 상관(22·30강) |
| 커널 | 필터·합성곱 층의 가중치 창(22·30강) | GPU에서 한 번에 실행되는 연산 단위, 층 융합(50·60강) |
| 스트라이드 | 합성곱 커널을 한 번에 옮기는 칸 수(30강) | 타일을 잘라 낼 때 미는 간격(43강) |
| DCR | 비활성 채널 비율(56강) | 원문 노트가 다크 채널 잔류도에도 쓴 약어(57강) |
| FDR | 거짓 발견율(4·54강) | 오퍼레이터 융합 결손도(60강) |
| RR | 상대위험(relative risk, 12·61강) | 회귀율(61강) |
| 누수 | 학습·검증·시험 자료가 섞이는 데이터 누수(15·43강) | NMS가 덜 지워 남는 중복 상자(53강) |
| 보정 | 영상 보정: 위치를 바로잡는 정사보정·GCP 보정(25·26강), 값을 바로잡는 대기보정(25강) | 확신도를 정확도에 맞추는 확률 보정(55강) |
"""


def sections(f):
    s = open(f, encoding="utf-8").read()
    meta = dict(re.findall(r"^(id|part|title):\s*(.*)$", s.split("---")[1], re.M))
    m = re.search(r"^## 헷갈리는 쌍\n(.*?)(?=^## )", s, re.S | re.M)
    items = [l for l in m.group(1).strip().splitlines() if l.startswith("- ")] if m else []
    return int(meta["part"]), meta["id"].strip(), meta["title"].strip(), items


def main():
    out = [HEAD.rstrip() + "\n"]
    rows = sorted((sections(f) for f in glob.glob(os.path.join(L, "[0-9][0-9].md"))), key=lambda r: (r[0], int(r[1])))
    n = 0
    for part in sorted({r[0] for r in rows}):
        lessons = [r for r in rows if r[0] == part and r[3]]
        if not lessons:
            continue
        p = PARTS[part]
        sub = f" ({p['subtitle']})" if p.get("subtitle") else ""
        out.append(f"\n## {part}부 {p['title']}{sub}\n")
        for _, lid, title, items in lessons:
            out.append(f"\n### {int(lid)}강 {title}\n\n")
            for it in items:
                # 강 안에서만 통하는 참조를 부록에서도 읽히게 바꿈("이 강" = 그 쌍이 실린 강)
                it = it.replace("그림의 타일도", "이 강 그림의 타일도")
                it = it.replace("(5절 예에서", "(이 강 5절 예에서")
                out.append(it + "\n"); n += 1
    open(os.path.join(L, "B.md"), "w", encoding="utf-8").write("".join(out))
    print("pairs", n)


if __name__ == "__main__":
    main()
