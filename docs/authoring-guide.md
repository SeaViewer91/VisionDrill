# VisionDrill 문항 작성 가이드 (출제자·검증자 공용)

## 0. 누구를 위한 문항인가
- 대상: 원격탐사·GIS 회사에 새로 들어온 컴퓨터 비전 초보 개발자, 비전공자.
- 목표: 전문용어를 외우는 것을 넘어 **"전문용어로 된 업무 지시를 듣고 무엇을 할지 아는 것"**.
- 학습 앱은 5지선다·단답형·지시문 해석형 세트를 풀게 한 뒤 한꺼번에 채점하고, 오답이면 짧은 해설 + 고른 보기가 틀린 이유를 보여준다.

## 1. 파일 형식 (교환 JSON)
장 하나당 파일 하나: `bank/incoming/<장ID>.json` (예: `bank/incoming/G.aug.json`).
장(chapters)은 `content/curriculum.json`에 이미 정의되어 있으므로 **파일에 chapters를 넣지 않는다.**

```json
{
  "format": "visiondrill-bank-exchange",
  "format_version": 1,
  "terms": [
    { "id": "mosaic-augmentation", "term_en": "Mosaic Augmentation", "term_ko": "모자이크 증강", "abbreviation": null,
      "definition": "네 장의 이미지를 한 장으로 이어 붙여 다양한 배경·축척의 객체를 한 번에 학습시키는 증강",
      "usage_example": "타일 경계에 걸린 소형 선박이 많으니 mosaic 확률 올려서 다시 돌려봐.",
      "origin": "standard", "track": "G" }
  ],
  "questions": [
    { "id": "G-AUG-001", "type": "mcq", "chapter": "G.aug", "cognitive_level": 2, "difficulty": 2,
      "stem": "문제 본문",
      "choices": [ { "text": "보기1", "rationale": "이 보기가 틀린 이유" }, { "text": "보기2(정답)", "rationale": null },
                   { "text": "보기3", "rationale": "..." }, { "text": "보기4", "rationale": "..." }, { "text": "보기5", "rationale": "..." } ],
      "answer": 2,
      "explanation_short": "2~3문장 해설",
      "explanation_full": "상세 해설 (배경, 실무 팁, 연결 개념)",
      "tags": ["augmentation"], "terms": ["mosaic-augmentation"],
      "source_ref": "출처", "status": "draft" },
    { "id": "G-AUG-002", "type": "short", "chapter": "G.aug", "cognitive_level": 1, "difficulty": 1,
      "stem": "…의 이름은? (영문 또는 한글)",
      "accepted_answers": { "en": ["Mosaic augmentation", "Mosaic"], "ko": ["모자이크 증강", "모자이크"] },
      "primary": "Mosaic augmentation",
      "explanation_short": "…", "explanation_full": "…", "tags": ["augmentation"], "terms": ["mosaic-augmentation"],
      "source_ref": "…", "status": "draft" }
  ]
}
```

필드 규칙
- `id`: `<트랙>-<장코드>-<3자리 번호>` (예: `G-AUG-001`). 장 코드는 curriculum.json의 `code`. 번호는 1부터 연속(기존 문항이 있는 장은 이어서).
- `type`: `mcq`(5지선다) · `short`(단답형) · `scenario`(지시문 해석형, 형식은 mcq와 같음)
- `cognitive_level`: 1 용어 인지 · 2 개념 구분 · 3 지시 해석. **scenario는 반드시 3**.
- `difficulty`: 1(쉬움)~5(어려움). 장 안에서 고르게.
- `answer`: 1~5. 정답 보기의 `rationale`은 `null`, **오답 보기 4개는 틀린 이유 필수**.
- `status`: 항상 `"draft"`. `reviewed_by`는 쓰지 않는다.
- `terms`: 이 문항과 연결할 용어 ID 목록 (1~3개).
- `source_ref`: 근거. 일반 지식은 표준 교재·공식 문서 이름(예: `ISLR 3장`, `COCO detection evaluation`, `PyTorch 문서 torch.optim`), H 트랙은 `Diagnostics/research/note/<파일>#<절 번호>`.

## 2. 용어
- 각 출제자는 **자기 장의 핵심 용어를 정의**한다 (장당 5~9개).
- 용어 ID: 영문 정식 명칭의 kebab-case 소문자 (예: `false-discovery-rate`, `batch-normalization`, `ground-sample-distance`).
- **이미 있는 용어는 다시 정의하지 말고 ID로만 참조**한다: `ols, r-squared, residual, heteroscedasticity, qq-plot, multicollinearity, vif, ridge-regression, leverage, cooks-distance, autocorrelation, durbin-watson, iou, nms, soft-nms, precision, recall, confidence-threshold, map, tide, e-loc, e-nms-kill, vdr`
- 다른 장 소관 개념을 참조할 때도 같은 규칙으로 ID를 짐작해 적는다 (병합 단계에서 맞춘다).
- `origin`: 업계·학계 표준은 `standard`, Diagnostics에서 새로 정의한 지표·절차는 `project`.
- `usage_example`: **원격탐사·GIS 실무에서 동료가 실제로 할 법한 한 문장** (반말 지시·보고체 자유).

## 3. 문항 구성 원칙: 용어 하나에 문항 2~3개
- 장의 **모든 핵심 용어가 1단계 문항 1개 이상, 2단계 문항 1개 이상**을 갖게 한다.
- 실무에서 지시로 오가는 용어에는 3단계(scenario) 문항을 붙인다.
- 장 안 유형 비율 목표: 단답형 ~30% · 5지선다 ~45% · 지시문 ~25%
  - A(통계)·D(고전 CV) 트랙은 지시문 10~20%, F·G·H 트랙은 25~35%
- 문항 하나에 개념 하나. 같은 사실을 묻는 중복 문항 금지.

## 4. 지시문 해석형(scenario) — 원격탐사·GIS 상황으로
형식:
```
지시: "(선임이 실제로 할 법한 한두 문장. 구체적 수치 포함)"

가장 적절한 조치는?        ← 또는 "가장 먼저 확인할 것은?", "이 지시의 의도로 옳은 것은?"
```
상황 소재 예시 (골고루 쓸 것):
- 광학 위성: Sentinel-2(10/20/60m 밴드), Landsat 8/9, KOMPSAT-3/3A, 고해상도 상용 위성, 다중분광·초분광
- SAR: Sentinel-1, 스페클, 편파(VV/VH), 입사각
- 항공·드론: 정사영상, GSD(cm급), 중복도, 모자이크 경계
- 태스크: 토지피복 분류, 건물·도로 추출, 선박·차량·양식장·해안 쓰레기 탐지, 변화탐지, 구름·그림자 마스킹, 수계 추출
- 지표·전처리: NDVI·NDWI, 대기보정(TOA/BOA), 정사보정, 밴드 정합, 리샘플링, 정규화
- GIS: 좌표계(EPSG:4326, EPSG:5186 등), GeoTIFF, Shapefile/GeoJSON, 래스터↔벡터 변환, 타일링(예: 512×512, overlap 64), 지상기준점(GCP), 공간 자기상관, 공간 교차검증
- **금지**: 실제 회사 과제명, 발주처·고객사 이름, 실제 과제의 수치. 수치는 그럴듯하게 새로 만든다.
- 1·2단계 문항도 예시가 필요하면 원격탐사·GIS 예시를 우선 쓴다 (단, 개념 자체가 일반적이면 억지로 끼우지 않는다).

## 5. 보기 설계
- **오답은 실제로 헷갈리는 인접 개념·흔한 오해**로 만든다 (예: Precision↔Recall, 레버리지↔Cook's D, NMS↔Max Pooling, 정사보정↔대기보정). 아무도 고르지 않을 보기 금지.
- **정답은 정확히 하나.** 다른 보기가 "상황에 따라 맞을 수도" 있으면 안 된다.
- 금지: "모두 정답", "위의 보기 모두", "①과 ② 모두" 등 위치·전체 참조 보기 (학습 앱이 보기를 섞는다), 이중 부정.
- "옳지 **않은** 것은?" 형식은 장당 1~2개 이하.
- **정답 위치를 1~5에 고르게** (장 안에서 한 번호가 35%를 넘지 않게).
- **정답 보기가 가장 긴 보기인 경우가 40%를 넘지 않게.** 오답 보기도 정답만큼 구체적으로 쓴다.
- `rationale`: "왜 틀렸는지" + 가능하면 "그 보기는 사실 무엇인지"를 1~2문장으로.

## 6. 단답형
- 본문 끝에 `(영문 또는 한글)` 같은 안내를 붙인다.
- `accepted_answers.en`과 `.ko` **둘 다** 채운다. 흔한 약어·동의어 포함.
- 대소문자·공백·하이픈·마침표·따옴표 차이는 자동으로 무시되므로 **그런 변형은 넣지 않는다** (넣으면 중복 오류).
- 괄호 안·밖은 각각 인정되므로 "Intersection over Union (IoU)" 하나로 IoU까지 인정된다.
- 답이 하나로 정해지는 질문만 낸다 (용어 이름·약어·간단한 수치).

## 7. 해설
- `explanation_short`: **2~3문장, 400자 이내.** 왜 그것이 정답인지 핵심만.
- `explanation_full`: 배경·수식·실무 팁·연결 개념. 원격탐사·GIS 맥락을 한 줄 넣으면 좋다.
- 프로젝트 정의 용어(origin=project)가 들어간 문항은 explanation_full에 **"외부와 소통할 때는 ~처럼 풀어서 말한다"**를 한 줄 넣는다.
- 계산 문항은 **직접 검산**하고 해설에 계산 과정을 보인다.
- 문체: 존댓말 평서문(~입니다), 한국어 본문 + 영문 용어 병기.

## 8. 정확성
- 확신이 없는 사실은 문항으로 만들지 않는다.
- 버전·제품에 따라 달라지는 세부(특정 라이브러리 기본값 등)는 피하거나 "일반적으로"로 한정한다.
- H 트랙은 **반드시 원문(`/mnt/user-data/uploads/Diagnostics/research/note/`)을 읽고** 원문 정의·수식·임계값에 맞춘다. 원문의 경험적 임계값은 "잠정치"임을 해설에 밝힌다.
- 교재·문서 문장을 그대로 베끼지 않는다.

## 9. 출제자 자체 점검 (필수)
1. `cd /home/claude/VisionDrill && npx tsx tools/lint-drafts.ts bank/incoming/<파일>.json` 을 실행해 **오류 0개**가 될 때까지 고친다.
2. 모든 문항을 **처음 보는 사람처럼 직접 풀어 본다**: 정답이 하나뿐인지, 정답 근거가 사실인지, 오답 이유가 사실인지, 본문만으로 풀 수 있는지.
3. 고친 내용을 `bank/incoming/_logs/<그룹>.author.md`에 목록으로 남긴다.

## 10. 교차 검증자 (출제하지 않은 에이전트)
- 각 문항을 **정답을 보지 않고 먼저 풀어 본 뒤** 파일의 정답과 비교한다.
- 확인 항목: 사실 오류, 정답 복수 가능성(모호성), 오답 이유의 사실성, 계산, 해설 오류, 단답형 허용 답 누락(흔한 동의어), 용어 정의 오류, H 트랙 원문과의 불일치, 금지 사항(회사 과제명 등).
- 문제가 있으면 **파일을 직접 고치고**, 고칠 수 없을 만큼 나쁘면 그 문항을 삭제한다.
- 수정·삭제 내역을 `bank/incoming/_logs/<그룹>.review.md`에 문항 ID별로 남긴다 (무엇이 문제였고 어떻게 고쳤는지).
- 마지막에 lint를 다시 돌려 오류 0개를 확인한다.

## 11. 병합 (문제은행 DB 만들기)
- `npm run lint:drafts` → 모든 초안 오류 0 확인
- `npm run merge:drafts` → `bank/visiondrill-bank.db`(원본 DB) + `bank/merge-report.md` + `bank/preview-draft.vdpack`(사람 검수 전 미리보기 팩)
  - 여러 장이 같은 용어 ID를 정의하면 커리큘럼 순서상 먼저 나온 장의 정의를 쓴다 (샘플 용어가 최우선).
  - 다른 이름으로 참조한 용어는 `tools/merge-drafts.ts`의 `ALIAS`, 아무 데도 정의되지 않은 용어는 `STUB_TERMS`에 추가한다.
  - 새 문항과 새 용어는 모두 초안(draft)으로 들어간다. Studio에서 검수 완료로 바꾼 것만 학습 팩으로 내보내진다.
- 이미 Studio로 검수를 시작한 DB가 있다면 병합을 다시 돌려 덮어쓰지 말고, Studio의 "가져오기"로 필요한 장 파일만 들여온다.
