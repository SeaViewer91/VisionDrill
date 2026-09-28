# VisionDrill

컴퓨터 비전 분야 신입 개발자·비전공자용 데스크톱 반복학습 앱. 전문용어와 기본 개념을 단기간에 익히는 것이 목적임.
5지선다·단답형·지시문 해석형 문항을 매일 풀고, 제출 후 일괄 채점·해설을 확인함. 틀린 문항은 오답노트와 간격 반복으로 다시 출제됨.

최종 목표는 암기가 아니라 **"mAP50은 높은데 50-95가 낮아, 박스 정밀도 봐줘" 같은 지시를 듣고 할 일을 아는 상태**임.
지시문 문항은 원격탐사·GIS 실무 상황(위성·항공·드론 영상, SAR, 좌표계, 타일링 등)을 배경으로 함.

## 구성

| 앱 | 대상 | 기능 |
|---|---|---|
| **VisionDrill** (`apps/drill`) | 학습자 | 오늘의 학습, 제출 후 일괄 채점·해설, 오답노트, 트랙 해금, 용어 사전 |
| **VisionDrill Studio** (`apps/studio`) | 출제·검수자 | 문항·용어·커리큘럼 편집, 검수, 변경 이력, JSON 가져오기/내보내기, 문제 팩(.vdpack) 생성 |

두 앱 모두 Tauri 2(Windows·macOS) 기반. 로직은 전부 웹뷰 안에서 TypeScript + SQLite(WASM, sql.js)로 처리함.

```
초안 JSON (bank/incoming/) ─ 병합 ─▶ 문제은행 원본 (Studio, SQLite 파일 1개)
                                        │  검수 완료 문항만 추림
                                        ▼
문제 팩 .vdpack (읽기 전용 SQLite) ── 앱 내장 / GitHub Releases ──▶ 학습 앱이 가져와서 학습 기록과 합침
```

## 커리큘럼

| 트랙 | 내용 | 문항 |
|---|---|---|
| A | 통계 기초 | 89 |
| B | 회귀분석·회귀진단 (OLS, 잔차, VIF, Cook's D, Durbin-Watson …) | 139 |
| C | ML 기초 | 133 |
| D | 영상·고전 CV | 141 |
| E | 딥러닝 기초·아키텍처 | 166 |
| F | 태스크·지표 (IoU, NMS, P/R, mAP …) | 157 |
| G | 학습·데이터 실무 | 149 |
| H | 비전 모델 진단 (Diagnostics: 오차 분해, 슬라이스, 신뢰도 …) | 230 |

선수 트랙 숙련도가 기준(기본 70%)을 넘으면 다음 트랙이 열림. *설정 → 모든 트랙 열기*로 해금 조건을 끌 수 있음.
용어는 **업계 표준 / 프로젝트 정의**로 구분함. H 트랙의 Diagnostics 프로젝트에서 새로 정의한 지표를 외부 통용 용어로 오해하지 않게 하기 위함.

## 문항 현황

- 기본 팩 `0.1.1-draft`: 1,204문항, 용어 487개, 63개 장
- 21문항(샘플)만 사람 검수 완료. 나머지 1,183문항은 **사람 검수 전 초안**임
  - 출제 에이전트가 작성하고, 출제하지 않은 별도 에이전트가 정답을 가린 채 다시 풀어 교차 검증함(계산 재계산, H 트랙은 원문 대조)
  - 틀린 문항이 섞여 있을 수 있음. 결과 화면의 **이 문항 오류 신고** 버튼으로 알려 주면 반영함

## 설치 (학습자)

[Releases](https://github.com/SeaViewer91/VisionDrill/releases)에서 운영체제에 맞는 파일을 받음.

| 앱 | Windows | macOS (Apple Silicon) |
|---|---|---|
| 학습 앱 | `VisionDrill_x.y.z_x64-setup.exe` (또는 `.msi`) | `VisionDrill_x.y.z_aarch64.dmg` |
| Studio | `VisionDrill.Studio_x.y.z_x64-setup.exe` (또는 `.msi`) | `VisionDrill.Studio_x.y.z_aarch64.dmg` |

- 학습만 할 사람은 학습 앱만 설치하면 됨. Studio는 문항을 고치거나 검수할 사람용.
- **Windows**: 서명되지 않은 앱이라 SmartScreen 경고가 뜨면 *추가 정보 → 실행*.
- **macOS**: "확인되지 않은 개발자" 또는 "손상되었기 때문에 열 수 없음" 경고가 뜨면 앱을 *우클릭 → 열기*, 또는 터미널에서
  `xattr -dr com.apple.quarantine /Applications/VisionDrill.app`

첫 실행 시 앱에 내장된 기본 문제 팩이 설치됨. 앱을 업데이트하면 기본 팩도 새 버전으로 교체되며, 학습 기록은 유지됨.
다른 문제 팩은 *설정 → 문제 팩 가져오기*로 적용함. 직접 가져온 팩은 앱 업데이트 때 덮어쓰지 않음.

## 개발

필요 환경: Node.js 22+ (npm 10 포함), Rust(stable, `rustup`). macOS는 Xcode Command Line Tools, Windows는 WebView2(기본 설치)와 Visual Studio Build Tools(C++ 데스크톱 개발).

```bash
npm install
npm test                     # 코어 단위 테스트
npm run tauri:drill dev      # 학습 앱 데스크톱 개발 실행
npm run tauri:studio dev     # Studio 데스크톱 개발 실행
npm run dev:drill            # 브라우저 미리보기 (http://localhost:1420, 기록은 브라우저에 저장)
npm run dev:studio           # 브라우저 미리보기 (http://localhost:1421)
npm run tauri:drill build    # 설치 파일 빌드 (현재 OS용) → target/release/bundle/
npm run lint:drafts          # bank/incoming/*.json 초안 점검 (docs/authoring-guide.md)
npm run merge:drafts         # 초안 + 샘플 → bank/visiondrill-bank.db, 학습 앱 기본 팩 재생성
npm run sample-pack          # samples/sample-bank.json → samples/sample.vdpack (테스트·시연용)
```

DB 스키마 변경 시 `packages/core/db/*.sql`을 수정한 뒤 `npm run gen -w @visiondrill/core`로 번들용 사본을 갱신함.

### 폴더 구조

```
apps/drill/        학습 앱 (React + Vite + Tauri). public/packs/default.vdpack = 내장 기본 팩
apps/studio/       문제은행 관리 앱
packages/core/     DB 스키마, 문제은행·학습 로직, 채점, 가져오기/내보내기, 팩 (UI 없음, 단위 테스트)
packages/ui/       두 앱 공용 문항·결과 카드와 스타일 (Studio 미리보기 = 학습 앱 화면)
packages/platform/ Tauri 파일 대화상자·원자적 저장·백업 / 브라우저 대체 구현
bank/incoming/     장별 문항 초안 JSON (교환 형식, 파일 하나 = 장 하나)
samples/           샘플 문항(JSON, 교환 형식)
content/           커리큘럼(장 목록)·장별 목표 문항 수
docs/              문항 작성 가이드
tools/             초안 점검(lint), 초안 병합, 샘플 팩 빌드 스크립트
```

## 문제은행 운영

**초안 단계**
1. `docs/authoring-guide.md` 형식에 맞춰 `bank/incoming/<장ID>.json`을 작성함.
2. `npm run lint:drafts`로 오류 0개 확인.
3. `npm run merge:drafts`로 원본 DB(`bank/visiondrill-bank.db`)와 학습 앱 기본 팩을 다시 만듦.

**검수 단계 (Studio)**
1. **Studio**에서 문제은행 파일(`.db`)을 엶. 저장할 때마다 파일에 바로 쓰고(임시 파일 → 교체), 열 때·닫을 때 같은 폴더의 `backups/`에 사본을 남김.
2. 문항은 직접 작성하거나, *가져오기·내보내기* 화면의 **LLM 초안 요청 프롬프트**로 받은 JSON을 붙여넣음.
3. 미리보기로 학습 앱 화면을 확인한 뒤 **검수 완료**로 전환. 검수 완료에는 보기 5개, 오답별 틀린 이유, 해설, 검수자 기록이 필요함(DB에서 강제).
4. *문제 팩 만들기*로 `.vdpack`을 만들어 `pack-YYYY.MM.DD` 태그의 Release에 올림.

문제은행 원본 `.db`와 백업은 저장소에 넣지 않음(`.gitignore`). 원본은 로컬·NAS에 두고, 저장소에는 앱 코드·초안 JSON·기본 팩만 둠.
Studio에서 검수를 시작한 뒤에는 `merge:drafts`로 원본 DB를 덮어쓰지 말 것. 새 초안은 Studio의 *가져오기*로 장 단위로 들여옴.

문항 오류는 학습 앱 결과 화면의 **이 문항 오류 신고** 버튼(GitHub 이슈)으로 접수함.

## 라이선스

- 코드: [MIT](LICENSE)
- 문항 콘텐츠: [CC BY 4.0](LICENSE-CONTENT.md)
