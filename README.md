# VisionDrill

컴퓨터 비전 분야 신입 개발자·비전공자용 데스크톱 반복학습 앱. 전문용어와 기본 개념을 단기간에 익히는 것이 목적임.
5지선다·단답형·지시문 해석형 문항을 매일 풀고, 제출 후 일괄 채점·해설을 확인함. 틀린 문항은 오답노트와 간격 반복으로 다시 출제됨.

최종 목표는 암기가 아니라 **"mAP50은 높은데 50-95가 낮아, 박스 정밀도 봐줘" 같은 지시를 듣고 할 일을 아는 상태**임.

## 구성

| 앱 | 대상 | 기능 |
|---|---|---|
| **VisionDrill** (`apps/drill`) | 학습자 | 오늘의 학습, 제출 후 일괄 채점·해설, 오답노트, 트랙 해금, 용어 사전 |
| **VisionDrill Studio** (`apps/studio`) | 출제·검수자 | 문항·용어·커리큘럼 편집, 검수, 변경 이력, JSON 가져오기/내보내기, 문제 팩(.vdpack) 생성 |

두 앱 모두 Tauri 2(Windows·macOS) 기반. 로직은 전부 웹뷰 안에서 TypeScript + SQLite(WASM, sql.js)로 처리함.

```
문제은행 원본 (Studio, SQLite 파일 1개)
   │  검수 완료 문항만 추림
   ▼
문제 팩 .vdpack (읽기 전용 SQLite) ── GitHub Releases ──▶ 학습 앱이 가져와서 학습 기록과 합침
```

## 커리큘럼

| 트랙 | 내용 |
|---|---|
| A | 통계 기초 |
| B | 회귀분석·회귀진단 (OLS, 잔차, VIF, Cook's D, Durbin-Watson …) |
| C | ML 기초 |
| D | 영상·고전 CV |
| E | 딥러닝 기초·아키텍처 |
| F | 태스크·지표 (IoU, NMS, P/R, mAP …) |
| G | 학습·데이터 실무 |
| H | 비전 모델 진단 (Diagnostics: 오차 분해, 슬라이스, 신뢰도 …) |

선수 트랙 숙련도가 기준(기본 70%)을 넘으면 다음 트랙이 열림.
용어는 **업계 표준 / 프로젝트 정의**로 구분함. 사내에서 정의한 지표를 외부 통용 용어로 오해하지 않게 하기 위함.

## 설치 (학습자)

[Releases](https://github.com/SeaViewer91/VisionDrill/releases)에서 운영체제에 맞는 파일을 받음.

- **Windows**: `VisionDrill_x.y.z_x64-setup.exe` (또는 `.msi`). 서명되지 않은 앱이라 SmartScreen 경고가 뜨면 *추가 정보 → 실행*.
- **macOS (Apple Silicon)**: `VisionDrill_x.y.z_aarch64.dmg`. "확인되지 않은 개발자" 경고가 뜨면 앱을 *우클릭 → 열기*, 또는 터미널에서
  `xattr -dr com.apple.quarantine /Applications/VisionDrill.app`

첫 실행 시 앱에 내장된 샘플 문제 팩이 설치됨. 새 문제 팩은 *설정 → 문제 팩 가져오기*로 적용하며, 학습 기록은 유지됨.

## 개발

필요 환경: Node.js 22+ (npm 10 포함), Rust(stable, `rustup`). macOS는 Xcode Command Line Tools, Windows는 WebView2(기본 설치)와 MSVC 빌드 도구.

```bash
npm install
npm test                     # 코어 단위 테스트
npm run tauri:drill dev      # 학습 앱 데스크톱 개발 실행
npm run tauri:studio dev     # Studio 데스크톱 개발 실행
npm run dev:drill            # 브라우저 미리보기 (http://localhost:1420, 기록은 브라우저에 저장)
npm run dev:studio           # 브라우저 미리보기 (http://localhost:1421)
npm run tauri:drill build    # 설치 파일 빌드 (현재 OS용)
npm run sample-pack          # samples/sample-bank.json → 학습 앱 기본 팩 재생성
npm run lint:drafts          # bank/incoming/*.json 초안 점검 (docs/authoring-guide.md)
npm run merge:drafts         # 초안 + 샘플 → bank/visiondrill-bank.db, 미리보기 팩
```

DB 스키마 변경 시 `packages/core/db/*.sql`을 수정한 뒤 `npm run gen -w @visiondrill/core`로 번들용 사본을 갱신함.

### 폴더 구조

```
apps/drill/        학습 앱 (React + Vite + Tauri)
apps/studio/       문제은행 관리 앱
packages/core/     DB 스키마, 문제은행·학습 로직, 채점, 가져오기/내보내기, 팩 (UI 없음, 단위 테스트)
packages/ui/       두 앱 공용 문항·결과 카드와 스타일 (Studio 미리보기 = 학습 앱 화면)
packages/platform/ Tauri 파일 대화상자·원자적 저장·백업 / 브라우저 대체 구현
samples/           샘플 문항(JSON, 교환 형식)
content/           커리큘럼(장 목록)·장별 목표 문항 수
docs/              문항 작성 가이드
tools/             샘플 팩 빌드, 초안 점검(lint), 초안 병합 스크립트
```

## 문제은행 운영

1. **Studio**에서 문제은행 파일(`.db`)을 열거나 새로 만듦. 저장할 때마다 파일에 바로 쓰고(임시 파일 → 교체), 열 때·닫을 때 같은 폴더의 `backups/`에 사본을 남김.
2. 문항은 직접 작성하거나, *가져오기·내보내기* 화면의 **LLM 초안 요청 프롬프트**로 받은 JSON을 붙여넣음.
3. 미리보기로 학습 앱 화면을 확인한 뒤 **검수 완료**로 전환. 검수 완료에는 보기 5개, 오답별 틀린 이유, 해설, 검수자 기록이 필요함(DB에서 강제).
4. *문제 팩 만들기*로 `.vdpack`을 만들어 `pack-YYYY.MM.DD` 태그의 Release에 올림.

문제은행 원본 `.db`는 저장소에 넣지 않음(`.gitignore`). 원본은 로컬·NAS에 두고, 저장소는 앱 코드와 배포만 담당함.

문항 오류는 학습 앱 결과 화면의 **이 문항 오류 신고** 버튼(GitHub 이슈)으로 접수함.

## 라이선스

- 코드: [MIT](LICENSE)
- 문항 콘텐츠: [CC BY 4.0](LICENSE-CONTENT.md)
