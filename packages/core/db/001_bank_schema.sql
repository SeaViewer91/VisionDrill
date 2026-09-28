-- =====================================================================
-- VisionDrill 문제은행(Bank) 스키마  v0.1 (초안)
-- 대상: Studio가 편집하는 원본 DB (visiondrill-bank.db)
--       배포 팩(.vdpack)도 같은 스키마에서 reviewed 문항만 추려 만든다.
-- 주의: SQLite는 연결마다 PRAGMA foreign_keys = ON; 을 실행해야 FK가 동작한다.
-- 시각 형식: ISO-8601 UTC 문자열 (예: 2026-09-28T08:30:00.000Z)
-- =====================================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- 0. 메타 정보
-- ---------------------------------------------------------------------
CREATE TABLE bank_meta (
    key   TEXT PRIMARY KEY,           -- schema_version, bank_name, created_at ...
    value TEXT NOT NULL
);

-- ---------------------------------------------------------------------
-- 1. 커리큘럼: 트랙 → 장
-- ---------------------------------------------------------------------
CREATE TABLE tracks (
    id          TEXT PRIMARY KEY CHECK (length(id) = 1),   -- 'A' ~ 'H'
    name_ko     TEXT NOT NULL,
    name_en     TEXT NOT NULL,
    description TEXT,
    sort_order  INTEGER NOT NULL
);

-- 트랙 해금 조건: track_id는 prereq_track_id 숙련도가 min_mastery 이상일 때 열림
CREATE TABLE track_prereqs (
    track_id        TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    prereq_track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    min_mastery     REAL NOT NULL DEFAULT 0.7 CHECK (min_mastery > 0 AND min_mastery <= 1),
    PRIMARY KEY (track_id, prereq_track_id),
    CHECK (track_id <> prereq_track_id)
);

CREATE TABLE chapters (
    id          TEXT PRIMARY KEY,                           -- 예: 'F.nms', 'B.vif', 'H.01'
    track_id    TEXT NOT NULL REFERENCES tracks(id),
    code        TEXT NOT NULL,                              -- 문항 ID에 쓰는 약어: 'NMS', 'VIF'
    name_ko     TEXT NOT NULL,
    name_en     TEXT,
    sort_order  INTEGER NOT NULL,
    stage       INTEGER NOT NULL DEFAULT 1 CHECK (stage BETWEEN 1 AND 3), -- H트랙 개방 차수(1차/2차/3차), 그 외 1
    source_ref  TEXT,                                       -- 장 단위 원문 (예: Diagnostics/research/note/01_...)
    UNIQUE (track_id, code)
);

-- ---------------------------------------------------------------------
-- 2. 이미지 등 첨부 파일 (DB 안에 BLOB으로 보관 → 파일 하나로 이동/배포)
-- ---------------------------------------------------------------------
CREATE TABLE assets (
    id         INTEGER PRIMARY KEY,
    filename   TEXT NOT NULL,
    mime       TEXT NOT NULL CHECK (mime LIKE 'image/%'),
    data       BLOB NOT NULL,
    sha256     TEXT NOT NULL UNIQUE,                        -- 같은 이미지 중복 저장 방지
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- ---------------------------------------------------------------------
-- 3. 용어
-- ---------------------------------------------------------------------
CREATE TABLE terms (
    id            TEXT PRIMARY KEY,                         -- slug: 'nms', 'cooks-distance'
    term_en       TEXT NOT NULL,                            -- Non-Maximum Suppression
    term_ko       TEXT,                                     -- 비최대 억제
    abbreviation  TEXT,                                     -- NMS
    definition    TEXT NOT NULL,                            -- 한 줄 정의
    usage_example TEXT,                                     -- 실무 예문: "이렇게 말한다"
    origin        TEXT NOT NULL DEFAULT 'standard'
                  CHECK (origin IN ('standard','project')), -- 업계 표준 / Diagnostics 정의 지표
    track_id      TEXT REFERENCES tracks(id),               -- 주 소속 트랙
    status        TEXT NOT NULL DEFAULT 'draft'
                  CHECK (status IN ('draft','reviewed','retired')),
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- ---------------------------------------------------------------------
-- 4. 문항 (본체)
-- ---------------------------------------------------------------------
CREATE TABLE questions (
    id                TEXT PRIMARY KEY,                     -- '{트랙}-{장코드}-{일련번호}' 예: 'F-NMS-003'
    type              TEXT NOT NULL CHECK (type IN ('mcq','short','scenario')),
                                                            -- 5지선다 / 단답형 / 지시문 해석형
    chapter_id        TEXT NOT NULL REFERENCES chapters(id),
    cognitive_level   INTEGER NOT NULL DEFAULT 1 CHECK (cognitive_level BETWEEN 1 AND 3),
                                                            -- 1 용어 인지 / 2 개념 구분 / 3 지시 해석
    difficulty        INTEGER NOT NULL DEFAULT 2 CHECK (difficulty BETWEEN 1 AND 5),
    stem              TEXT NOT NULL CHECK (trim(stem) <> ''),   -- 문제 본문 (지시문 해석형은 지시문+질문)
    image_asset_id    INTEGER REFERENCES assets(id) ON DELETE SET NULL,
    answer_choice     INTEGER,                              -- 객관식 정답 번호(1~5), 단답형은 NULL
    explanation_short TEXT NOT NULL CHECK (trim(explanation_short) <> ''), -- 오답 시 자동 표시
    explanation_full  TEXT,                                 -- "더 보기"로 펼치는 상세 해설
    source_ref        TEXT,                                 -- 출처 문서#섹션
    status            TEXT NOT NULL DEFAULT 'draft'
                      CHECK (status IN ('draft','reviewed','retired')),
    version           INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
    breaking_version  INTEGER NOT NULL DEFAULT 1,           -- 정답이 바뀐 마지막 버전 → 학습 앱이 복습 일정 초기화 판단
    reviewed_by       TEXT,
    reviewed_at       TEXT,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    CHECK ( (type = 'short' AND answer_choice IS NULL)
         OR (type IN ('mcq','scenario') AND answer_choice IS NOT NULL
                                        AND answer_choice BETWEEN 1 AND 5) ),
                                                            -- IS NOT NULL 필수: SQLite CHECK는 NULL 결과를 통과시킴
    CHECK (breaking_version BETWEEN 1 AND version)
);

CREATE INDEX idx_questions_chapter ON questions(chapter_id);
CREATE INDEX idx_questions_status  ON questions(status);
CREATE INDEX idx_questions_type    ON questions(type);

-- 4-1. 객관식·지시문 해석형 보기 (문항당 정확히 5개 — 검수 게이트에서 강제)
CREATE TABLE choices (
    question_id TEXT    NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL CHECK (idx BETWEEN 1 AND 5),
    text        TEXT    NOT NULL CHECK (trim(text) <> ''),
    rationale   TEXT,                                       -- 이 보기가 왜 틀렸는지 (정답 보기는 NULL 가능)
    PRIMARY KEY (question_id, idx)
);

-- 4-2. 단답형 허용 답 (영문 또는 한글 중 하나만 일치하면 정답)
CREATE TABLE accepted_answers (
    id          INTEGER PRIMARY KEY,
    question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    lang        TEXT NOT NULL CHECK (lang IN ('en','ko')),
    text        TEXT NOT NULL CHECK (trim(text) <> ''),     -- 표시용 원문: 'Non-Maximum Suppression'
    normalized  TEXT NOT NULL,                              -- 채점용 정규화 값: 'nonmaximumsuppression'
    is_primary  INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0,1)), -- 결과 화면에 "정답"으로 보여줄 대표 답
    UNIQUE (question_id, lang, normalized)
);
CREATE INDEX idx_answers_question ON accepted_answers(question_id);

-- 4-3. 태그
CREATE TABLE question_tags (
    question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    tag         TEXT NOT NULL,
    PRIMARY KEY (question_id, tag)
);
CREATE INDEX idx_tags_tag ON question_tags(tag);

-- 4-4. 문항 ↔ 용어 연결
CREATE TABLE question_terms (
    question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    term_id     TEXT NOT NULL REFERENCES terms(id),
    PRIMARY KEY (question_id, term_id)
);
CREATE INDEX idx_qterms_term ON question_terms(term_id);

-- ---------------------------------------------------------------------
-- 5. 변경 이력 (git 이력 대체) — Studio가 저장 트랜잭션 안에서 "수정 전" 스냅샷을 기록
-- ---------------------------------------------------------------------
CREATE TABLE change_log (
    id            INTEGER PRIMARY KEY,
    entity_type   TEXT NOT NULL CHECK (entity_type IN ('question','term')),
    entity_id     TEXT NOT NULL,
    version       INTEGER,                                  -- 스냅샷 당시 문항 버전
    action        TEXT NOT NULL CHECK (action IN ('create','update','status','retire','delete')),
    snapshot_json TEXT NOT NULL CHECK (json_valid(snapshot_json)), -- 보기·허용답·태그 포함 전체 상태
    note          TEXT,                                     -- 변경 사유 (선택)
    changed_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX idx_changelog_entity ON change_log(entity_type, entity_id, changed_at);

-- ---------------------------------------------------------------------
-- 6. 팩 내보내기 기록
-- ---------------------------------------------------------------------
CREATE TABLE pack_exports (
    id             INTEGER PRIMARY KEY,
    pack_version   TEXT NOT NULL UNIQUE,                    -- 예: '2026.10.01'
    question_count INTEGER NOT NULL,
    term_count     INTEGER NOT NULL,
    sha256         TEXT NOT NULL,
    exported_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    note           TEXT
);

-- =====================================================================
-- 트리거
-- =====================================================================

-- updated_at 자동 갱신
CREATE TRIGGER trg_questions_touch AFTER UPDATE ON questions
WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE questions SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = NEW.id;
END;

CREATE TRIGGER trg_terms_touch AFTER UPDATE ON terms
WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE terms SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = NEW.id;
END;

-- 새 문항은 반드시 draft로 생성 (보기·허용답을 넣은 뒤 검수 게이트를 통과해야 reviewed)
CREATE TRIGGER trg_questions_insert_draft BEFORE INSERT ON questions
WHEN NEW.status <> 'draft'
BEGIN
    SELECT RAISE(ABORT, '새 문항은 draft 상태로만 생성할 수 있습니다');
END;

-- 검수 게이트: reviewed로 바꾸기 전 필수 조건 확인
CREATE TRIGGER trg_questions_review_gate BEFORE UPDATE OF status ON questions
WHEN NEW.status = 'reviewed' AND OLD.status <> 'reviewed'
BEGIN
    SELECT RAISE(ABORT, '객관식/지시문 문항은 보기가 정확히 5개여야 합니다')
     WHERE NEW.type IN ('mcq','scenario')
       AND (SELECT COUNT(*) FROM choices WHERE question_id = NEW.id) <> 5;

    SELECT RAISE(ABORT, '오답 보기마다 틀린 이유(rationale)가 필요합니다')
     WHERE NEW.type IN ('mcq','scenario')
       AND EXISTS (SELECT 1 FROM choices
                    WHERE question_id = NEW.id AND idx <> NEW.answer_choice
                      AND (rationale IS NULL OR trim(rationale) = ''));

    SELECT RAISE(ABORT, '단답형은 허용 답이 1개 이상 필요합니다')
     WHERE NEW.type = 'short'
       AND NOT EXISTS (SELECT 1 FROM accepted_answers WHERE question_id = NEW.id);

    SELECT RAISE(ABORT, '단답형은 대표 정답(is_primary)이 1개 필요합니다')
     WHERE NEW.type = 'short'
       AND (SELECT COUNT(*) FROM accepted_answers
             WHERE question_id = NEW.id AND is_primary = 1) <> 1;

    SELECT RAISE(ABORT, '검수자와 검수일을 기록해야 합니다')
     WHERE NEW.reviewed_by IS NULL OR NEW.reviewed_at IS NULL;
END;

-- 삭제 규칙: draft만 완전 삭제 가능, 그 외는 retired로 처리
CREATE TRIGGER trg_questions_delete_guard BEFORE DELETE ON questions
WHEN OLD.status <> 'draft'
BEGIN
    SELECT RAISE(ABORT, '검수된 문항은 삭제할 수 없습니다. retired로 변경하세요');
END;

-- retired 문항은 다시 draft로 되돌릴 수 없음 (필요하면 새 문항으로 복제)
CREATE TRIGGER trg_questions_no_unretire BEFORE UPDATE OF status ON questions
WHEN OLD.status = 'retired' AND NEW.status <> 'retired'
BEGIN
    SELECT RAISE(ABORT, 'retired 문항은 되살릴 수 없습니다. 복제해서 새 문항으로 만드세요');
END;

-- 문항이 쓰는 용어는 삭제 불가 (FK로도 막히지만 메시지를 명확히)
CREATE TRIGGER trg_terms_delete_guard BEFORE DELETE ON terms
WHEN EXISTS (SELECT 1 FROM question_terms WHERE term_id = OLD.id)
BEGIN
    SELECT RAISE(ABORT, '문항에 연결된 용어는 삭제할 수 없습니다. retired로 변경하세요');
END;

-- =====================================================================
-- 뷰
-- =====================================================================

-- 팩에 들어갈 문항 (reviewed만)
CREATE VIEW v_pack_questions AS
SELECT q.*, c.track_id
  FROM questions q JOIN chapters c ON c.id = q.chapter_id
 WHERE q.status = 'reviewed';

-- 문항별 '프로젝트 정의 용어' 배지 여부 (연결 용어 중 하나라도 project면 1)
CREATE VIEW v_question_badges AS
SELECT q.id AS question_id,
       MAX(CASE WHEN t.origin = 'project' THEN 1 ELSE 0 END) AS has_project_term
  FROM questions q
  LEFT JOIN question_terms qt ON qt.question_id = q.id
  LEFT JOIN terms t ON t.id = qt.term_id
 GROUP BY q.id;

-- Studio 통계: 트랙·유형·상태별 문항 수
CREATE VIEW v_bank_stats AS
SELECT c.track_id, q.type, q.status, COUNT(*) AS n
  FROM questions q JOIN chapters c ON c.id = q.chapter_id
 GROUP BY c.track_id, q.type, q.status;

-- =====================================================================
-- 기본 데이터
-- =====================================================================
INSERT INTO bank_meta(key, value) VALUES
  ('schema_version', '1'),
  ('bank_name', 'VisionDrill Core');

INSERT INTO tracks(id, name_ko, name_en, sort_order) VALUES
  ('A', '통계 기초',            'Statistics Fundamentals',          1),
  ('B', '회귀분석·진단',        'Regression & Diagnostics',         2),
  ('C', 'ML 기초',              'Machine Learning Fundamentals',    3),
  ('D', '영상·고전 CV',         'Imaging & Classical CV',           4),
  ('E', '딥러닝 기초·아키텍처', 'Deep Learning & Architectures',    5),
  ('F', '태스크·지표',          'Vision Tasks & Metrics',           6),
  ('G', '학습·데이터 실무',     'Training & Data Practice',         7),
  ('H', '비전 모델 진단',       'Vision Model Diagnostics',         8);

INSERT INTO track_prereqs(track_id, prereq_track_id) VALUES
  ('B','A'), ('C','B'), ('E','D'), ('F','E'), ('G','F'),
  ('H','B'), ('H','F');
