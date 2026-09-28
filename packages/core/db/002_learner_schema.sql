-- =====================================================================
-- VisionDrill 학습 앱(Drill) 학습 기록 스키마 v0.1
-- 학습 앱 DB = 문제은행 스키마(콘텐츠 테이블, 트리거 제거) + 아래 학습 기록 테이블
-- 콘텐츠 테이블은 팩을 가져올 때마다 통째로 교체되고, 학습 기록은 문항 id 기준으로 유지된다.
-- =====================================================================

CREATE TABLE learner_meta (
    key   TEXT PRIMARY KEY,          -- pack_version, pack_imported_at, settings_json ...
    value TEXT NOT NULL
);

-- 문항별 학습 상태 (MVP: Leitner 박스, Phase 2에서 FSRS로 교체 예정)
CREATE TABLE progress (
    question_id      TEXT PRIMARY KEY,           -- 콘텐츠 교체와 무관하게 유지 (FK 없음)
    box              INTEGER NOT NULL DEFAULT 1 CHECK (box BETWEEN 1 AND 6),
    due_at           TEXT NOT NULL,              -- 다음 복습 시각
    seen_count       INTEGER NOT NULL DEFAULT 0,
    correct_count    INTEGER NOT NULL DEFAULT 0,
    wrong_count      INTEGER NOT NULL DEFAULT 0,
    streak           INTEGER NOT NULL DEFAULT 0, -- 연속 정답(찍은 정답 제외)
    last_result      TEXT CHECK (last_result IN ('correct','guessed','wrong')),
    last_answered_at TEXT,
    seen_version     INTEGER NOT NULL DEFAULT 1  -- 마지막으로 푼 문항 버전 (breaking_version 비교용)
);

-- 오답노트
CREATE TABLE wrong_notes (
    question_id  TEXT PRIMARY KEY,
    reason       TEXT CHECK (reason IN ('unknown','confused','mistake')), -- 몰랐음/헷갈림/실수
    memo         TEXT,
    added_at     TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    graduated_at TEXT                             -- N회 연속 정답으로 졸업한 시각 (NULL = 아직 오답노트에 있음)
);

CREATE TABLE sessions (
    id           INTEGER PRIMARY KEY,
    started_at   TEXT NOT NULL,
    submitted_at TEXT,
    total        INTEGER NOT NULL,
    correct      INTEGER,
    duration_sec INTEGER,
    mode         TEXT NOT NULL DEFAULT 'daily' CHECK (mode IN ('daily','wrong_note','track'))
);

CREATE TABLE attempts (
    id               INTEGER PRIMARY KEY,
    session_id       INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    question_id      TEXT NOT NULL,
    question_version INTEGER NOT NULL,
    response         TEXT,                        -- 객관식: 보기 번호(원래 순서 기준), 단답형: 입력 문자열
    is_correct       INTEGER NOT NULL CHECK (is_correct IN (0,1)),
    guessed          INTEGER NOT NULL DEFAULT 0 CHECK (guessed IN (0,1)),
    near_miss        INTEGER NOT NULL DEFAULT 0 CHECK (near_miss IN (0,1)),
    overridden       INTEGER NOT NULL DEFAULT 0 CHECK (overridden IN (0,1)), -- 오타 의심을 사용자가 정답 인정
    prev_state       TEXT,                        -- 채점 반영 전 progress/오답노트 상태(JSON). 오타 인정 시 되돌리는 데 사용
    answered_at      TEXT NOT NULL
);
CREATE INDEX idx_attempts_question ON attempts(question_id);
CREATE INDEX idx_attempts_session  ON attempts(session_id);
