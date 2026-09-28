import type { SqlJsStatic } from 'sql.js';
import { BankError, rowToQuestion } from './bank';
import { gradeQuestion, type Grade, type Response } from './grading';
import { copyRows, createContentDb, readPackManifest, type PackManifest } from './pack';
import { LEARNER_SCHEMA_SQL } from './schema.generated';
import { Db, nowIso, type Row } from './sql';
import type { Question, Term, Track } from './types';

export interface LearnerSettings {
  sessionSize: number; // 한 세션 문항 수
  newPerSession: number; // 세션당 신규 문항 최대 수
  shuffleChoices: boolean; // 보기 순서 섞기
  guessedToWrongNote: boolean; // 찍어서 맞힌 문항도 오답노트에 넣기
  graduateStreak: number; // 오답노트 졸업에 필요한 연속 정답 수
  unlockAll: boolean; // 트랙 해금 무시 (관리자/테스트용)
}

export const DEFAULT_SETTINGS: LearnerSettings = {
  sessionSize: 20,
  newPerSession: 10,
  shuffleChoices: true,
  guessedToWrongNote: true,
  graduateStreak: 3,
  unlockAll: false,
};

/** Leitner 박스별 다음 복습까지 간격(일). 박스 1 = 다음 세션에 바로 재출제 */
export const BOX_INTERVAL_DAYS: Record<number, number> = { 1: 0, 2: 1, 3: 3, 4: 7, 5: 16, 6: 35 };
/** 이 박스 이상이면 "숙련"으로 본다 (트랙 해금 기준 계산용) */
export const MASTERED_BOX = 3;

export type SessionMode = 'daily' | 'wrong_note' | 'track';
export type ResultKind = 'correct' | 'guessed' | 'wrong';

export interface SessionItem {
  question: Question;
  /** 화면에 보여줄 보기 순서 (원래 번호 배열) */
  order: number[];
  terms: Term[];
  isNew: boolean;
}

export interface ItemResult {
  item: SessionItem;
  attemptId: number;
  response?: Response;
  guessed: boolean;
  grade: Grade;
  result: ResultKind;
}

export interface SessionResult {
  sessionId: number;
  total: number;
  correct: number;
  durationSec: number;
  items: ItemResult[];
}

export interface TrackStatus extends Track {
  total: number;
  seen: number;
  mastered: number;
  mastery: number;
  unlocked: boolean;
  prereqs: { track_id: string; min_mastery: number; mastery: number }[];
}

export interface WrongNoteItem {
  question: Question;
  reason: 'unknown' | 'confused' | 'mistake' | null;
  memo: string | null;
  added_at: string;
  updated_at: string;
  wrong_count: number;
  streak: number;
  last_response: string | null;
}

interface PrevState {
  progress: Row | null;
  note: Row | null;
}

export class Learner {
  constructor(public readonly db: Db) {}

  static create(SQL: SqlJsStatic): Learner {
    const db = createContentDb(SQL);
    db.exec(LEARNER_SCHEMA_SQL);
    db.run(`DELETE FROM bank_meta`);
    db.run(`INSERT INTO learner_meta(key,value) VALUES('kind','learner'),('created_at',?)`, [nowIso()]);
    return new Learner(db);
  }

  static open(SQL: SqlJsStatic, bytes: Uint8Array): Learner {
    const db = Db.create(SQL, bytes);
    const kind = db.value<string>(`SELECT value FROM learner_meta WHERE key='kind'`);
    if (kind !== 'learner') throw new BankError('학습 기록 파일이 아닙니다');
    return new Learner(db);
  }

  export(): Uint8Array {
    return this.db.export();
  }

  // ------------------------------------------------------------------ 설정
  settings(): LearnerSettings {
    const raw = this.db.value<string>(`SELECT value FROM learner_meta WHERE key='settings'`);
    return { ...DEFAULT_SETTINGS, ...(raw ? JSON.parse(raw) : {}) };
  }

  saveSettings(s: Partial<LearnerSettings>): LearnerSettings {
    const next = { ...this.settings(), ...s };
    this.db.run(`INSERT INTO learner_meta(key,value) VALUES('settings',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value`, [
      JSON.stringify(next),
    ]);
    return next;
  }

  // ------------------------------------------------------------------ 팩
  packInfo(): PackManifest | undefined {
    try {
      return readPackManifest(this.db);
    } catch {
      return undefined;
    }
  }

  /**
   * 문제 팩 가져오기: 콘텐츠 테이블을 통째로 교체하고 학습 기록은 유지한다.
   * 정답이 바뀐 문항(breaking_version > 마지막으로 푼 버전)은 복습 일정을 초기화한다.
   */
  importPack(SQL: SqlJsStatic, packBytes: Uint8Array) {
    const pack = Db.create(SQL, packBytes);
    try {
      const manifest = readPackManifest(pack);
      const before = new Set(this.db.all<{ id: string }>(`SELECT id FROM questions`).map((r) => r.id));
      const result = { manifest, added: 0, updated: 0, reset: 0, removed: 0 };
      this.db.tx(() => {
        for (const t of [
          'question_terms',
          'question_tags',
          'accepted_answers',
          'choices',
          'questions',
          'terms',
          'assets',
          'chapters',
          'track_prereqs',
          'tracks',
          'bank_meta',
        ])
          this.db.run(`DELETE FROM ${t}`);
        for (const t of [
          'bank_meta',
          'tracks',
          'track_prereqs',
          'chapters',
          'assets',
          'terms',
          'questions',
          'choices',
          'accepted_answers',
          'question_tags',
          'question_terms',
        ])
          copyRows(pack, this.db, t);
        const after = this.db.all<{ id: string; version: number; breaking_version: number }>(
          `SELECT id, version, breaking_version FROM questions`,
        );
        const afterIds = new Set(after.map((r) => r.id));
        for (const q of after) {
          if (!before.has(q.id)) result.added++;
          const p = this.db.get<{ seen_version: number }>(`SELECT seen_version FROM progress WHERE question_id=?`, [q.id]);
          if (p && p.seen_version < q.breaking_version) {
            this.db.run(`UPDATE progress SET box=1, streak=0, due_at=?, seen_version=? WHERE question_id=?`, [nowIso(), q.version, q.id]);
            result.reset++;
          }
        }
        for (const id of before) if (!afterIds.has(id)) result.removed++;
        result.updated = [...before].filter((id) => afterIds.has(id)).length;
        this.db.run(
          `INSERT INTO learner_meta(key,value) VALUES('pack_imported_at',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
          [nowIso()],
        );
      });
      return result;
    } finally {
      pack.close();
    }
  }

  /**
   * 앱에 내장된 기본 팩 적용.
   * - 설치된 팩이 없으면 설치
   * - 지금 설치된 팩이 예전에 앱이 넣어 준 기본 팩이고 새 기본 팩과 버전이 다르면 교체 (앱 업데이트로 문항이 늘어난 경우)
   * - 사용자가 직접 가져온 팩을 쓰고 있으면 건드리지 않음
   * 교체했으면 importPack 결과, 아니면 null.
   */
  applyBundledPack(SQL: SqlJsStatic, packBytes: Uint8Array) {
    const pack = Db.create(SQL, packBytes);
    let bundled: PackManifest;
    try {
      bundled = readPackManifest(pack);
    } finally {
      pack.close();
    }
    const current = this.packInfo()?.pack_version;
    const lastBundled =
      this.db.value<string>(`SELECT value FROM learner_meta WHERE key='bundled_pack'`) ??
      (current?.startsWith('sample-') ? current : undefined); // 0.1.0 이전 설치본은 샘플 팩이 기본 팩이었음
    const shouldApply = !current || (current === lastBundled && current !== bundled.pack_version);
    if (!shouldApply) return null;
    const r = this.importPack(SQL, packBytes);
    this.db.run(
      `INSERT INTO learner_meta(key,value) VALUES('bundled_pack',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
      [bundled.pack_version],
    );
    return r;
  }

  // ------------------------------------------------------------------ 트랙 숙련도·해금
  trackStatus(): TrackStatus[] {
    const tracks = this.db.all<Track & { total: number; seen: number; mastered: number }>(
      `SELECT t.*,
              COUNT(q.id) AS total,
              COUNT(p.question_id) AS seen,
              SUM(CASE WHEN p.box >= ${MASTERED_BOX} THEN 1 ELSE 0 END) AS mastered
         FROM tracks t
         LEFT JOIN chapters c ON c.track_id = t.id
         LEFT JOIN questions q ON q.chapter_id = c.id
         LEFT JOIN progress p ON p.question_id = q.id
        GROUP BY t.id ORDER BY t.sort_order`,
    );
    const prereqs = this.db.all<{ track_id: string; prereq_track_id: string; min_mastery: number }>(`SELECT * FROM track_prereqs`);
    const unlockAll = this.settings().unlockAll;
    const mastery = new Map(tracks.map((t) => [t.id, t.total ? (t.mastered ?? 0) / t.total : 1]));
    return tracks.map((t) => {
      const ps = prereqs
        .filter((p) => p.track_id === t.id)
        .map((p) => ({ track_id: p.prereq_track_id, min_mastery: p.min_mastery, mastery: mastery.get(p.prereq_track_id) ?? 1 }));
      return {
        ...t,
        mastered: t.mastered ?? 0,
        mastery: mastery.get(t.id) ?? 0,
        prereqs: ps,
        unlocked: unlockAll || ps.every((p) => p.mastery >= p.min_mastery),
      };
    });
  }

  // ------------------------------------------------------------------ 세션 구성
  private item(r: Row, shuffle: boolean): SessionItem {
    const question = rowToQuestion(this.db, r);
    const order = question.choices.map((c) => c.idx);
    if (shuffle && question.type !== 'short') {
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
    }
    const terms = this.db.all<Term>(
      `SELECT t.* FROM terms t JOIN question_terms qt ON qt.term_id=t.id WHERE qt.question_id=? ORDER BY t.term_en`,
      [question.id],
    );
    const isNew = !this.db.value(`SELECT 1 FROM progress WHERE question_id=?`, [question.id]);
    return { question, order, terms, isNew };
  }

  /**
   * 세션 문항 구성.
   * daily: 복습 예정(오래 기다린 순) → 신규(커리큘럼 순, newPerSession개까지) → 남으면 가까운 복습 예정
   * wrong_note: 오답노트(졸업 전) 문항
   * track: 특정 트랙에서 신규 → 복습 예정 → 나머지
   */
  buildSession(mode: SessionMode = 'daily', trackId?: string): SessionItem[] {
    const s = this.settings();
    const unlocked = this.trackStatus()
      .filter((t) => t.unlocked)
      .map((t) => t.id);
    if (!unlocked.length) return [];
    const inTracks = `c.track_id IN (${unlocked.map(() => '?').join(',')})`;
    const now = nowIso();
    const base = `SELECT q.*, p.due_at, p.box FROM questions q JOIN chapters c ON c.id=q.chapter_id
                  JOIN tracks t ON t.id=c.track_id LEFT JOIN progress p ON p.question_id=q.id`;
    const curriculum = `t.sort_order, c.sort_order, q.id`;
    let rows: Row[] = [];

    if (mode === 'wrong_note') {
      rows = this.db.all(`${base} JOIN wrong_notes w ON w.question_id=q.id WHERE w.graduated_at IS NULL ORDER BY w.updated_at LIMIT ?`, [
        s.sessionSize,
      ]);
    } else if (mode === 'track') {
      if (!trackId || !unlocked.includes(trackId)) return [];
      rows = this.db.all(
        `${base} WHERE c.track_id=?
          ORDER BY (p.question_id IS NOT NULL), (p.due_at > ?), p.due_at, ${curriculum} LIMIT ?`,
        [trackId, now, s.sessionSize],
      );
    } else {
      const due = this.db.all(`${base} WHERE ${inTracks} AND p.question_id IS NOT NULL AND p.due_at <= ? ORDER BY p.due_at LIMIT ?`, [
        ...unlocked,
        now,
        s.sessionSize,
      ]);
      const room = s.sessionSize - due.length;
      const fresh =
        room > 0
          ? this.db.all(`${base} WHERE ${inTracks} AND p.question_id IS NULL ORDER BY ${curriculum} LIMIT ?`, [
              ...unlocked,
              Math.min(room, s.newPerSession),
            ])
          : [];
      const room2 = s.sessionSize - due.length - fresh.length;
      const upcoming =
        room2 > 0
          ? this.db.all(`${base} WHERE ${inTracks} AND p.question_id IS NOT NULL AND p.due_at > ? ORDER BY p.due_at LIMIT ?`, [
              ...unlocked,
              now,
              room2,
            ])
          : [];
      rows = [...due, ...fresh, ...upcoming];
      // 복습과 신규가 섞이도록 순서 섞기
      for (let i = rows.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [rows[i], rows[j]] = [rows[j], rows[i]];
      }
    }
    return rows.map((r) => this.item(r, s.shuffleChoices));
  }

  // ------------------------------------------------------------------ 제출·채점
  submitSession(
    mode: SessionMode,
    items: SessionItem[],
    responses: Record<string, Response | undefined>,
    guessed: Record<string, boolean>,
    startedAt: string,
  ): SessionResult {
    const s = this.settings();
    const submittedAt = nowIso();
    const durationSec = Math.max(0, Math.round((Date.parse(submittedAt) - Date.parse(startedAt)) / 1000));
    return this.db.tx(() => {
      this.db.run(`INSERT INTO sessions(started_at,submitted_at,total,correct,duration_sec,mode) VALUES(?,?,?,0,?,?)`, [
        startedAt,
        submittedAt,
        items.length,
        durationSec,
        mode,
      ]);
      const sessionId = this.db.value<number>(`SELECT last_insert_rowid()`)!;
      let correct = 0;
      const out: ItemResult[] = [];
      for (const item of items) {
        const q = item.question;
        const r = responses[q.id];
        const grade = gradeQuestion(q, r);
        const isGuessed = !!guessed[q.id];
        const result: ResultKind = grade.correct ? (isGuessed ? 'guessed' : 'correct') : 'wrong';
        if (grade.correct) correct++;
        const prev = this.snapshot(q.id);
        this.apply(q.id, q.version, result, s);
        const responseText = r ? (r.kind === 'choice' ? (r.idx == null ? null : String(r.idx)) : r.text) : null;
        this.db.run(
          `INSERT INTO attempts(session_id,question_id,question_version,response,is_correct,guessed,near_miss,prev_state,answered_at)
           VALUES(?,?,?,?,?,?,?,?,?)`,
          [
            sessionId,
            q.id,
            q.version,
            responseText,
            grade.correct ? 1 : 0,
            isGuessed ? 1 : 0,
            grade.nearMiss ? 1 : 0,
            JSON.stringify(prev),
            submittedAt,
          ],
        );
        const attemptId = this.db.value<number>(`SELECT last_insert_rowid()`)!;
        out.push({ item, attemptId, response: r, guessed: isGuessed, grade, result });
      }
      this.db.run(`UPDATE sessions SET correct=? WHERE id=?`, [correct, sessionId]);
      return { sessionId, total: items.length, correct, durationSec, items: out };
    });
  }

  /** 오타 의심 답을 사용자가 정답으로 인정: 채점 전 상태로 되돌린 뒤 정답으로 다시 반영 */
  acceptNearMiss(attemptId: number): void {
    const s = this.settings();
    this.db.tx(() => {
      const a = this.db.get<{
        session_id: number;
        question_id: string;
        question_version: number;
        near_miss: number;
        is_correct: number;
        guessed: number;
        prev_state: string;
      }>(`SELECT * FROM attempts WHERE id=?`, [attemptId]);
      if (!a || !a.near_miss || a.is_correct) return;
      const prev = JSON.parse(a.prev_state) as PrevState;
      this.restore(a.question_id, prev);
      this.apply(a.question_id, a.question_version, a.guessed ? 'guessed' : 'correct', s);
      this.db.run(`UPDATE attempts SET is_correct=1, overridden=1 WHERE id=?`, [attemptId]);
      this.db.run(`UPDATE sessions SET correct=correct+1 WHERE id=?`, [a.session_id]);
    });
  }

  private snapshot(qid: string): PrevState {
    return {
      progress: this.db.get<Row>(`SELECT * FROM progress WHERE question_id=?`, [qid]) ?? null,
      note: this.db.get<Row>(`SELECT * FROM wrong_notes WHERE question_id=?`, [qid]) ?? null,
    };
  }

  private restore(qid: string, prev: PrevState) {
    this.db.run(`DELETE FROM progress WHERE question_id=?`, [qid]);
    this.db.run(`DELETE FROM wrong_notes WHERE question_id=?`, [qid]);
    const put = (table: string, row: Row | null) => {
      if (!row) return;
      const cols = Object.keys(row);
      this.db.run(
        `INSERT INTO ${table}(${cols.join(',')}) VALUES(${cols.map(() => '?').join(',')})`,
        cols.map((c) => row[c]),
      );
    };
    put('progress', prev.progress);
    put('wrong_notes', prev.note);
  }

  private apply(qid: string, version: number, result: ResultKind, s: LearnerSettings) {
    const now = new Date();
    const p = this.db.get<{ box: number; streak: number }>(`SELECT box, streak FROM progress WHERE question_id=?`, [qid]);
    let box = p?.box ?? 1;
    let streak = p?.streak ?? 0;
    if (result === 'correct') {
      box = Math.min(box + 1, 6);
      streak += 1;
    } else if (result === 'guessed') {
      streak = 0; // 찍은 정답은 박스 유지, 연속 정답 초기화
    } else {
      box = 1;
      streak = 0;
    }
    const due = new Date(now.getTime() + BOX_INTERVAL_DAYS[box] * 86400000).toISOString();
    this.db.run(
      `INSERT INTO progress(question_id,box,due_at,seen_count,correct_count,wrong_count,streak,last_result,last_answered_at,seen_version)
       VALUES(?,?,?,1,?,?,?,?,?,?)
       ON CONFLICT(question_id) DO UPDATE SET box=excluded.box, due_at=excluded.due_at, seen_count=seen_count+1,
         correct_count=correct_count+excluded.correct_count, wrong_count=wrong_count+excluded.wrong_count,
         streak=excluded.streak, last_result=excluded.last_result, last_answered_at=excluded.last_answered_at,
         seen_version=excluded.seen_version`,
      [qid, box, due, result === 'wrong' ? 0 : 1, result === 'wrong' ? 1 : 0, streak, result, now.toISOString(), version],
    );
    const ts = now.toISOString();
    const toNote = result === 'wrong' || (result === 'guessed' && s.guessedToWrongNote);
    if (toNote) {
      this.db.run(
        `INSERT INTO wrong_notes(question_id,added_at,updated_at) VALUES(?,?,?)
         ON CONFLICT(question_id) DO UPDATE SET updated_at=excluded.updated_at, graduated_at=NULL`,
        [qid, ts, ts],
      );
    } else if (result === 'correct' && streak >= s.graduateStreak) {
      this.db.run(`UPDATE wrong_notes SET graduated_at=?, updated_at=? WHERE question_id=? AND graduated_at IS NULL`, [ts, ts, qid]);
    }
  }

  // ------------------------------------------------------------------ 오답노트
  wrongNotes(includeGraduated = false): WrongNoteItem[] {
    const rows = this.db.all<Row>(
      `SELECT q.*, w.reason AS w_reason, w.memo AS w_memo, w.added_at AS w_added, w.updated_at AS w_updated,
              COALESCE(p.wrong_count,0) AS p_wrong, COALESCE(p.streak,0) AS p_streak,
              (SELECT response FROM attempts a WHERE a.question_id=q.id AND a.is_correct=0 ORDER BY a.id DESC LIMIT 1) AS last_resp
         FROM wrong_notes w JOIN questions q ON q.id=w.question_id LEFT JOIN progress p ON p.question_id=q.id
        WHERE ${includeGraduated ? '1' : 'w.graduated_at IS NULL'}
        ORDER BY w.updated_at DESC`,
    );
    return rows.map((r) => ({
      question: rowToQuestion(this.db, r),
      reason: (r.w_reason as WrongNoteItem['reason']) ?? null,
      memo: (r.w_memo as string) ?? null,
      added_at: r.w_added as string,
      updated_at: r.w_updated as string,
      wrong_count: r.p_wrong as number,
      streak: r.p_streak as number,
      last_response: (r.last_resp as string) ?? null,
    }));
  }

  updateWrongNote(qid: string, patch: { reason?: WrongNoteItem['reason']; memo?: string | null }) {
    const cur = this.db.get<{ reason: string | null; memo: string | null }>(`SELECT reason, memo FROM wrong_notes WHERE question_id=?`, [
      qid,
    ]);
    if (!cur) return;
    this.db.run(`UPDATE wrong_notes SET reason=?, memo=?, updated_at=? WHERE question_id=?`, [
      patch.reason !== undefined ? patch.reason : cur.reason,
      patch.memo !== undefined ? patch.memo : cur.memo,
      nowIso(),
      qid,
    ]);
  }

  graduateWrongNote(qid: string) {
    this.db.run(`UPDATE wrong_notes SET graduated_at=? WHERE question_id=?`, [nowIso(), qid]);
  }

  // ------------------------------------------------------------------ 통계
  stats() {
    const now = nowIso();
    const q = (sql: string, p: any[] = []) => this.db.value<number>(sql, p) ?? 0;
    const days = this.db
      .all<{ d: string }>(`SELECT DISTINCT substr(submitted_at,1,10) AS d FROM sessions WHERE submitted_at IS NOT NULL ORDER BY d DESC`)
      .map((r) => r.d);
    let streakDays = 0;
    const cursor = new Date();
    const fmt = (d: Date) => d.toISOString().slice(0, 10);
    if (days[0] !== fmt(cursor)) cursor.setUTCDate(cursor.getUTCDate() - 1); // 오늘 아직 안 했으면 어제부터 셈
    for (const d of days) {
      if (d === fmt(cursor)) {
        streakDays++;
        cursor.setUTCDate(cursor.getUTCDate() - 1);
      } else break;
    }
    const weekAgo = new Date(Date.now() - 7 * 86400000).toISOString();
    return {
      questions: q(`SELECT COUNT(*) FROM questions`),
      seen: q(`SELECT COUNT(*) FROM progress WHERE question_id IN (SELECT id FROM questions)`),
      due: q(`SELECT COUNT(*) FROM progress WHERE due_at <= ? AND question_id IN (SELECT id FROM questions)`, [now]),
      wrongNotes: q(`SELECT COUNT(*) FROM wrong_notes WHERE graduated_at IS NULL AND question_id IN (SELECT id FROM questions)`),
      sessions: q(`SELECT COUNT(*) FROM sessions WHERE submitted_at IS NOT NULL`),
      streakDays,
      weekAccuracy: (() => {
        const r = this.db.get<{ t: number; c: number }>(`SELECT COUNT(*) AS t, SUM(is_correct) AS c FROM attempts WHERE answered_at >= ?`, [
          weekAgo,
        ]);
        return r && r.t ? (r.c ?? 0) / r.t : null;
      })(),
      recent: this.db.all<{ id: number; submitted_at: string; total: number; correct: number; duration_sec: number; mode: string }>(
        `SELECT id, submitted_at, total, correct, duration_sec, mode FROM sessions WHERE submitted_at IS NOT NULL ORDER BY id DESC LIMIT 10`,
      ),
    };
  }

  glossary(): (Term & { question_count: number })[] {
    return this.db.all(
      `SELECT t.*, (SELECT COUNT(*) FROM question_terms qt WHERE qt.term_id=t.id) AS question_count
         FROM terms t WHERE t.status <> 'retired' ORDER BY lower(t.term_en)`,
    );
  }
}
