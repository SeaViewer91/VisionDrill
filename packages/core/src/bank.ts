import type { SqlJsStatic } from 'sql.js';
import { normalize } from './grading';
import { BANK_SCHEMA_SQL } from './schema.generated';
import { Db, nowIso, sha256Hex, type Row } from './sql';
import type { Answer, Chapter, ChangeLogEntry, Question, QuestionFilter, QuestionSummary, Status, Term, Track, TrackPrereq } from './types';
import { hasErrors, validateQuestion } from './validate';

export const BANK_SCHEMA_VERSION = '1';

export class BankError extends Error {}

/** SQLite 트리거의 RAISE 메시지를 사람이 읽을 메시지로 꺼냄 */
export function dbMessage(e: unknown): string {
  const m = e instanceof Error ? e.message : String(e);
  return m.replace(/^.*?Error:\s*/, '');
}

/**
 * 문제은행 원본 DB. Studio가 사용한다.
 * 모든 쓰기는 트랜잭션 + 변경 이력(change_log) 기록을 함께 수행한다.
 */
export class Bank {
  constructor(public readonly db: Db) {}

  static create(SQL: SqlJsStatic, name = 'VisionDrill Core'): Bank {
    const db = Db.create(SQL);
    db.exec(BANK_SCHEMA_SQL);
    db.run(`UPDATE bank_meta SET value=? WHERE key='bank_name'`, [name]);
    db.run(`INSERT INTO bank_meta(key,value) VALUES('kind','bank'),('created_at',?)`, [nowIso()]);
    return new Bank(db);
  }

  static open(SQL: SqlJsStatic, bytes: Uint8Array): Bank {
    const db = Db.create(SQL, bytes);
    const kind = db.value<string>(`SELECT value FROM bank_meta WHERE key='kind'`);
    if (kind === 'pack') throw new BankError('배포용 팩(.vdpack)은 Studio에서 원본으로 열 수 없습니다. 원본 DB를 여세요.');
    const ver = db.value<string>(`SELECT value FROM bank_meta WHERE key='schema_version'`);
    if (!ver) throw new BankError('VisionDrill 문제은행 파일이 아닙니다');
    if (ver !== BANK_SCHEMA_VERSION) throw new BankError(`지원하지 않는 스키마 버전입니다: ${ver}`);
    return new Bank(db);
  }

  export(): Uint8Array {
    return this.db.export();
  }

  meta(key: string): string | undefined {
    return this.db.value<string>(`SELECT value FROM bank_meta WHERE key=?`, [key]);
  }

  // ------------------------------------------------------------------ 커리큘럼
  tracks(): Track[] {
    return this.db.all<Track>(`SELECT * FROM tracks ORDER BY sort_order`);
  }

  trackPrereqs(): TrackPrereq[] {
    return this.db.all<TrackPrereq>(`SELECT * FROM track_prereqs`);
  }

  chapters(trackId?: string): Chapter[] {
    return trackId
      ? this.db.all<Chapter>(`SELECT * FROM chapters WHERE track_id=? ORDER BY sort_order, id`, [trackId])
      : this.db.all<Chapter>(`SELECT c.* FROM chapters c JOIN tracks t ON t.id=c.track_id ORDER BY t.sort_order, c.sort_order, c.id`);
  }

  saveChapter(c: Chapter): void {
    try {
      this.db.run(
        `INSERT INTO chapters(id,track_id,code,name_ko,name_en,sort_order,stage,source_ref)
         VALUES(?,?,?,?,?,?,?,?)
         ON CONFLICT(id) DO UPDATE SET track_id=excluded.track_id, code=excluded.code, name_ko=excluded.name_ko,
           name_en=excluded.name_en, sort_order=excluded.sort_order, stage=excluded.stage, source_ref=excluded.source_ref`,
        [c.id, c.track_id, c.code.toUpperCase(), c.name_ko, c.name_en, c.sort_order, c.stage ?? 1, c.source_ref],
      );
    } catch (e) {
      throw new BankError(dbMessage(e));
    }
  }

  deleteChapter(id: string): void {
    const n = this.db.value<number>(`SELECT COUNT(*) FROM questions WHERE chapter_id=?`, [id]) ?? 0;
    if (n > 0) throw new BankError(`이 장에 문항이 ${n}개 있어 삭제할 수 없습니다`);
    this.db.run(`DELETE FROM chapters WHERE id=?`, [id]);
  }

  // ------------------------------------------------------------------ 문항 조회
  listQuestions(f: QuestionFilter = {}): QuestionSummary[] {
    const where: string[] = [];
    const p: (string | number)[] = [];
    if (f.track) (where.push('c.track_id=?'), p.push(f.track));
    if (f.chapter) (where.push('q.chapter_id=?'), p.push(f.chapter));
    if (f.type) (where.push('q.type=?'), p.push(f.type));
    if (f.status) (where.push('q.status=?'), p.push(f.status));
    if (f.level) (where.push('q.cognitive_level=?'), p.push(f.level));
    if (f.tag) (where.push('EXISTS(SELECT 1 FROM question_tags t WHERE t.question_id=q.id AND t.tag=?)'), p.push(f.tag));
    if (f.origin === 'project') where.push('b.has_project_term=1');
    if (f.origin === 'standard') where.push('b.has_project_term=0');
    if (f.text?.trim()) {
      const like = `%${f.text.trim()}%`;
      where.push(`(q.id LIKE ? OR q.stem LIKE ? OR q.explanation_short LIKE ? OR q.explanation_full LIKE ?
        OR EXISTS(SELECT 1 FROM choices ch WHERE ch.question_id=q.id AND (ch.text LIKE ? OR ch.rationale LIKE ?))
        OR EXISTS(SELECT 1 FROM accepted_answers a WHERE a.question_id=q.id AND a.text LIKE ?)
        OR EXISTS(SELECT 1 FROM question_terms qt JOIN terms tm ON tm.id=qt.term_id WHERE qt.question_id=q.id
                  AND (tm.term_en LIKE ? OR tm.term_ko LIKE ? OR tm.abbreviation LIKE ?)))`);
      p.push(like, like, like, like, like, like, like, like, like, like);
    }
    return this.db.all<QuestionSummary>(
      `SELECT q.id, q.type, q.chapter_id, c.track_id, q.cognitive_level, q.difficulty, q.stem, q.status,
              q.version, q.updated_at, b.has_project_term
         FROM questions q
         JOIN chapters c ON c.id=q.chapter_id
         JOIN tracks tr ON tr.id=c.track_id
         JOIN v_question_badges b ON b.question_id=q.id
        ${where.length ? 'WHERE ' + where.join(' AND ') : ''}
        ORDER BY tr.sort_order, c.sort_order, q.id`,
      p,
    );
  }

  getQuestion(id: string): Question | undefined {
    const r = this.db.get<Row>(`SELECT * FROM questions WHERE id=?`, [id]);
    if (!r) return undefined;
    return rowToQuestion(this.db, r);
  }

  allTags(): string[] {
    return this.db.all<{ tag: string }>(`SELECT DISTINCT tag FROM question_tags ORDER BY tag`).map((r) => r.tag);
  }

  nextQuestionId(chapterId: string): string {
    const ch = this.db.get<Chapter>(`SELECT * FROM chapters WHERE id=?`, [chapterId]);
    if (!ch) throw new BankError('장을 찾을 수 없습니다');
    const prefix = `${ch.track_id}-${ch.code}-`;
    const ids = this.db.all<{ id: string }>(`SELECT id FROM questions WHERE id LIKE ?`, [prefix + '%']);
    const max = ids.reduce((m, r) => Math.max(m, parseInt(r.id.slice(prefix.length), 10) || 0), 0);
    return prefix + String(max + 1).padStart(3, '0');
  }

  // ------------------------------------------------------------------ 문항 쓰기
  /**
   * 문항 저장(신규/수정). 수정이면 이전 상태를 change_log에 남기고 version을 올린다.
   * 정답이 바뀌면 breaking_version도 올려 학습 앱이 해당 문항의 복습 일정을 초기화하게 한다.
   * 검수 완료 문항은 검증 오류가 있으면 저장할 수 없다.
   */
  saveQuestion(input: Question, note?: string): Question {
    const q = cleanQuestion(input);
    return this.db.tx(() => {
      const prev = this.getQuestion(q.id);
      if (prev) {
        if (prev.status === 'retired') throw new BankError('출제 중단된 문항은 수정할 수 없습니다. 복제해서 새 문항으로 만드세요');
        if (prev.status === 'reviewed' && hasErrors(validateQuestion(q)))
          throw new BankError('검수 완료 문항은 검증 오류가 없어야 저장됩니다. 초안으로 되돌린 뒤 수정하세요');
        if (!contentChanged(prev, q)) return prev;
        this.log('question', prev.id, prev.version, 'update', prev, note);
        const version = prev.version + 1;
        const breaking = answerChanged(prev, q) ? version : prev.breaking_version;
        this.writeChildren(q, true);
        this.db.run(
          `UPDATE questions SET type=?, chapter_id=?, cognitive_level=?, difficulty=?, stem=?, image_asset_id=?,
             answer_choice=?, explanation_short=?, explanation_full=?, source_ref=?, version=?, breaking_version=?
           WHERE id=?`,
          [
            q.type,
            q.chapter_id,
            q.cognitive_level,
            q.difficulty,
            q.stem,
            q.image_asset_id,
            q.answer_choice,
            q.explanation_short,
            q.explanation_full,
            q.source_ref,
            version,
            breaking,
            q.id,
          ],
        );
      } else {
        try {
          this.db.run(
            `INSERT INTO questions(id,type,chapter_id,cognitive_level,difficulty,stem,image_asset_id,answer_choice,
               explanation_short,explanation_full,source_ref,status)
             VALUES(?,?,?,?,?,?,?,?,?,?,?,'draft')`,
            [
              q.id,
              q.type,
              q.chapter_id,
              q.cognitive_level,
              q.difficulty,
              q.stem,
              q.image_asset_id,
              q.answer_choice,
              q.explanation_short,
              q.explanation_full,
              q.source_ref,
            ],
          );
        } catch (e) {
          throw new BankError(friendlyInsertError(e));
        }
        this.writeChildren(q, false);
        this.log('question', q.id, 1, 'create', this.getQuestion(q.id)!, note);
      }
      return this.getQuestion(q.id)!;
    });
  }

  private writeChildren(q: Question, replace: boolean) {
    if (replace) {
      for (const t of ['choices', 'accepted_answers', 'question_tags', 'question_terms'])
        this.db.run(`DELETE FROM ${t} WHERE question_id=?`, [q.id]);
    }
    try {
      for (const c of q.choices)
        this.db.run(`INSERT INTO choices(question_id,idx,text,rationale) VALUES(?,?,?,?)`, [q.id, c.idx, c.text, c.rationale || null]);
      for (const a of q.answers)
        this.db.run(`INSERT INTO accepted_answers(question_id,lang,text,normalized,is_primary) VALUES(?,?,?,?,?)`, [
          q.id,
          a.lang,
          a.text,
          normalize(a.text),
          a.is_primary ? 1 : 0,
        ]);
      for (const t of q.tags) this.db.run(`INSERT INTO question_tags(question_id,tag) VALUES(?,?)`, [q.id, t]);
      for (const t of q.term_ids) this.db.run(`INSERT INTO question_terms(question_id,term_id) VALUES(?,?)`, [q.id, t]);
    } catch (e) {
      const m = dbMessage(e);
      if (m.includes('choices.text') || m.includes('CHECK constraint failed: trim(text)'))
        throw new BankError('비어 있는 보기가 있습니다. 보기 5개를 모두 입력하세요');
      if (m.includes('accepted_answers.question_id, accepted_answers.lang, accepted_answers.normalized'))
        throw new BankError('같은 허용 답이 중복되었습니다');
      if (m.includes('FOREIGN KEY')) throw new BankError('존재하지 않는 용어가 연결되어 있습니다');
      throw new BankError(m);
    }
  }

  /** 상태 변경: draft ↔ reviewed, → retired */
  setStatus(id: string, status: Status, reviewer?: string, note?: string): Question {
    return this.db.tx(() => {
      const prev = this.getQuestion(id);
      if (!prev) throw new BankError('문항을 찾을 수 없습니다');
      if (prev.status === status) return prev;
      if (status === 'reviewed') {
        const issues = validateQuestion(prev).filter((i) => i.level === 'error');
        if (issues.length) throw new BankError('검수 완료로 바꿀 수 없습니다: ' + issues.map((i) => i.message).join(' / '));
        if (!reviewer?.trim()) throw new BankError('검수자 이름을 설정하세요');
      }
      this.log('question', id, prev.version, status === 'retired' ? 'retire' : 'status', prev, note ?? `${prev.status} → ${status}`);
      try {
        if (status === 'reviewed')
          this.db.run(`UPDATE questions SET status='reviewed', reviewed_by=?, reviewed_at=? WHERE id=?`, [reviewer!, nowIso(), id]);
        else this.db.run(`UPDATE questions SET status=? WHERE id=?`, [status, id]);
      } catch (e) {
        throw new BankError(dbMessage(e));
      }
      return this.getQuestion(id)!;
    });
  }

  /** 삭제: 초안만 가능 (DB 트리거가 한 번 더 막음) */
  deleteQuestion(id: string): void {
    this.db.tx(() => {
      const prev = this.getQuestion(id);
      if (!prev) return;
      if (prev.status !== 'draft') throw new BankError('검수된 문항은 삭제할 수 없습니다. 출제 중단으로 변경하세요');
      this.log('question', id, prev.version, 'delete', prev, '삭제');
      this.db.run(`DELETE FROM questions WHERE id=?`, [id]);
    });
  }

  duplicateQuestion(id: string, newId?: string): Question {
    const src = this.getQuestion(id);
    if (!src) throw new BankError('문항을 찾을 수 없습니다');
    const copy: Question = {
      ...structuredClone(src),
      id: newId ?? this.nextQuestionId(src.chapter_id),
      status: 'draft',
      version: 1,
      breaking_version: 1,
      reviewed_by: null,
      reviewed_at: null,
    };
    return this.saveQuestion(copy, `${id}에서 복제`);
  }

  history(entityType: 'question' | 'term', id: string): ChangeLogEntry[] {
    return this.db.all<ChangeLogEntry>(`SELECT * FROM change_log WHERE entity_type=? AND entity_id=? ORDER BY id DESC`, [entityType, id]);
  }

  /** 변경 이력의 스냅샷 내용으로 되돌리기 (현재 상태는 다시 이력으로 남음) */
  restoreQuestion(logId: number): Question {
    const e = this.db.get<ChangeLogEntry>(`SELECT * FROM change_log WHERE id=?`, [logId]);
    if (!e || e.entity_type !== 'question') throw new BankError('이력을 찾을 수 없습니다');
    const snap = JSON.parse(e.snapshot_json) as Question;
    const cur = this.getQuestion(e.entity_id);
    if (!cur) {
      return this.saveQuestion({ ...snap, status: 'draft' }, `v${e.version} 내용으로 복원`);
    }
    if (cur.status === 'reviewed') this.setStatus(cur.id, 'draft', undefined, '복원을 위해 초안으로 전환');
    return this.saveQuestion({ ...snap, status: 'draft' }, `v${e.version} 내용으로 복원`);
  }

  private log(type: 'question' | 'term', id: string, version: number | null, action: string, snapshot: unknown, note?: string) {
    this.db.run(`INSERT INTO change_log(entity_type,entity_id,version,action,snapshot_json,note) VALUES(?,?,?,?,?,?)`, [
      type,
      id,
      version,
      action,
      JSON.stringify(snapshot),
      note ?? null,
    ]);
  }

  // ------------------------------------------------------------------ 용어
  terms(): (Term & { question_count: number })[] {
    return this.db.all(
      `SELECT t.*, (SELECT COUNT(*) FROM question_terms qt WHERE qt.term_id=t.id) AS question_count
         FROM terms t ORDER BY lower(t.term_en)`,
    );
  }

  getTerm(id: string): Term | undefined {
    return this.db.get<Term>(`SELECT * FROM terms WHERE id=?`, [id]);
  }

  saveTerm(t: Term, note?: string): Term {
    const term = {
      ...t,
      id: t.id.trim(),
      term_en: t.term_en.trim(),
      term_ko: t.term_ko?.trim() || null,
      abbreviation: t.abbreviation?.trim() || null,
      definition: t.definition.trim(),
      usage_example: t.usage_example?.trim() || null,
    };
    if (!/^[a-z0-9][a-z0-9-]*$/.test(term.id))
      throw new BankError('용어 ID는 영문 소문자·숫자·하이픈만 쓸 수 있습니다 (예: cooks-distance)');
    if (!term.term_en) throw new BankError('영문 용어를 입력하세요');
    if (!term.definition) throw new BankError('한 줄 정의를 입력하세요');
    return this.db.tx(() => {
      const prev = this.getTerm(term.id);
      if (prev) this.log('term', term.id, null, 'update', prev, note);
      this.db.run(
        `INSERT INTO terms(id,term_en,term_ko,abbreviation,definition,usage_example,origin,track_id,status)
         VALUES(?,?,?,?,?,?,?,?,?)
         ON CONFLICT(id) DO UPDATE SET term_en=excluded.term_en, term_ko=excluded.term_ko, abbreviation=excluded.abbreviation,
           definition=excluded.definition, usage_example=excluded.usage_example, origin=excluded.origin,
           track_id=excluded.track_id, status=excluded.status`,
        [
          term.id,
          term.term_en,
          term.term_ko,
          term.abbreviation,
          term.definition,
          term.usage_example,
          term.origin,
          term.track_id,
          term.status,
        ],
      );
      if (!prev) this.log('term', term.id, null, 'create', this.getTerm(term.id), note);
      return this.getTerm(term.id)!;
    });
  }

  deleteTerm(id: string): void {
    this.db.tx(() => {
      const prev = this.getTerm(id);
      if (!prev) return;
      this.log('term', id, null, 'delete', prev, '삭제');
      try {
        this.db.run(`DELETE FROM terms WHERE id=?`, [id]);
      } catch (e) {
        throw new BankError(dbMessage(e));
      }
    });
  }

  // ------------------------------------------------------------------ 첨부 이미지
  async addAsset(filename: string, mime: string, data: Uint8Array): Promise<number> {
    const hash = await sha256Hex(data);
    const existing = this.db.value<number>(`SELECT id FROM assets WHERE sha256=?`, [hash]);
    if (existing) return existing;
    this.db.run(`INSERT INTO assets(filename,mime,data,sha256) VALUES(?,?,?,?)`, [filename, mime, data, hash]);
    return this.db.value<number>(`SELECT last_insert_rowid()`)!;
  }

  assetDataUrl(id: number): string | undefined {
    return assetDataUrl(this.db, id);
  }

  // ------------------------------------------------------------------ 통계
  stats() {
    const byTrack = this.db.all<{ track_id: string; name_ko: string; draft: number; reviewed: number; retired: number; total: number }>(
      `SELECT t.id AS track_id, t.name_ko,
              SUM(q.status='draft') AS draft, SUM(q.status='reviewed') AS reviewed, SUM(q.status='retired') AS retired,
              COUNT(q.id) AS total
         FROM tracks t LEFT JOIN chapters c ON c.track_id=t.id LEFT JOIN questions q ON q.chapter_id=c.id
        GROUP BY t.id ORDER BY t.sort_order`,
    );
    const byType = this.db.all<{ type: string; n: number }>(`SELECT type, COUNT(*) AS n FROM questions GROUP BY type`);
    const byLevel = this.db.all<{ level: number; n: number }>(
      `SELECT cognitive_level AS level, COUNT(*) AS n FROM questions GROUP BY cognitive_level`,
    );
    const terms = this.db.get<{ total: number; project: number; unlinked: number }>(
      `SELECT COUNT(*) AS total, SUM(origin='project') AS project,
              SUM(NOT EXISTS(SELECT 1 FROM question_terms qt WHERE qt.term_id=terms.id)) AS unlinked FROM terms`,
    )!;
    const exports = this.db.all<Row>(`SELECT * FROM pack_exports ORDER BY id DESC LIMIT 10`);
    return { byTrack, byType, byLevel, terms, exports };
  }
}

// ==================================================================== 헬퍼

export function rowToQuestion(db: Db, r: Row): Question {
  const id = r.id as string;
  return {
    id,
    type: r.type as Question['type'],
    chapter_id: r.chapter_id as string,
    cognitive_level: r.cognitive_level as number,
    difficulty: r.difficulty as number,
    stem: r.stem as string,
    image_asset_id: (r.image_asset_id as number) ?? null,
    answer_choice: (r.answer_choice as number) ?? null,
    explanation_short: r.explanation_short as string,
    explanation_full: (r.explanation_full as string) ?? null,
    source_ref: (r.source_ref as string) ?? null,
    status: r.status as Status,
    version: r.version as number,
    breaking_version: r.breaking_version as number,
    reviewed_by: (r.reviewed_by as string) ?? null,
    reviewed_at: (r.reviewed_at as string) ?? null,
    created_at: r.created_at as string,
    updated_at: r.updated_at as string,
    choices: db.all<{ idx: number; text: string; rationale: string | null }>(
      `SELECT idx, text, rationale FROM choices WHERE question_id=? ORDER BY idx`,
      [id],
    ),
    answers: db
      .all<{ lang: Answer['lang']; text: string; is_primary: number }>(
        `SELECT lang, text, is_primary FROM accepted_answers WHERE question_id=? ORDER BY is_primary DESC, lang, id`,
        [id],
      )
      .map((a) => ({ lang: a.lang, text: a.text, is_primary: !!a.is_primary })),
    tags: db.all<{ tag: string }>(`SELECT tag FROM question_tags WHERE question_id=? ORDER BY tag`, [id]).map((t) => t.tag),
    term_ids: db
      .all<{ term_id: string }>(`SELECT term_id FROM question_terms WHERE question_id=? ORDER BY term_id`, [id])
      .map((t) => t.term_id),
  };
}

export function assetDataUrl(db: Db, id: number): string | undefined {
  const r = db.get<{ mime: string; data: Uint8Array }>(`SELECT mime, data FROM assets WHERE id=?`, [id]);
  if (!r) return undefined;
  let bin = '';
  const bytes = r.data;
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return `data:${r.mime};base64,${btoa(bin)}`;
}

function cleanQuestion(q: Question): Question {
  const isShort = q.type === 'short';
  const answers = isShort ? q.answers.map((a) => ({ ...a, text: a.text.trim() })).filter((a) => a.text) : [];
  if (isShort && answers.length && !answers.some((a) => a.is_primary)) answers[0].is_primary = true;
  return {
    ...q,
    id: q.id.trim(),
    stem: q.stem.trim(),
    explanation_short: q.explanation_short.trim(),
    explanation_full: q.explanation_full?.trim() || null,
    source_ref: q.source_ref?.trim() || null,
    answer_choice: isShort ? null : q.answer_choice,
    choices: isShort ? [] : q.choices.map((c) => ({ idx: c.idx, text: c.text.trim(), rationale: c.rationale?.trim() || null })),
    answers,
    tags: [...new Set(q.tags.map((t) => t.trim()).filter(Boolean))],
    term_ids: [...new Set(q.term_ids)],
  };
}

const CONTENT_KEYS: (keyof Question)[] = [
  'type',
  'chapter_id',
  'cognitive_level',
  'difficulty',
  'stem',
  'image_asset_id',
  'answer_choice',
  'explanation_short',
  'explanation_full',
  'source_ref',
  'choices',
  'answers',
  'tags',
  'term_ids',
];

function contentChanged(a: Question, b: Question): boolean {
  return CONTENT_KEYS.some((k) => JSON.stringify(a[k]) !== JSON.stringify(b[k]));
}

/** 정답이 바뀌었는지: 유형, 정답 번호, 정답 보기 내용, 허용 답 집합 */
function answerChanged(a: Question, b: Question): boolean {
  if (a.type !== b.type) return true;
  if (a.type === 'short') {
    const set = (q: Question) =>
      q.answers
        .map((x) => `${x.lang}:${normalize(x.text)}`)
        .sort()
        .join('|');
    return set(a) !== set(b);
  }
  if (a.answer_choice !== b.answer_choice) return true;
  const correctText = (q: Question) => normalize(q.choices.find((c) => c.idx === q.answer_choice)?.text ?? '');
  return correctText(a) !== correctText(b);
}

function friendlyInsertError(e: unknown): string {
  const m = dbMessage(e);
  if (m.includes('UNIQUE constraint failed: questions.id')) return '같은 ID의 문항이 이미 있습니다';
  if (m.includes('FOREIGN KEY')) return '존재하지 않는 장(chapter)입니다';
  if (m.includes('answer_choice')) return '객관식 문항은 정답 번호(1~5)가 필요합니다';
  return m;
}
