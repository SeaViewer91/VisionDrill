/**
 * 교환용 JSON 형식 (가져오기/내보내기, LLM 초안 입력 형식)
 *
 * {
 *   "format": "visiondrill-bank-exchange", "format_version": 1,
 *   "chapters":  [{ "id": "F.nms", "track": "F", "code": "NMS", "name_ko": "NMS", ... }],
 *   "terms":     [{ "id": "nms", "term_en": "Non-Maximum Suppression", "definition": "...", ... }],
 *   "questions": [{ "id": "F-NMS-001", "type": "mcq", "chapter": "F.nms", "stem": "...",
 *                   "choices": [{ "text": "...", "rationale": "..." } x5], "answer": 2,
 *                   "explanation_short": "...", "terms": ["nms"], ... }]
 * }
 */
import { Bank, BankError, dbMessage } from './bank';
import type { Answer, Chapter, Question, QType, Status, Term, TermOrigin } from './types';

export const EXCHANGE_FORMAT = 'visiondrill-bank-exchange';

export interface XChapter {
  id: string;
  track: string;
  code: string;
  name_ko: string;
  name_en?: string | null;
  sort_order?: number;
  stage?: number;
  source_ref?: string | null;
}

export interface XTerm {
  id: string;
  term_en: string;
  term_ko?: string | null;
  abbreviation?: string | null;
  definition: string;
  usage_example?: string | null;
  origin?: TermOrigin;
  track?: string | null;
  status?: Status;
}

export interface XQuestion {
  id: string;
  type: QType;
  chapter: string;
  cognitive_level?: number;
  difficulty?: number;
  stem: string;
  choices?: { text: string; rationale?: string | null }[];
  answer?: number;
  accepted_answers?: { en?: string[]; ko?: string[] };
  primary?: string;
  explanation_short: string;
  explanation_full?: string | null;
  tags?: string[];
  terms?: string[];
  source_ref?: string | null;
  status?: Status;
  reviewed_by?: string | null;
}

export interface ExchangeFile {
  format: typeof EXCHANGE_FORMAT;
  format_version: 1;
  exported_at?: string;
  chapters?: XChapter[];
  terms?: XTerm[];
  questions?: XQuestion[];
}

export interface ImportMessage {
  id: string;
  level: 'error' | 'warning' | 'info';
  message: string;
}

export interface ImportResult {
  chapters: number;
  terms: number;
  created: number;
  updated: number;
  unchanged: number;
  failed: number;
  reviewed: number;
  messages: ImportMessage[];
}

export interface ImportOptions {
  /** 파일에 status: "reviewed"가 있을 때 검수자로 기록할 이름 */
  reviewer?: string;
  /** true면 이미 있는 문항은 건너뜀 */
  skipExisting?: boolean;
}

// ---------------------------------------------------------------- 구조 검사
export function parseExchange(text: string): { file?: ExchangeFile; errors: string[] } {
  let raw: any;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return { errors: [`JSON 형식 오류: ${(e as Error).message}`] };
  }
  // 문항 배열만 붙여넣은 경우도 허용 (LLM 초안)
  if (Array.isArray(raw)) raw = { format: EXCHANGE_FORMAT, format_version: 1, questions: raw };
  const errors: string[] = [];
  if (raw.format !== EXCHANGE_FORMAT) errors.push(`format 값은 "${EXCHANGE_FORMAT}" 이어야 합니다`);
  for (const k of ['chapters', 'terms', 'questions'])
    if (raw[k] !== undefined && !Array.isArray(raw[k])) errors.push(`${k}는 배열이어야 합니다`);
  (raw.questions ?? []).forEach((q: any, i: number) => {
    const at = `questions[${i}]${q?.id ? ` (${q.id})` : ''}`;
    if (typeof q?.id !== 'string' || !q.id) errors.push(`${at}: id가 없습니다`);
    if (!['mcq', 'short', 'scenario'].includes(q?.type)) errors.push(`${at}: type은 mcq/short/scenario 중 하나여야 합니다`);
    if (typeof q?.chapter !== 'string') errors.push(`${at}: chapter가 없습니다`);
    if (typeof q?.stem !== 'string') errors.push(`${at}: stem이 없습니다`);
    if (typeof q?.explanation_short !== 'string') errors.push(`${at}: explanation_short가 없습니다`);
    if (q?.type === 'short') {
      const en = q.accepted_answers?.en ?? [];
      const ko = q.accepted_answers?.ko ?? [];
      if (!Array.isArray(en) || !Array.isArray(ko) || en.length + ko.length === 0)
        errors.push(`${at}: 단답형은 accepted_answers.en 또는 .ko가 필요합니다`);
    } else if (q?.type) {
      if (!Array.isArray(q.choices)) errors.push(`${at}: choices 배열이 필요합니다`);
      else if (q.choices.some((c: any) => typeof c?.text !== 'string'))
        errors.push(`${at}: choices 항목은 { "text": ..., "rationale": ... } 형식이어야 합니다`);
      if (!Number.isInteger(q.answer) || q.answer < 1 || q.answer > 5) errors.push(`${at}: answer는 1~5 정수여야 합니다`);
    }
  });
  (raw.terms ?? []).forEach((t: any, i: number) => {
    if (typeof t?.id !== 'string' || typeof t?.term_en !== 'string' || typeof t?.definition !== 'string')
      errors.push(`terms[${i}]: id, term_en, definition이 필요합니다`);
  });
  (raw.chapters ?? []).forEach((c: any, i: number) => {
    if (!c?.id || !c?.track || !c?.code || !c?.name_ko) errors.push(`chapters[${i}]: id, track, code, name_ko가 필요합니다`);
  });
  return errors.length ? { errors } : { file: raw as ExchangeFile, errors };
}

// ---------------------------------------------------------------- 변환
export function fromXQuestion(x: XQuestion): Question {
  const isShort = x.type === 'short';
  let answers: Answer[] = [];
  if (isShort) {
    const en = (x.accepted_answers?.en ?? []).map((text) => ({ lang: 'en' as const, text, is_primary: false }));
    const ko = (x.accepted_answers?.ko ?? []).map((text) => ({ lang: 'ko' as const, text, is_primary: false }));
    answers = [...en, ...ko];
    const p = answers.find((a) => a.text === x.primary) ?? answers[0];
    if (p) p.is_primary = true;
  }
  return {
    id: x.id,
    type: x.type,
    chapter_id: x.chapter,
    cognitive_level: x.cognitive_level ?? 1,
    difficulty: x.difficulty ?? 2,
    stem: x.stem,
    image_asset_id: null,
    answer_choice: isShort ? null : (x.answer ?? null),
    explanation_short: x.explanation_short,
    explanation_full: x.explanation_full ?? null,
    source_ref: x.source_ref ?? null,
    status: 'draft',
    version: 1,
    breaking_version: 1,
    reviewed_by: null,
    reviewed_at: null,
    choices: isShort ? [] : (x.choices ?? []).map((c, i) => ({ idx: i + 1, text: c.text, rationale: c.rationale ?? null })),
    answers,
    tags: x.tags ?? [],
    term_ids: x.terms ?? [],
  };
}

export function toXQuestion(q: Question, trackOf: (chapterId: string) => string | undefined): XQuestion & { track?: string } {
  const x: XQuestion & { track?: string } = {
    id: q.id,
    type: q.type,
    track: trackOf(q.chapter_id),
    chapter: q.chapter_id,
    cognitive_level: q.cognitive_level,
    difficulty: q.difficulty,
    stem: q.stem,
    explanation_short: q.explanation_short,
    explanation_full: q.explanation_full,
    tags: q.tags,
    terms: q.term_ids,
    source_ref: q.source_ref,
    status: q.status,
    reviewed_by: q.reviewed_by,
  };
  if (q.type === 'short') {
    x.accepted_answers = {
      en: q.answers.filter((a) => a.lang === 'en').map((a) => a.text),
      ko: q.answers.filter((a) => a.lang === 'ko').map((a) => a.text),
    };
    x.primary = q.answers.find((a) => a.is_primary)?.text;
  } else {
    x.choices = q.choices.map((c) => ({ text: c.text, rationale: c.rationale }));
    x.answer = q.answer_choice ?? undefined;
  }
  return x;
}

// ---------------------------------------------------------------- 가져오기
export function importExchange(bank: Bank, file: ExchangeFile, opt: ImportOptions = {}): ImportResult {
  const res: ImportResult = { chapters: 0, terms: 0, created: 0, updated: 0, unchanged: 0, failed: 0, reviewed: 0, messages: [] };
  const msg = (id: string, level: ImportMessage['level'], message: string) => res.messages.push({ id, level, message });

  bank.db.tx(() => {
    for (const c of file.chapters ?? []) {
      try {
        const existing = bank.chapters().find((x) => x.id === c.id);
        const ch: Chapter = {
          id: c.id,
          track_id: c.track,
          code: c.code,
          name_ko: c.name_ko,
          name_en: c.name_en ?? existing?.name_en ?? null,
          sort_order: c.sort_order ?? existing?.sort_order ?? bank.chapters(c.track).length + 1,
          stage: c.stage ?? existing?.stage ?? 1,
          source_ref: c.source_ref ?? existing?.source_ref ?? null,
        };
        bank.db.tx(() => bank.saveChapter(ch));
        res.chapters++;
      } catch (e) {
        msg(c.id, 'error', `장 저장 실패: ${dbMessage(e)}`);
      }
    }
    for (const t of file.terms ?? []) {
      try {
        const term: Term = {
          id: t.id,
          term_en: t.term_en,
          term_ko: t.term_ko ?? null,
          abbreviation: t.abbreviation ?? null,
          definition: t.definition,
          usage_example: t.usage_example ?? null,
          origin: t.origin ?? 'standard',
          track_id: t.track ?? null,
          status: t.status ?? 'reviewed',
        };
        bank.db.tx(() => bank.saveTerm(term, '가져오기'));
        res.terms++;
      } catch (e) {
        msg(t.id, 'error', `용어 저장 실패: ${dbMessage(e)}`);
      }
    }
    for (const x of file.questions ?? []) {
      try {
        bank.db.tx(() => {
          const existing = bank.getQuestion(x.id);
          if (existing && opt.skipExisting) {
            res.unchanged++;
            msg(x.id, 'info', '이미 있어 건너뜀');
            return;
          }
          if (existing?.status === 'retired') {
            res.unchanged++;
            msg(x.id, 'warning', '출제 중단된 문항이라 건너뜀');
            return;
          }
          const q = fromXQuestion(x);
          const before = existing?.version;
          let saved: Question;
          try {
            saved = bank.saveQuestion(q, '가져오기');
          } catch (e) {
            if (existing?.status !== 'reviewed') throw e;
            // 검수 완료 문항인데 새 내용에 검증 오류가 있으면 초안으로 내리고 저장
            bank.setStatus(x.id, 'draft', undefined, '가져오기: 검증 오류로 초안 전환');
            saved = bank.saveQuestion(q, '가져오기');
            msg(x.id, 'warning', `검증 오류가 있어 초안으로 전환됨 — ${(e as Error).message}`);
          }
          if (!existing) res.created++;
          else if (saved.version !== before) res.updated++;
          else res.unchanged++;
          if (x.status === 'reviewed' && saved.status !== 'reviewed') {
            const reviewer = x.reviewed_by || opt.reviewer;
            try {
              bank.setStatus(x.id, 'reviewed', reviewer ?? undefined, '가져오기');
              res.reviewed++;
            } catch (e) {
              msg(x.id, 'warning', `초안으로 저장됨 — ${e instanceof Error ? e.message : dbMessage(e)}`);
            }
          }
        });
      } catch (e) {
        res.failed++;
        msg(x.id, 'error', e instanceof BankError ? e.message : dbMessage(e));
      }
    }
  });
  return res;
}

// ---------------------------------------------------------------- 내보내기
export function exportExchange(bank: Bank, statuses?: Status[]): ExchangeFile {
  const chapters = bank.chapters();
  const trackOf = (id: string) => chapters.find((c) => c.id === id)?.track_id;
  const qs = bank
    .listQuestions()
    .filter((s) => !statuses || statuses.includes(s.status))
    .map((s) => bank.getQuestion(s.id)!);
  return {
    format: EXCHANGE_FORMAT,
    format_version: 1,
    exported_at: new Date().toISOString(),
    chapters: chapters.map((c) => ({
      id: c.id,
      track: c.track_id,
      code: c.code,
      name_ko: c.name_ko,
      name_en: c.name_en,
      sort_order: c.sort_order,
      stage: c.stage,
      source_ref: c.source_ref,
    })),
    terms: bank.terms().map((t) => ({
      id: t.id,
      term_en: t.term_en,
      term_ko: t.term_ko,
      abbreviation: t.abbreviation,
      definition: t.definition,
      usage_example: t.usage_example,
      origin: t.origin,
      track: t.track_id,
      status: t.status,
    })),
    questions: qs.map((q) => toXQuestion(q, trackOf)),
  };
}
