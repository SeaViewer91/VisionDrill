import type { Answer, Question } from './types';

/**
 * 단답형 채점용 정규화.
 * - 유니코드 정규화(NFKC), 소문자화
 * - 공백·하이픈·언더스코어·마침표·가운뎃점·따옴표 제거
 * 예) "Non-Maximum  Suppression" → "nonmaximumsuppression", "비최대 억제" → "비최대억제"
 */
export function normalize(s: string): string {
  return s
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[\s\-_.·'"`’‘“”,/]+/g, '')
    .trim();
}

/** 괄호 부분을 뺀 변형까지 포함한 비교 후보: "Intersection over Union (IoU)" → 전체, 괄호 제거, 괄호 안 */
export function variants(s: string): string[] {
  const out = new Set<string>();
  out.add(normalize(s));
  const noParen = s.replace(/\([^)]*\)|（[^）]*）|\[[^\]]*\]/g, ' ');
  out.add(normalize(noParen));
  for (const m of s.matchAll(/\(([^)]*)\)|（([^）]*)）/g)) out.add(normalize(m[1] ?? m[2] ?? ''));
  out.delete('');
  return [...out];
}

export function levenshtein(a: string, b: string): number {
  const A = [...a];
  const B = [...b];
  if (A.length === 0) return B.length;
  if (B.length === 0) return A.length;
  let prev = Array.from({ length: B.length + 1 }, (_, j) => j);
  for (let i = 1; i <= A.length; i++) {
    const cur = [i];
    for (let j = 1; j <= B.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (A[i - 1] === B[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[B.length];
}

/** 오타 허용 거리: 짧은 약어(3자 이하)는 오타 판정 안 함, 4~7자는 1, 8자 이상은 2 */
export function nearMissThreshold(len: number): number {
  if (len <= 3) return 0;
  if (len <= 7) return 1;
  return 2;
}

export interface ShortGrade {
  correct: boolean;
  nearMiss: boolean;
  matched?: string;
}

/** 영문·한글 허용 답 중 하나라도 일치하면 정답. 근접하면 nearMiss(사용자가 결과 화면에서 인정 여부 결정) */
export function gradeShort(input: string, answers: Pick<Answer, 'text'>[]): ShortGrade {
  const inVars = variants(input);
  if (inVars.length === 0) return { correct: false, nearMiss: false };
  const accVars = answers.flatMap((a) => variants(a.text).map((v) => ({ v, text: a.text })));
  for (const iv of inVars) {
    const hit = accVars.find((a) => a.v === iv);
    if (hit) return { correct: true, nearMiss: false, matched: hit.text };
  }
  const main = normalize(input);
  for (const a of accVars) {
    const th = nearMissThreshold(Math.max([...a.v].length, [...main].length));
    if (th > 0 && levenshtein(main, a.v) <= th) return { correct: false, nearMiss: true, matched: a.text };
  }
  return { correct: false, nearMiss: false };
}

export type Response = { kind: 'choice'; idx: number | null } | { kind: 'text'; text: string };

export interface Grade {
  correct: boolean;
  nearMiss: boolean;
  answered: boolean;
}

export function gradeQuestion(q: Pick<Question, 'type' | 'answer_choice' | 'answers'>, r: Response | undefined): Grade {
  if (!r) return { correct: false, nearMiss: false, answered: false };
  if (q.type === 'short') {
    const text = r.kind === 'text' ? r.text : '';
    if (!text.trim()) return { correct: false, nearMiss: false, answered: false };
    const g = gradeShort(text, q.answers);
    return { correct: g.correct, nearMiss: g.nearMiss, answered: true };
  }
  const idx = r.kind === 'choice' ? r.idx : null;
  if (idx == null) return { correct: false, nearMiss: false, answered: false };
  return { correct: idx === q.answer_choice, nearMiss: false, answered: true };
}

export function primaryAnswer(q: Pick<Question, 'answers'>): Answer | undefined {
  return q.answers.find((a) => a.is_primary) ?? q.answers[0];
}
