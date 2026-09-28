import { normalize } from './grading';
import type { Question } from './types';

export interface Issue {
  level: 'error' | 'warning';
  field: string;
  message: string;
}

const ID_RE = /^[A-H]-[A-Z0-9]{1,12}-\d{3,}$/;

/**
 * 문항 검증. error는 검수 완료(reviewed) 전환을 막는 항목(DB 검수 게이트와 동일 기준),
 * warning은 권장 사항.
 */
export function validateQuestion(q: Question): Issue[] {
  const issues: Issue[] = [];
  const err = (field: string, message: string) => issues.push({ level: 'error', field, message });
  const warn = (field: string, message: string) => issues.push({ level: 'warning', field, message });

  if (!ID_RE.test(q.id)) warn('id', '문항 ID 형식 권장: 트랙-장코드-번호 (예: F-NMS-003)');
  if (!q.stem.trim()) err('stem', '문제 본문이 비어 있습니다');
  if (!q.explanation_short.trim()) err('explanation_short', '짧은 해설이 비어 있습니다');
  if (q.explanation_short.length > 400) warn('explanation_short', '짧은 해설은 2~3문장(400자 이내)을 권장합니다');

  if (q.type === 'short') {
    const filled = q.answers.filter((a) => a.text.trim());
    if (filled.length === 0) err('answers', '허용 답을 1개 이상 입력하세요');
    if (filled.filter((a) => a.is_primary).length !== 1) err('answers', '대표 정답을 정확히 1개 지정하세요');
    if (!filled.some((a) => a.lang === 'en')) warn('answers', '영문 허용 답이 없습니다');
    if (!filled.some((a) => a.lang === 'ko')) warn('answers', '한글 허용 답이 없습니다');
    const seen = new Set<string>();
    for (const a of filled) {
      const k = `${a.lang}:${normalize(a.text)}`;
      if (seen.has(k)) err('answers', `같은 허용 답이 중복되었습니다: ${a.text}`);
      seen.add(k);
    }
  } else {
    if (q.choices.length !== 5) err('choices', '보기는 정확히 5개여야 합니다');
    q.choices.forEach((c) => {
      if (!c.text.trim()) err(`choice.${c.idx}`, `보기 ${c.idx}가 비어 있습니다`);
      if (c.idx !== q.answer_choice && !(c.rationale ?? '').trim()) err(`choice.${c.idx}`, `보기 ${c.idx}의 틀린 이유가 비어 있습니다`);
    });
    if (q.answer_choice == null || q.answer_choice < 1 || q.answer_choice > 5) err('answer_choice', '정답 보기를 선택하세요');
    const texts = q.choices.map((c) => normalize(c.text)).filter(Boolean);
    if (new Set(texts).size !== texts.length) err('choices', '같은 내용의 보기가 있습니다');
    if (q.choices.some((c) => /모두\s*(정답|맞|옳)|위\s*(의)?\s*(모든|보기)/.test(c.text)))
      warn('choices', '"모두 정답" 류 보기는 보기 순서를 섞을 때 어색해질 수 있습니다');
  }
  if (q.term_ids.length === 0) warn('term_ids', '연결된 용어가 없습니다');
  if (!q.source_ref) warn('source_ref', '출처가 비어 있습니다');
  return issues;
}

export function hasErrors(issues: Issue[]): boolean {
  return issues.some((i) => i.level === 'error');
}
