import { LEVEL_LABEL, TYPE_LABEL, type Question, type Response, type Term } from '@visiondrill/core';
import { useEffect, useRef } from 'react';

export const CIRCLED = ['①', '②', '③', '④', '⑤', '⑥'];

export interface QuestionCardProps {
  question: Question;
  order?: number[];
  terms?: Term[];
  response?: Response;
  onResponse?: (r: Response) => void;
  guessed?: boolean;
  onGuessed?: (g: boolean) => void;
  index?: number;
  total?: number;
  imageUrl?: string;
  onEnter?: () => void;
  autoFocus?: boolean;
}

export function TypeChips({ question, terms }: { question: Question; terms?: Term[] }) {
  const hasProject = terms?.some((t) => t.origin === 'project');
  return (
    <div className="vd-chips">
      <span className={`vd-chip vd-chip-${question.type}`}>{TYPE_LABEL[question.type]}</span>
      <span className="vd-chip">{LEVEL_LABEL[question.cognitive_level] ?? `L${question.cognitive_level}`}</span>
      <span className="vd-chip vd-chip-muted">{question.id}</span>
      {hasProject && (
        <span className="vd-chip vd-chip-project" title="Diagnostics에서 정의한 용어가 포함된 문항입니다. 외부 소통 시 풀어서 설명하세요.">
          프로젝트 정의 용어
        </span>
      )}
    </div>
  );
}

/** 풀이 화면의 문항 카드 (학습 앱 풀이·Studio 미리보기 공용) */
export function QuestionCard(p: QuestionCardProps) {
  const q = p.question;
  const order = p.order ?? q.choices.map((c) => c.idx);
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (p.autoFocus && q.type === 'short') inputRef.current?.focus();
  }, [q.id, p.autoFocus, q.type]);
  const selected = p.response?.kind === 'choice' ? p.response.idx : null;
  const text = p.response?.kind === 'text' ? p.response.text : '';

  return (
    <article className="vd-card">
      <header className="vd-card-head">
        {p.index !== undefined && p.total !== undefined && (
          <span className="vd-counter">
            {p.index + 1} / {p.total}
          </span>
        )}
        <TypeChips question={q} terms={p.terms} />
      </header>
      <div className="vd-stem">{q.stem || <span className="vd-placeholder">문제 본문</span>}</div>
      {p.imageUrl && <img className="vd-image" src={p.imageUrl} alt="문항 그림" />}

      {q.type === 'short' ? (
        <div className="vd-short">
          <input
            ref={inputRef}
            className="vd-input"
            value={text}
            placeholder="영문 또는 한글로 입력"
            onChange={(e) => p.onResponse?.({ kind: 'text', text: e.target.value })}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.nativeEvent.isComposing) p.onEnter?.();
            }}
            readOnly={!p.onResponse}
          />
        </div>
      ) : (
        <ol className="vd-choices">
          {order.map((idx, pos) => {
            const c = q.choices.find((x) => x.idx === idx);
            if (!c) return null;
            return (
              <li key={idx}>
                <button
                  type="button"
                  className={`vd-choice ${selected === idx ? 'is-selected' : ''}`}
                  onClick={() => p.onResponse?.({ kind: 'choice', idx })}
                  aria-pressed={selected === idx}
                >
                  <span className="vd-choice-no">{CIRCLED[pos]}</span>
                  <span className="vd-choice-text">{c.text || <span className="vd-placeholder">보기 {idx}</span>}</span>
                </button>
              </li>
            );
          })}
        </ol>
      )}

      {p.onGuessed && (
        <label className="vd-guess">
          <input type="checkbox" checked={!!p.guessed} onChange={(e) => p.onGuessed?.(e.target.checked)} />
          <span>찍었음 (확신 없음)</span>
        </label>
      )}
    </article>
  );
}
