import { primaryAnswer, type Grade, type Question, type Response, type Term } from '@visiondrill/core';
import { useState } from 'react';
import { CIRCLED, TypeChips } from './QuestionCard';

export interface ResultCardProps {
  question: Question;
  order?: number[];
  terms?: Term[];
  response?: Response;
  grade: Grade;
  guessed?: boolean;
  overridden?: boolean;
  index?: number;
  imageUrl?: string;
  onAcceptNearMiss?: () => void;
  onReport?: () => void;
}

/** 채점 후 결과·해설 카드. 오답은 해설이 자동으로 펼쳐지고, 정답은 눌러서 볼 수 있다. */
export function ResultCard(p: ResultCardProps) {
  const q = p.question;
  const correct = p.grade.correct || p.overridden;
  const [open, setOpen] = useState(!correct);
  const [full, setFull] = useState(false);
  const order = p.order ?? q.choices.map((c) => c.idx);
  const pos = (idx: number | null | undefined) => (idx == null ? -1 : order.indexOf(idx));
  const chosen = p.response?.kind === 'choice' ? p.response.idx : null;
  const chosenChoice = q.choices.find((c) => c.idx === chosen);
  const answerChoice = q.choices.find((c) => c.idx === q.answer_choice);
  const status = !p.grade.answered ? 'blank' : correct ? (p.guessed ? 'guessed' : 'correct') : 'wrong';
  const statusLabel = { blank: '미응답', correct: '정답', guessed: '정답 (찍음)', wrong: '오답' }[status];

  return (
    <article className={`vd-card vd-result is-${status}`}>
      <header className="vd-card-head">
        <span className={`vd-mark is-${status}`} aria-label={statusLabel}>
          {status === 'wrong' || status === 'blank' ? '✕' : '○'}
        </span>
        {p.index !== undefined && <span className="vd-counter">{p.index + 1}</span>}
        <TypeChips question={q} terms={p.terms} />
        <span className={`vd-status is-${status}`}>{statusLabel}</span>
      </header>
      <div className="vd-stem">{q.stem}</div>
      {p.imageUrl && <img className="vd-image" src={p.imageUrl} alt="문항 그림" />}

      <dl className="vd-answers">
        <div>
          <dt>내 답</dt>
          <dd>
            {q.type === 'short'
              ? p.response?.kind === 'text' && p.response.text.trim()
                ? p.response.text
                : '—'
              : chosenChoice
                ? `${CIRCLED[pos(chosen)]} ${chosenChoice.text}`
                : '—'}
          </dd>
        </div>
        {!correct && (
          <div>
            <dt>정답</dt>
            <dd className="vd-correct-answer">
              {q.type === 'short' ? primaryAnswer(q)?.text : answerChoice ? `${CIRCLED[pos(q.answer_choice)]} ${answerChoice.text}` : '—'}
            </dd>
          </div>
        )}
        {q.type === 'short' && (
          <div>
            <dt>인정 답</dt>
            <dd className="vd-muted">{q.answers.map((a) => a.text).join(' · ')}</dd>
          </div>
        )}
      </dl>

      {p.grade.nearMiss && !p.overridden && (
        <div className="vd-nearmiss">
          <span>오타로 보입니다. 정답으로 인정할까요?</span>
          {p.onAcceptNearMiss && (
            <button type="button" className="vd-btn vd-btn-small" onClick={p.onAcceptNearMiss}>
              정답으로 인정
            </button>
          )}
        </div>
      )}
      {p.overridden && <div className="vd-nearmiss is-done">오타를 정답으로 인정했습니다</div>}

      {!open ? (
        <button type="button" className="vd-link" onClick={() => setOpen(true)}>
          해설 보기
        </button>
      ) : (
        <section className="vd-explain">
          {q.type !== 'short' && chosenChoice && chosen !== q.answer_choice && chosenChoice.rationale && (
            <p className="vd-rationale">
              <strong>고른 답 {CIRCLED[pos(chosen)]}번이 틀린 이유</strong> {chosenChoice.rationale}
            </p>
          )}
          <p>{q.explanation_short}</p>
          {q.explanation_full && !full && (
            <button type="button" className="vd-link" onClick={() => setFull(true)}>
              더 보기
            </button>
          )}
          {q.explanation_full && full && <p className="vd-full">{q.explanation_full}</p>}
          {full && q.type !== 'short' && (
            <ul className="vd-rationale-list">
              {order
                .map((idx) => q.choices.find((c) => c.idx === idx)!)
                .filter((c) => c && c.idx !== q.answer_choice && c.idx !== chosen && c.rationale)
                .map((c) => (
                  <li key={c.idx}>
                    <strong>{CIRCLED[pos(c.idx)]}</strong> {c.rationale}
                  </li>
                ))}
            </ul>
          )}
          {!!p.terms?.length && (
            <ul className="vd-terms">
              {p.terms.map((t) => (
                <li key={t.id}>
                  <span className="vd-term-name">
                    {t.abbreviation && t.abbreviation !== t.term_en ? `${t.abbreviation} · ` : ''}
                    {t.term_en}
                    {t.term_ko ? ` (${t.term_ko})` : ''}
                    {t.origin === 'project' && <span className="vd-chip vd-chip-project vd-chip-xs">프로젝트 정의</span>}
                  </span>
                  <span className="vd-term-def">{t.definition}</span>
                  {t.usage_example && <span className="vd-term-usage">“{t.usage_example}”</span>}
                </li>
              ))}
            </ul>
          )}
          {correct && (
            <button type="button" className="vd-link" onClick={() => (setOpen(false), setFull(false))}>
              해설 접기
            </button>
          )}
        </section>
      )}
      {p.onReport && (
        <footer className="vd-card-foot">
          <button type="button" className="vd-link vd-link-muted" onClick={p.onReport}>
            이 문항 오류 신고
          </button>
        </footer>
      )}
    </article>
  );
}
