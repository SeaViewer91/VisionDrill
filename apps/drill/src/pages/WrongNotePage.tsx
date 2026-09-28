import { primaryAnswer, type SessionItem, type WrongNoteItem } from '@visiondrill/core';
import { CIRCLED, TypeChips } from '@visiondrill/ui';
import { useState } from 'react';
import { useDrill } from '../store';

const REASONS: { key: NonNullable<WrongNoteItem['reason']>; label: string }[] = [
  { key: 'unknown', label: '몰랐음' },
  { key: 'confused', label: '헷갈림' },
  { key: 'mistake', label: '실수' },
];

export function WrongNotePage({ onStart }: { onStart: (items: SessionItem[]) => void }) {
  const { learner, mutate, rev, toast } = useDrill();
  const [showGraduated, setShowGraduated] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  void rev;
  const notes = learner.wrongNotes(showGraduated);
  const graduateStreak = learner.settings().graduateStreak;

  return (
    <div className="page">
      <div className="page-head">
        <h1>오답노트</h1>
        <span className="vd-muted">{notes.length}문항</span>
        <span className="spacer" />
        <label style={{ fontSize: '0.88rem', display: 'flex', gap: 6, alignItems: 'center' }}>
          <input type="checkbox" checked={showGraduated} onChange={(e) => setShowGraduated(e.target.checked)} />
          졸업한 문항도 보기
        </label>
        <button
          className="vd-btn vd-btn-primary"
          disabled={!notes.length}
          onClick={() => {
            const items = learner.buildSession('wrong_note');
            if (!items.length) return toast('복습할 오답이 없습니다');
            onStart(items);
          }}
        >
          오답노트 복습 시작
        </button>
      </div>
      <p className="vd-muted" style={{ marginTop: -8, fontSize: '0.88rem' }}>
        틀린 이유를 표시해 두면 복습할 때 도움이 됩니다. 찍지 않고 {graduateStreak}번 연속으로 맞히면 자동으로 졸업합니다.
      </p>

      {!notes.length && <div className="empty">오답노트가 비어 있습니다 🎉</div>}
      <div style={{ display: 'grid', gap: 10 }}>
        {notes.map((n) => {
          const q = n.question;
          const isOpen = open === q.id;
          const last =
            q.type === 'short' ? n.last_response : n.last_response ? q.choices.find((c) => String(c.idx) === n.last_response)?.text : null;
          const answer = q.type === 'short' ? primaryAnswer(q)?.text : q.choices.find((c) => c.idx === q.answer_choice)?.text;
          return (
            <article className="vd-card note" key={q.id}>
              <div className="vd-card-head">
                <TypeChips question={q} />
                <span className="vd-muted" style={{ marginLeft: 'auto', fontSize: '0.82rem' }}>
                  틀린 횟수 {n.wrong_count} · 연속 정답 {n.streak}/{graduateStreak}
                </span>
              </div>
              <div className="vd-stem" style={{ marginBottom: 8 }}>
                {q.stem}
              </div>
              <div className="note-row">
                <div className="reason-group">
                  {REASONS.map((r) => (
                    <button
                      key={r.key}
                      className={`vd-btn vd-btn-small ${n.reason === r.key ? 'is-on' : ''}`}
                      onClick={() => mutate((l) => l.updateWrongNote(q.id, { reason: n.reason === r.key ? null : r.key }))}
                    >
                      {r.label}
                    </button>
                  ))}
                </div>
                <button className="vd-link" onClick={() => setOpen(isOpen ? null : q.id)}>
                  {isOpen ? '접기' : '정답·해설 보기'}
                </button>
                <button
                  className="vd-link vd-link-muted"
                  style={{ marginLeft: 'auto' }}
                  onClick={() => mutate((l) => l.graduateWrongNote(q.id), '오답노트에서 뺐습니다')}
                >
                  다 외웠음 (졸업)
                </button>
              </div>
              {isOpen && (
                <div className="vd-explain" style={{ marginTop: 10 }}>
                  {last && (
                    <p className="vd-rationale">
                      <strong>최근 오답</strong> {last}
                    </p>
                  )}
                  <p>
                    <strong style={{ color: 'var(--ok)' }}>정답</strong> {answer}
                  </p>
                  <p>{q.explanation_short}</p>
                  {q.explanation_full && <p className="vd-full">{q.explanation_full}</p>}
                  {q.type !== 'short' && (
                    <ul className="vd-rationale-list">
                      {q.choices
                        .filter((c) => c.idx !== q.answer_choice)
                        .map((c) => (
                          <li key={c.idx}>
                            <strong>{CIRCLED[c.idx - 1]}</strong> {c.text} — {c.rationale}
                          </li>
                        ))}
                    </ul>
                  )}
                </div>
              )}
              <textarea
                className="vd-textarea note-memo"
                rows={1}
                placeholder="메모 (예: IoU 분모는 합집합!)"
                defaultValue={n.memo ?? ''}
                onBlur={(e) =>
                  e.target.value !== (n.memo ?? '') && mutate((l) => l.updateWrongNote(q.id, { memo: e.target.value || null }))
                }
              />
            </article>
          );
        })}
      </div>
    </div>
  );
}
