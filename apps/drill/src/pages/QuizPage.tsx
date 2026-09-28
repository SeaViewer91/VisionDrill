import { askConfirm } from '@visiondrill/platform';
import { assetDataUrl, type Learner, type Response, type SessionItem, type SessionMode, type SessionResult } from '@visiondrill/core';
import { QuestionCard } from '@visiondrill/ui';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useDrill } from '../store';

interface Props {
  mode: SessionMode;
  items: SessionItem[];
  startedAt: string;
  onSubmitted: (r: SessionResult) => void;
  onQuit: () => void;
}

function answered(r?: Response) {
  if (!r) return false;
  return r.kind === 'choice' ? r.idx != null : r.text.trim().length > 0;
}

export function QuizPage({ mode, items, startedAt, onSubmitted, onQuit }: Props) {
  const { mutate, learner } = useDrill();
  const [i, setI] = useState(0);
  const [responses, setResponses] = useState<Record<string, Response>>({});
  const [guessed, setGuessed] = useState<Record<string, boolean>>({});
  const item = items[i];
  const q = item.question;
  const done = items.filter((it) => answered(responses[it.question.id])).length;

  const submitting = useRef(false);
  const submit = useCallback(async () => {
    if (submitting.current) return; // 확인 창이 떠 있는 동안 Enter로 중복 제출되는 것 방지
    submitting.current = true;
    try {
      const blank = items.length - done;
      if (
        blank > 0 &&
        !(await askConfirm(`아직 ${blank}문항에 답하지 않았습니다. 이대로 제출할까요? (미응답은 오답 처리)`, '제출', '계속 풀기'))
      )
        return;
      const r = mutate((l) => l.submitSession(mode, items, responses, guessed, startedAt));
      if (r) onSubmitted(r);
    } finally {
      submitting.current = false;
    }
  }, [items, done, mutate, mode, responses, guessed, startedAt, onSubmitted]);

  const next = useCallback(() => setI((x) => Math.min(items.length - 1, x + 1)), [items.length]);
  const prev = useCallback(() => setI((x) => Math.max(0, x - 1)), []);

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.key === 'ArrowRight') next();
      else if (e.key === 'ArrowLeft') prev();
      else if (q.type !== 'short' && /^[1-5]$/.test(e.key)) {
        const idx = item.order[Number(e.key) - 1];
        if (idx) setResponses((r) => ({ ...r, [q.id]: { kind: 'choice', idx } }));
      } else if (e.key === 'Enter') {
        if (i === items.length - 1) submit();
        else next();
      }
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [q, item, i, items.length, next, prev, submit]);

  const imageUrl = q.image_asset_id ? assetUrl(learner, q.image_asset_id) : undefined;
  const modeLabel = { daily: '오늘의 학습', wrong_note: '오답노트 복습', track: '트랙 학습' }[mode];

  return (
    <div className="page">
      <div className="page-head">
        <h1>{modeLabel}</h1>
        <span className="vd-muted">
          {done}/{items.length} 응답
        </span>
        <span className="spacer" />
        <button className="vd-btn vd-btn-small" onClick={onQuit}>
          그만두기
        </button>
      </div>
      <div className="bar" style={{ marginBottom: 12 }}>
        <span style={{ width: `${(done / items.length) * 100}%` }} />
      </div>
      <div className="dots">
        {items.map((it, k) => (
          <button
            key={it.question.id}
            className={`dot ${k === i ? 'is-current' : ''} ${answered(responses[it.question.id]) ? 'is-done' : ''} ${guessed[it.question.id] ? 'is-guessed' : ''}`}
            onClick={() => setI(k)}
            title={`${k + 1}번`}
          >
            {k + 1}
          </button>
        ))}
      </div>

      <QuestionCard
        question={q}
        order={item.order}
        terms={item.terms}
        response={responses[q.id]}
        onResponse={(r) => setResponses((prev) => ({ ...prev, [q.id]: r }))}
        guessed={!!guessed[q.id]}
        onGuessed={(g) => setGuessed((prev) => ({ ...prev, [q.id]: g }))}
        index={i}
        total={items.length}
        imageUrl={imageUrl}
        autoFocus
        onEnter={() => (i === items.length - 1 ? submit() : next())}
      />

      <div className="quiz-nav">
        <button className="vd-btn" onClick={prev} disabled={i === 0}>
          ← 이전
        </button>
        <span className="vd-muted" style={{ fontSize: '0.82rem' }}>
          {q.type === 'short' ? 'Enter: 다음' : '1~5: 보기 선택 · ←/→: 이동 · Enter: 다음'}
        </span>
        {i < items.length - 1 ? (
          <button className="vd-btn" onClick={next}>
            다음 →
          </button>
        ) : (
          <button className="vd-btn vd-btn-primary" onClick={submit}>
            제출하고 채점
          </button>
        )}
      </div>
      {i < items.length - 1 && done === items.length && (
        <div style={{ textAlign: 'center', marginTop: 12 }}>
          <button className="vd-btn vd-btn-primary" onClick={submit}>
            모두 답했습니다 · 제출하고 채점
          </button>
        </div>
      )}
    </div>
  );
}

export function assetUrl(l: Learner, id: number) {
  return assetDataUrl(l.db, id);
}
