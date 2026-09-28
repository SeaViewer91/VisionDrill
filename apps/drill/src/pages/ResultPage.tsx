import type { ItemResult, SessionItem, SessionMode, SessionResult } from '@visiondrill/core';
import { openExternal } from '@visiondrill/platform';
import { ResultCard } from '@visiondrill/ui';
import { useState } from 'react';
import { REPO_URL, useDrill } from '../store';
import { assetUrl } from './QuizPage';

type Filter = 'all' | 'wrong' | 'correct';

export function reportUrl(it: SessionItem, packVersion?: string) {
  const q = it.question;
  const title = `[문항 오류] ${q.id}`;
  const body = [
    `- 문항: ${q.id} (v${q.version})`,
    `- 문제 팩: ${packVersion ?? '-'}`,
    '',
    '### 문제 본문',
    q.stem,
    '',
    '### 어떤 점이 잘못되었나요?',
    '(정답 오류 / 해설 오류 / 보기 모호 / 오타 등)',
    '',
  ].join('\n');
  return `${REPO_URL}/issues/new?labels=question-error&title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
}

export function ResultPage({
  result,
  onHome,
  onRetry,
}: {
  result: SessionResult;
  onHome: () => void;
  onRetry: (mode: SessionMode, items: SessionItem[]) => void;
}) {
  const { mutate, learner } = useDrill();
  const [filter, setFilter] = useState<Filter>('all');
  const [overrides, setOverrides] = useState<Record<number, boolean>>({});
  const pack = learner.packInfo();
  const isCorrect = (r: ItemResult) => r.grade.correct || overrides[r.attemptId];
  const correct = result.items.filter(isCorrect).length;
  const pct = Math.round((correct / Math.max(1, result.total)) * 100);
  const wrongItems = result.items.filter((r) => !isCorrect(r));
  const guessedRight = result.items.filter((r) => isCorrect(r) && r.guessed).length;
  const shown = result.items.filter((r) => (filter === 'all' ? true : filter === 'wrong' ? !isCorrect(r) : isCorrect(r)));

  return (
    <div className="page">
      <div className="score">
        <div className={`score-ring ${pct >= 80 ? 'is-good' : pct >= 50 ? 'is-mid' : 'is-low'}`}>
          <strong>{pct}</strong>
          <span>점</span>
        </div>
        <div style={{ flex: 1 }}>
          <h1 style={{ marginBottom: 4 }}>
            {correct} / {result.total} 정답
          </h1>
          <div className="vd-muted">
            오답 {wrongItems.length} · 찍어서 맞힘 {guessedRight} · {Math.max(1, Math.round(result.durationSec / 60))}분
          </div>
          <div className="vd-muted" style={{ fontSize: '0.85rem', marginTop: 4 }}>
            틀린 문항과 찍어서 맞힌 문항은 오답노트에 들어갔고, 다음 학습에 다시 나옵니다.
          </div>
        </div>
        <div style={{ display: 'grid', gap: 8 }}>
          {wrongItems.length > 0 && (
            <button
              className="vd-btn vd-btn-primary"
              onClick={() =>
                onRetry(
                  'wrong_note',
                  wrongItems.map((r) => ({ ...r.item, order: [...r.item.order].sort(() => Math.random() - 0.5) })),
                )
              }
            >
              틀린 {wrongItems.length}문항 다시 풀기
            </button>
          )}
          <button className="vd-btn" onClick={onHome}>
            홈으로
          </button>
        </div>
      </div>

      <div className="tabs">
        {(
          [
            ['all', `전체 ${result.total}`],
            ['wrong', `오답 ${wrongItems.length}`],
            ['correct', `정답 ${correct}`],
          ] as [Filter, string][]
        ).map(([k, label]) => (
          <button key={k} className={`tab ${filter === k ? 'is-active' : ''}`} onClick={() => setFilter(k)}>
            {label}
          </button>
        ))}
      </div>

      <div style={{ display: 'grid', gap: 12 }}>
        {shown.map((r) => (
          <ResultCard
            key={r.attemptId}
            question={r.item.question}
            order={r.item.order}
            terms={r.item.terms}
            response={r.response}
            grade={r.grade}
            guessed={r.guessed}
            overridden={overrides[r.attemptId]}
            index={result.items.indexOf(r)}
            imageUrl={r.item.question.image_asset_id ? assetUrl(learner, r.item.question.image_asset_id) : undefined}
            onAcceptNearMiss={() => {
              if (mutate((l) => (l.acceptNearMiss(r.attemptId), true), '정답으로 인정했습니다'))
                setOverrides((o) => ({ ...o, [r.attemptId]: true }));
            }}
            onReport={() => openExternal(reportUrl(r.item, pack?.pack_version))}
          />
        ))}
        {!shown.length && <div className="empty">해당하는 문항이 없습니다</div>}
      </div>
    </div>
  );
}
