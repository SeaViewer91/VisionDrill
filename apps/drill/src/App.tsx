import { askConfirm } from '@visiondrill/platform';
import type { SessionItem, SessionMode, SessionResult } from '@visiondrill/core';
import { useState } from 'react';
import { GlossaryPage } from './pages/GlossaryPage';
import { HomePage } from './pages/HomePage';
import { QuizPage } from './pages/QuizPage';
import { ResultPage } from './pages/ResultPage';
import { SettingsPage } from './pages/SettingsPage';
import { WrongNotePage } from './pages/WrongNotePage';
import { useDrill } from './store';

export type View =
  | { name: 'home' }
  | { name: 'quiz'; mode: SessionMode; items: SessionItem[]; startedAt: string }
  | { name: 'result'; mode: SessionMode; result: SessionResult }
  | { name: 'wrong' }
  | { name: 'glossary' }
  | { name: 'settings' };

const NAV: { name: View['name']; label: string }[] = [
  { name: 'home', label: '오늘의 학습' },
  { name: 'wrong', label: '오답노트' },
  { name: 'glossary', label: '용어 사전' },
  { name: 'settings', label: '설정' },
];

export default function App() {
  const { learner, rev, saveError } = useDrill();
  const [view, setView] = useState<View>({ name: 'home' });
  void rev;
  const stats = learner.stats();
  const pack = learner.packInfo();
  const inQuiz = view.name === 'quiz';

  const go = async (v: View) => {
    if (
      inQuiz &&
      v.name !== 'result' &&
      !(await askConfirm('풀던 세션을 그만둘까요? 제출하지 않은 답은 저장되지 않습니다.', '그만두기', '계속 풀기'))
    )
      return;
    setView(v);
  };

  return (
    <div className="app">
      <aside className="side">
        <div className="brand">
          VisionDrill
          <small>비전 용어·개념 반복학습</small>
        </div>
        {NAV.map((n) => (
          <button
            key={n.name}
            className={`nav-btn ${view.name === n.name || (n.name === 'home' && (view.name === 'quiz' || view.name === 'result')) ? 'is-active' : ''}`}
            onClick={() => go({ name: n.name } as View)}
          >
            <span>{n.label}</span>
            {n.name === 'home' && stats.due > 0 && <span className="nav-badge">{stats.due}</span>}
            {n.name === 'wrong' && stats.wrongNotes > 0 && <span className="nav-badge">{stats.wrongNotes}</span>}
          </button>
        ))}
        <div className="side-foot">
          <span>🔥 연속 {stats.streakDays}일</span>
          <span>{pack ? `문제 팩 ${pack.pack_version} · ${pack.question_count}문항` : '문제 팩 없음'}</span>
          {saveError && <span style={{ color: 'var(--bad)' }}>⚠ 학습 기록 저장 실패: {saveError}</span>}
        </div>
      </aside>
      <main className="main">
        {view.name === 'home' && (
          <HomePage onStart={(mode, items) => setView({ name: 'quiz', mode, items, startedAt: new Date().toISOString() })} />
        )}
        {view.name === 'quiz' && (
          <QuizPage
            key={view.startedAt}
            mode={view.mode}
            items={view.items}
            startedAt={view.startedAt}
            onSubmitted={(result) => setView({ name: 'result', mode: view.mode, result })}
            onQuit={() => go({ name: 'home' })}
          />
        )}
        {view.name === 'result' && (
          <ResultPage
            result={view.result}
            onHome={() => setView({ name: 'home' })}
            onRetry={(mode, items) => setView({ name: 'quiz', mode, items, startedAt: new Date().toISOString() })}
          />
        )}
        {view.name === 'wrong' && (
          <WrongNotePage onStart={(items) => setView({ name: 'quiz', mode: 'wrong_note', items, startedAt: new Date().toISOString() })} />
        )}
        {view.name === 'glossary' && <GlossaryPage />}
        {view.name === 'settings' && <SettingsPage />}
      </main>
    </div>
  );
}
