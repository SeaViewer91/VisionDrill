import { isTauri } from '@visiondrill/platform';
import { useState } from 'react';
import { useStudio } from './store';
import { CurriculumPage } from './pages/CurriculumPage';
import { QuestionsPage } from './pages/QuestionsPage';
import { SettingsPage } from './pages/SettingsPage';
import { StatsPage } from './pages/StatsPage';
import { TermsPage } from './pages/TermsPage';
import { TransferPage } from './pages/TransferPage';
import { WelcomePage } from './pages/WelcomePage';

type PageId = 'questions' | 'terms' | 'curriculum' | 'transfer' | 'stats' | 'settings';

const NAV: { id: PageId; label: string }[] = [
  { id: 'questions', label: '문항' },
  { id: 'terms', label: '용어' },
  { id: 'curriculum', label: '커리큘럼' },
  { id: 'transfer', label: '가져오기·내보내기' },
  { id: 'stats', label: '통계' },
  { id: 'settings', label: '설정' },
];

const SAVE_LABEL = { saved: '저장됨', dirty: '변경됨', saving: '저장 중…', error: '저장 실패' } as const;

export default function App() {
  const { bank, filePath, saveState, lastSaved, rev, reviewer } = useStudio();
  const [page, setPage] = useState<PageId>('questions');
  if (!bank) return <WelcomePage />;
  void rev;
  const counts = bank.db.get<{ q: number; t: number; d: number }>(
    `SELECT (SELECT COUNT(*) FROM questions) AS q, (SELECT COUNT(*) FROM terms) AS t,
            (SELECT COUNT(*) FROM questions WHERE status='draft') AS d`,
  )!;
  const fileName = filePath ? filePath.split(/[\\/]/).pop() : '브라우저 저장소';

  return (
    <div className="app">
      <aside className="side">
        <div className="brand">
          VisionDrill Studio
          <small>문제은행 관리</small>
        </div>
        {NAV.map((n) => (
          <button key={n.id} className={`nav-btn ${page === n.id ? 'is-active' : ''}`} onClick={() => setPage(n.id)}>
            <span>{n.label}</span>
            {n.id === 'questions' && <span className="nav-badge">{counts.q}</span>}
            {n.id === 'terms' && <span className="nav-badge">{counts.t}</span>}
          </button>
        ))}
        <div className="side-foot">
          <span title={filePath ?? undefined}>📄 {fileName}</span>
          <span className={`save-state is-${saveState}`}>
            ● {SAVE_LABEL[saveState]}
            {saveState === 'saved' && lastSaved
              ? ` ${lastSaved.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`
              : ''}
          </span>
          <span>초안 {counts.d}개</span>
          {!reviewer && (
            <button className="vd-link" onClick={() => setPage('settings')}>
              검수자 이름 설정 필요
            </button>
          )}
          {!isTauri() && <span>브라우저 미리보기 모드</span>}
        </div>
      </aside>
      <main className="main">
        {page === 'questions' && <QuestionsPage />}
        {page === 'terms' && <TermsPage />}
        {page === 'curriculum' && <CurriculumPage />}
        {page === 'transfer' && <TransferPage />}
        {page === 'stats' && <StatsPage />}
        {page === 'settings' && <SettingsPage />}
      </main>
    </div>
  );
}
