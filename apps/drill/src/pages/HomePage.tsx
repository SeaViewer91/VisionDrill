import type { SessionItem, SessionMode } from '@visiondrill/core';
import { useDrill } from '../store';

export function HomePage({ onStart }: { onStart: (mode: SessionMode, items: SessionItem[]) => void }) {
  const { learner, toast, importPackFile } = useDrill();
  const stats = learner.stats();
  const tracks = learner.trackStatus();
  const settings = learner.settings();
  const pack = learner.packInfo();
  const unseen = stats.questions - stats.seen;

  const start = (mode: SessionMode, trackId?: string) => {
    const items = learner.buildSession(mode, trackId);
    if (!items.length)
      return toast(mode === 'wrong_note' ? '오답노트가 비어 있습니다' : '풀 문항이 없습니다. 트랙 해금 조건을 확인하세요.');
    onStart(mode, items);
  };

  if (!pack) {
    return (
      <div className="page">
        <div className="empty">
          <h2>문제 팩이 없습니다</h2>
          <p>Studio에서 만든 .vdpack 파일을 가져오세요.</p>
          <button className="vd-btn vd-btn-primary" onClick={importPackFile}>
            문제 팩 가져오기
          </button>
        </div>
      </div>
    );
  }

  const todayCount =
    Math.min(settings.sessionSize, stats.due + Math.min(settings.newPerSession, unseen)) || Math.min(settings.sessionSize, stats.seen);

  return (
    <div className="page">
      <div className="page-head">
        <h1>오늘의 학습</h1>
      </div>

      <div className="hero">
        <div>
          <div className="hero-title">
            복습 {stats.due}문항 · 새 문항 {Math.min(settings.newPerSession, unseen)}문항
          </div>
          <div className="vd-muted">다 풀고 제출하면 한꺼번에 채점합니다. 틀린 문항은 해설이 바로 펼쳐지고 오답노트에 들어갑니다.</div>
        </div>
        <button className="vd-btn vd-btn-primary vd-btn-lg" onClick={() => start('daily')}>
          시작하기 ({todayCount}문항)
        </button>
      </div>

      <div className="grid-3" style={{ margin: '14px 0' }}>
        <div className="stat">
          <div className="stat-label">연속 학습</div>
          <div className="stat-value">{stats.streakDays}일</div>
          <div className="stat-sub">총 {stats.sessions}회</div>
        </div>
        <div className="stat">
          <div className="stat-label">오답노트</div>
          <div className="stat-value">{stats.wrongNotes}</div>
          <button className="vd-link" onClick={() => start('wrong_note')} disabled={!stats.wrongNotes}>
            오답만 다시 풀기 →
          </button>
        </div>
        <div className="stat">
          <div className="stat-label">최근 7일 정답률</div>
          <div className="stat-value">{stats.weekAccuracy == null ? '—' : `${Math.round(stats.weekAccuracy * 100)}%`}</div>
          <div className="stat-sub">
            푼 문항 {stats.seen} / {stats.questions}
          </div>
        </div>
      </div>

      <div className="section-card">
        <h2>트랙</h2>
        <p className="vd-muted" style={{ marginTop: 0, fontSize: '0.88rem' }}>
          숙련도 = 3회 이상 연속으로 맞혀 복습 간격이 3일 이상으로 늘어난 문항 비율. 선수 트랙 숙련도가 기준을 넘으면 다음 트랙이 열립니다.
        </p>
        {tracks.map((t) => (
          <div className={`track-row ${t.unlocked ? '' : 'is-locked'}`} key={t.id}>
            <div className="track-name">
              <strong>
                {t.unlocked ? '' : '🔒 '}
                {t.id}. {t.name_ko}
              </strong>
              <span className="vd-muted" style={{ fontSize: '0.82rem' }}>
                {t.total ? `${t.mastered}/${t.total} 숙련` : '문항 없음'}
                {!t.unlocked &&
                  ` · 필요: ${t.prereqs
                    .filter((p) => p.mastery < p.min_mastery)
                    .map((p) => `${p.track_id} ${Math.round(p.mastery * 100)}%→${Math.round(p.min_mastery * 100)}%`)
                    .join(', ')}`}
              </span>
            </div>
            <div className="bar" style={{ flex: 1 }}>
              <span style={{ width: `${t.total ? t.mastery * 100 : 0}%` }} />
            </div>
            <span className="track-pct">{t.total ? `${Math.round(t.mastery * 100)}%` : ''}</span>
            <button className="vd-btn vd-btn-small" disabled={!t.unlocked || !t.total} onClick={() => start('track', t.id)}>
              이 트랙만
            </button>
          </div>
        ))}
      </div>

      {stats.recent.length > 0 && (
        <div className="section-card">
          <h2>최근 세션</h2>
          <table className="table">
            <tbody>
              {stats.recent.map((s) => (
                <tr key={s.id}>
                  <td>
                    {new Date(s.submitted_at).toLocaleString('ko-KR', {
                      month: 'numeric',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </td>
                  <td>{{ daily: '오늘의 학습', wrong_note: '오답노트', track: '트랙' }[s.mode as SessionMode]}</td>
                  <td>
                    {s.correct}/{s.total}
                  </td>
                  <td>{Math.round((s.correct / Math.max(1, s.total)) * 100)}%</td>
                  <td className="vd-muted">{Math.round(s.duration_sec / 60)}분</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
