import { LEVEL_LABEL, TYPE_LABEL } from '@visiondrill/core';
import { useMemo } from 'react';
import { useBank, useStudio } from '../store';

export function StatsPage() {
  const bank = useBank();
  const { rev } = useStudio();
  const s = useMemo(() => bank.stats(), [bank, rev]);
  const total = s.byTrack.reduce((a, t) => a + t.total, 0);
  const reviewed = s.byTrack.reduce((a, t) => a + (t.reviewed ?? 0), 0);
  const max = Math.max(1, ...s.byTrack.map((t) => t.total));
  return (
    <div className="page">
      <div className="page-head">
        <h1>통계</h1>
      </div>
      <div className="grid-3" style={{ marginBottom: 14 }}>
        <div className="stat">
          <div className="stat-label">전체 문항</div>
          <div className="stat-value">{total}</div>
          <div className="stat-sub">검수 완료 {reviewed}</div>
        </div>
        <div className="stat">
          <div className="stat-label">용어</div>
          <div className="stat-value">{s.terms.total}</div>
          <div className="stat-sub">
            프로젝트 정의 {s.terms.project ?? 0} · 미연결 {s.terms.unlinked ?? 0}
          </div>
        </div>
        <div className="stat">
          <div className="stat-label">검수 진행률</div>
          <div className="stat-value">{total ? Math.round((reviewed / total) * 100) : 0}%</div>
          <div className="bar" style={{ marginTop: 6 }}>
            <span style={{ width: `${total ? (reviewed / total) * 100 : 0}%` }} />
          </div>
        </div>
      </div>
      <div className="section">
        <h3>트랙별 문항</h3>
        <table className="table">
          <thead>
            <tr>
              <th>트랙</th>
              <th style={{ width: '40%' }} />
              <th>초안</th>
              <th>검수 완료</th>
              <th>출제 중단</th>
              <th>합계</th>
            </tr>
          </thead>
          <tbody>
            {s.byTrack.map((t) => (
              <tr key={t.track_id}>
                <td>
                  {t.track_id}. {t.name_ko}
                </td>
                <td>
                  <div className="bar">
                    <span style={{ width: `${(t.total / max) * 100}%` }} />
                  </div>
                </td>
                <td>{t.draft ?? 0}</td>
                <td>{t.reviewed ?? 0}</td>
                <td>{t.retired ?? 0}</td>
                <td>
                  <strong>{t.total}</strong>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid-2">
        <div className="section">
          <h3>유형별</h3>
          {s.byType.map((r) => (
            <div key={r.type} style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>{TYPE_LABEL[r.type as keyof typeof TYPE_LABEL]}</span>
              <strong>{r.n}</strong>
            </div>
          ))}
          {!s.byType.length && <span className="vd-muted">문항 없음</span>}
        </div>
        <div className="section">
          <h3>인지 단계별</h3>
          {s.byLevel.map((r) => (
            <div key={r.level} style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>
                {r.level}. {LEVEL_LABEL[r.level]}
              </span>
              <strong>{r.n}</strong>
            </div>
          ))}
          {!s.byLevel.length && <span className="vd-muted">문항 없음</span>}
        </div>
      </div>
    </div>
  );
}
