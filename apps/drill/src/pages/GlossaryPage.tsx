import { useMemo, useState } from 'react';
import { useDrill } from '../store';

export function GlossaryPage() {
  const { learner, rev } = useDrill();
  const [q, setQ] = useState('');
  const [origin, setOrigin] = useState<'' | 'standard' | 'project'>('');
  const terms = useMemo(() => learner.glossary(), [learner, rev]);
  const tracks = useMemo(() => new Map(learner.trackStatus().map((t) => [t.id, t.name_ko])), [learner, rev]);
  const list = terms.filter(
    (t) =>
      (!origin || t.origin === origin) &&
      (!q.trim() || [t.term_en, t.term_ko, t.abbreviation, t.definition].some((s) => s?.toLowerCase().includes(q.trim().toLowerCase()))),
  );
  return (
    <div className="page">
      <div className="page-head">
        <h1>용어 사전</h1>
        <span className="vd-muted">{terms.length}개</span>
        <span className="spacer" />
        <input
          className="vd-input"
          style={{ width: 220 }}
          placeholder="영문·한글·약어 검색"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        <select className="vd-select" style={{ width: 150 }} value={origin} onChange={(e) => setOrigin(e.target.value as any)}>
          <option value="">전체</option>
          <option value="standard">업계 표준</option>
          <option value="project">프로젝트 정의</option>
        </select>
      </div>
      <p className="vd-muted" style={{ marginTop: -8, fontSize: '0.88rem' }}>
        <span className="vd-chip vd-chip-project">프로젝트 정의</span> 표시는 사내(Diagnostics)에서 정의한 용어입니다. 외부와 이야기할 때는
        풀어서 설명하세요.
      </p>
      <div style={{ display: 'grid', gap: 8 }}>
        {list.map((t) => (
          <div className="vd-card glossary-item" key={t.id}>
            <div className="glossary-head">
              <strong>{t.abbreviation && t.abbreviation !== t.term_en ? t.abbreviation : t.term_en}</strong>
              {t.abbreviation && t.abbreviation !== t.term_en && <span>{t.term_en}</span>}
              {t.term_ko && <span className="vd-muted">{t.term_ko}</span>}
              {t.origin === 'project' && <span className="vd-chip vd-chip-project">프로젝트 정의</span>}
              {t.track_id && (
                <span className="vd-chip" style={{ marginLeft: 'auto' }}>
                  {t.track_id}. {tracks.get(t.track_id)}
                </span>
              )}
            </div>
            <div>{t.definition}</div>
            {t.usage_example && <div className="vd-term-usage">“{t.usage_example}”</div>}
          </div>
        ))}
        {!list.length && <div className="empty">찾는 용어가 없습니다</div>}
      </div>
    </div>
  );
}
