import { STATUS_LABEL, type Status, type Term, type TermOrigin } from '@visiondrill/core';
import { askConfirm } from '@visiondrill/platform';
import { useMemo, useState } from 'react';
import { useBank, useStudio } from '../store';

const EMPTY: Term = {
  id: '',
  term_en: '',
  term_ko: '',
  abbreviation: '',
  definition: '',
  usage_example: '',
  origin: 'standard',
  track_id: null,
  status: 'reviewed',
};

export function TermsPage() {
  const bank = useBank();
  const { rev, mutate } = useStudio();
  const [q, setQ] = useState('');
  const [origin, setOrigin] = useState<'' | TermOrigin>('');
  const [edit, setEdit] = useState<Term | null>(null);
  const [isNew, setIsNew] = useState(false);
  const tracks = useMemo(() => bank.tracks(), [bank, rev]);
  const terms = useMemo(() => bank.terms(), [bank, rev]);
  const list = terms.filter(
    (t) =>
      (!origin || t.origin === origin) &&
      (!q.trim() ||
        [t.id, t.term_en, t.term_ko, t.abbreviation, t.definition].some((s) => s?.toLowerCase().includes(q.trim().toLowerCase()))),
  );

  const save = () => {
    if (!edit) return;
    const saved = mutate((b) => b.saveTerm(edit), isNew ? '용어를 추가했습니다' : '저장했습니다');
    if (saved) (setEdit(saved), setIsNew(false));
  };

  return (
    <div className="page page-wide">
      <div className="page-head">
        <h1>용어</h1>
        <span className="vd-muted">{terms.length}개</span>
        <span className="spacer" />
        <input className="vd-input" style={{ width: 240 }} placeholder="검색" value={q} onChange={(e) => setQ(e.target.value)} />
        <select className="vd-select" style={{ width: 160 }} value={origin} onChange={(e) => setOrigin(e.target.value as any)}>
          <option value="">출처 전체</option>
          <option value="standard">업계 표준</option>
          <option value="project">프로젝트 정의</option>
        </select>
        <button className="vd-btn vd-btn-primary" onClick={() => (setEdit({ ...EMPTY }), setIsNew(true))}>
          + 새 용어
        </button>
      </div>
      <div className="split">
        <div className="section" style={{ padding: 0, overflow: 'hidden' }}>
          <table className="table">
            <thead>
              <tr>
                <th>용어</th>
                <th>한글</th>
                <th>정의</th>
                <th>문항</th>
              </tr>
            </thead>
            <tbody>
              {list.map((t) => (
                <tr
                  key={t.id}
                  className={`is-clickable ${edit?.id === t.id && !isNew ? 'is-selected' : ''}`}
                  onClick={() => (setEdit({ ...t }), setIsNew(false))}
                >
                  <td>
                    <strong>{t.abbreviation ?? t.term_en}</strong>
                    {t.abbreviation && (
                      <div className="vd-muted" style={{ fontSize: '0.82rem' }}>
                        {t.term_en}
                      </div>
                    )}
                    {t.origin === 'project' && (
                      <span className="vd-chip vd-chip-project vd-chip-xs" style={{ marginLeft: 0 }}>
                        프로젝트 정의
                      </span>
                    )}
                    {t.status !== 'reviewed' && <span className={`vd-chip vd-chip-${t.status} vd-chip-xs`}>{STATUS_LABEL[t.status]}</span>}
                  </td>
                  <td>{t.term_ko}</td>
                  <td className="vd-muted">{t.definition}</td>
                  <td style={{ textAlign: 'right' }}>{t.question_count || <span className="vd-placeholder">0</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!list.length && <div className="empty">용어가 없습니다</div>}
        </div>

        {edit ? (
          <div className="section" style={{ position: 'sticky', top: 0 }}>
            <h3>{isNew ? '새 용어' : edit.id}</h3>
            {isNew && (
              <label className="field">
                <span>ID (영문 소문자·하이픈, 예: cooks-distance)</span>
                <input className="vd-input" value={edit.id} onChange={(e) => setEdit({ ...edit, id: e.target.value })} />
              </label>
            )}
            <label className="field">
              <span>영문 용어</span>
              <input className="vd-input" value={edit.term_en} onChange={(e) => setEdit({ ...edit, term_en: e.target.value })} />
            </label>
            <div className="field-row">
              <label className="field">
                <span>한글</span>
                <input className="vd-input" value={edit.term_ko ?? ''} onChange={(e) => setEdit({ ...edit, term_ko: e.target.value })} />
              </label>
              <label className="field">
                <span>약어</span>
                <input
                  className="vd-input"
                  value={edit.abbreviation ?? ''}
                  onChange={(e) => setEdit({ ...edit, abbreviation: e.target.value })}
                />
              </label>
            </div>
            <label className="field">
              <span>한 줄 정의</span>
              <textarea
                className="vd-textarea"
                rows={2}
                value={edit.definition}
                onChange={(e) => setEdit({ ...edit, definition: e.target.value })}
              />
            </label>
            <label className="field">
              <span>실무 예문 ("이렇게 말한다")</span>
              <textarea
                className="vd-textarea"
                rows={2}
                value={edit.usage_example ?? ''}
                onChange={(e) => setEdit({ ...edit, usage_example: e.target.value })}
              />
            </label>
            <div className="field-row">
              <label className="field">
                <span>출처 구분</span>
                <select
                  className="vd-select"
                  value={edit.origin}
                  onChange={(e) => setEdit({ ...edit, origin: e.target.value as TermOrigin })}
                >
                  <option value="standard">업계 표준</option>
                  <option value="project">프로젝트 정의</option>
                </select>
              </label>
              <label className="field">
                <span>트랙</span>
                <select
                  className="vd-select"
                  value={edit.track_id ?? ''}
                  onChange={(e) => setEdit({ ...edit, track_id: e.target.value || null })}
                >
                  <option value="">-</option>
                  {tracks.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.id}. {t.name_ko}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>상태</span>
                <select className="vd-select" value={edit.status} onChange={(e) => setEdit({ ...edit, status: e.target.value as Status })}>
                  {Object.entries(STATUS_LABEL).map(([k, v]) => (
                    <option key={k} value={k}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <button className="vd-btn vd-btn-primary" onClick={save}>
                저장
              </button>
              <button className="vd-btn" onClick={() => setEdit(null)}>
                닫기
              </button>
              {!isNew && (
                <button
                  className="vd-btn vd-btn-danger"
                  style={{ marginLeft: 'auto' }}
                  onClick={async () => {
                    if (!(await askConfirm(`${edit.id} 용어를 삭제할까요?`, '삭제'))) return;
                    if (mutate((b) => (b.deleteTerm(edit.id), true), '삭제했습니다')) setEdit(null);
                  }}
                >
                  삭제
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="section vd-muted">
            용어를 고르면 여기서 편집합니다. 문항에 연결된 용어는 삭제할 수 없고, 상태를 출제 중단으로 바꿀 수 있습니다.
          </div>
        )}
      </div>
    </div>
  );
}
