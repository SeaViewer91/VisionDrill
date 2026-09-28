import type { Chapter } from '@visiondrill/core';
import { askConfirm, showMessage } from '@visiondrill/platform';
import { useMemo, useState } from 'react';
import { useBank, useStudio } from '../store';

export function CurriculumPage() {
  const bank = useBank();
  const { rev, mutate } = useStudio();
  const tracks = useMemo(() => bank.tracks(), [bank, rev]);
  const prereqs = useMemo(() => bank.trackPrereqs(), [bank, rev]);
  const chapters = useMemo(() => bank.chapters(), [bank, rev]);
  const counts = useMemo(
    () =>
      new Map(
        bank.db
          .all<{ chapter_id: string; n: number }>(`SELECT chapter_id, COUNT(*) AS n FROM questions GROUP BY chapter_id`)
          .map((r) => [r.chapter_id, r.n]),
      ),
    [bank, rev],
  );
  const [edit, setEdit] = useState<Chapter | null>(null);
  const [isNew, setIsNew] = useState(false);

  const startNew = (trackId: string) => {
    const n = chapters.filter((c) => c.track_id === trackId).length + 1;
    setEdit({ id: `${trackId}.`, track_id: trackId, code: '', name_ko: '', name_en: '', sort_order: n, stage: 1, source_ref: '' });
    setIsNew(true);
  };

  return (
    <div className="page page-wide">
      <div className="page-head">
        <h1>커리큘럼</h1>
        <span className="vd-muted">트랙 → 장. 문항 ID는 '트랙-장코드-번호'로 만들어집니다.</span>
      </div>
      <div className="split">
        <div>
          {tracks.map((t) => {
            const pre = prereqs.filter((p) => p.track_id === t.id);
            const chs = chapters.filter((c) => c.track_id === t.id);
            return (
              <div className="section" key={t.id}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <h3 style={{ margin: 0 }}>
                    {t.id}. {t.name_ko}{' '}
                    <span className="vd-muted" style={{ fontWeight: 400 }}>
                      {t.name_en}
                    </span>
                  </h3>
                  <span style={{ marginLeft: 'auto' }} className="vd-muted">
                    {pre.length
                      ? `선수: ${pre.map((p) => `${p.prereq_track_id}(${Math.round(p.min_mastery * 100)}%)`).join(', ')}`
                      : '선수 없음'}
                  </span>
                  <button className="vd-btn vd-btn-small" onClick={() => startNew(t.id)}>
                    + 장
                  </button>
                </div>
                {chs.length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    {chs.map((c) => (
                      <div className="chapter-row" key={c.id}>
                        <span className="qitem-id">{c.id}</span>
                        <span>
                          {c.name_ko} <span className="vd-muted">{c.name_en}</span>
                        </span>
                        <span className="vd-chip vd-chip-muted">{c.code}</span>
                        <span className="vd-muted">{counts.get(c.id) ?? 0}문항</span>
                        <button className="vd-link" onClick={() => (setEdit({ ...c }), setIsNew(false))}>
                          편집
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        {edit ? (
          <div className="section" style={{ position: 'sticky', top: 0 }}>
            <h3>{isNew ? `새 장 (${edit.track_id} 트랙)` : edit.id}</h3>
            {isNew && (
              <label className="field">
                <span>장 ID (예: F.nms)</span>
                <input className="vd-input" value={edit.id} onChange={(e) => setEdit({ ...edit, id: e.target.value.trim() })} />
              </label>
            )}
            <div className="field-row">
              <label className="field">
                <span>장 코드 (문항 ID에 사용, 영문 대문자)</span>
                <input
                  className="vd-input"
                  value={edit.code}
                  onChange={(e) => setEdit({ ...edit, code: e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '') })}
                />
              </label>
              <label className="field" style={{ maxWidth: 90 }}>
                <span>순서</span>
                <input
                  className="vd-input"
                  type="number"
                  value={edit.sort_order}
                  onChange={(e) => setEdit({ ...edit, sort_order: Number(e.target.value) })}
                />
              </label>
            </div>
            <label className="field">
              <span>이름 (한글)</span>
              <input className="vd-input" value={edit.name_ko} onChange={(e) => setEdit({ ...edit, name_ko: e.target.value })} />
            </label>
            <label className="field">
              <span>이름 (영문)</span>
              <input className="vd-input" value={edit.name_en ?? ''} onChange={(e) => setEdit({ ...edit, name_en: e.target.value })} />
            </label>
            <div className="field-row">
              <label className="field">
                <span>개방 차수 (H 트랙용)</span>
                <select className="vd-select" value={edit.stage} onChange={(e) => setEdit({ ...edit, stage: Number(e.target.value) })}>
                  <option value={1}>1차</option>
                  <option value={2}>2차</option>
                  <option value={3}>3차</option>
                </select>
              </label>
            </div>
            <label className="field">
              <span>출처 (원문 경로)</span>
              <input
                className="vd-input"
                value={edit.source_ref ?? ''}
                onChange={(e) => setEdit({ ...edit, source_ref: e.target.value })}
              />
            </label>
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                className="vd-btn vd-btn-primary"
                onClick={() => {
                  if (!edit.code || !edit.name_ko || !/^[A-H]\..+/.test(edit.id))
                    return void showMessage('장 ID(예: F.nms), 코드, 한글 이름을 입력하세요.');
                  const ok = mutate(
                    (b) => (b.saveChapter({ ...edit, name_en: edit.name_en || null, source_ref: edit.source_ref || null }), true),
                    '저장했습니다',
                  );
                  if (ok) setIsNew(false);
                }}
              >
                저장
              </button>
              <button className="vd-btn" onClick={() => setEdit(null)}>
                닫기
              </button>
              {!isNew && (
                <button
                  className="vd-btn vd-btn-danger"
                  style={{ marginLeft: 'auto' }}
                  onClick={async () =>
                    (await askConfirm('이 장을 삭제할까요?', '삭제')) &&
                    mutate((b) => (b.deleteChapter(edit.id), setEdit(null), true), '삭제했습니다')
                  }
                >
                  삭제
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="section vd-muted">
            트랙 해금 기준(선수 트랙과 숙련도)은 현재 기본값으로 고정되어 있습니다. 장을 추가하거나 편집하려면 왼쪽에서 고르세요.
          </div>
        )}
      </div>
    </div>
  );
}
