import {
  LEVEL_LABEL,
  STATUS_LABEL,
  TYPE_LABEL,
  emptyQuestion,
  type QType,
  type Question,
  type QuestionFilter,
  type Status,
} from '@visiondrill/core';
import { askConfirm, showMessage } from '@visiondrill/platform';
import { useEffect, useMemo, useState } from 'react';
import { useBank, useStudio } from '../store';
import { QuestionEditor } from './QuestionEditor';

export function QuestionsPage() {
  const bank = useBank();
  const { rev } = useStudio();
  const [f, setF] = useState<QuestionFilter>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [draftNew, setDraftNew] = useState<Question | null>(null);
  const [dirty, setDirty] = useState(false);

  const tracks = useMemo(() => bank.tracks(), [bank, rev]);
  const chapters = useMemo(() => bank.chapters(f.track), [bank, rev, f.track]);
  const allChapters = useMemo(() => bank.chapters(), [bank, rev]);
  const items = useMemo(() => bank.listQuestions(f), [bank, rev, f]);

  useEffect(() => {
    if (!selected && !draftNew && items.length) setSelected(items[0].id);
  }, [items, selected, draftNew]);

  const guard = async () => !dirty || askConfirm('저장하지 않은 변경 사항이 있습니다. 버리고 이동할까요?', '버리고 이동');

  const select = async (id: string) => {
    if (id === selected || !(await guard())) return;
    setDraftNew(null);
    setSelected(id);
    setDirty(false);
  };

  const startNew = async () => {
    if (!(await guard())) return;
    const chapterId = f.chapter ?? chapters[0]?.id ?? allChapters[0]?.id;
    if (!chapterId) {
      await showMessage('먼저 커리큘럼 화면에서 장(chapter)을 하나 이상 만드세요.');
      return;
    }
    const type: QType = f.type ?? 'mcq';
    setDraftNew(emptyQuestion(bank.nextQuestionId(chapterId), chapterId, type));
    setSelected(null);
    setDirty(false);
  };

  const set = (patch: Partial<QuestionFilter>) => setF((prev) => ({ ...prev, ...patch }));

  return (
    <div className="qpage">
      <section className="qlist">
        <div className="qlist-filters">
          <input
            className="vd-input"
            placeholder="검색: ID, 본문, 보기, 해설, 허용 답, 용어"
            value={f.text ?? ''}
            onChange={(e) => set({ text: e.target.value || undefined })}
          />
          <div className="row">
            <select
              className="vd-select"
              value={f.track ?? ''}
              onChange={(e) => set({ track: e.target.value || undefined, chapter: undefined })}
            >
              <option value="">전체 트랙</option>
              {tracks.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.id}. {t.name_ko}
                </option>
              ))}
            </select>
            <select className="vd-select" value={f.chapter ?? ''} onChange={(e) => set({ chapter: e.target.value || undefined })}>
              <option value="">전체 장</option>
              {chapters.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.id} {c.name_ko}
                </option>
              ))}
            </select>
            <select className="vd-select" value={f.type ?? ''} onChange={(e) => set({ type: (e.target.value || undefined) as QType })}>
              <option value="">전체 유형</option>
              {Object.entries(TYPE_LABEL).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
            <select className="vd-select" value={f.status ?? ''} onChange={(e) => set({ status: (e.target.value || undefined) as Status })}>
              <option value="">전체 상태</option>
              {Object.entries(STATUS_LABEL).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
            <select className="vd-select" value={f.level ?? ''} onChange={(e) => set({ level: Number(e.target.value) || undefined })}>
              <option value="">전체 인지 단계</option>
              {[1, 2, 3].map((l) => (
                <option key={l} value={l}>
                  {l}. {LEVEL_LABEL[l]}
                </option>
              ))}
            </select>
            <select
              className="vd-select"
              value={f.origin ?? ''}
              onChange={(e) => set({ origin: (e.target.value || undefined) as QuestionFilter['origin'] })}
            >
              <option value="">용어 출처 전체</option>
              <option value="standard">업계 표준만</option>
              <option value="project">프로젝트 정의 포함</option>
            </select>
          </div>
        </div>
        <div className="qlist-items">
          {draftNew && (
            <div className="qitem is-selected">
              <div className="qitem-top">
                <span className="qitem-id">{draftNew.id}</span>
                <span className="vd-chip vd-chip-draft">새 문항 (저장 전)</span>
              </div>
            </div>
          )}
          {items.map((q) => (
            <div key={q.id} className={`qitem ${selected === q.id ? 'is-selected' : ''}`} onClick={() => select(q.id)}>
              <div className="qitem-top">
                <span className="qitem-id">{q.id}</span>
                <span className={`vd-chip vd-chip-${q.type}`}>{TYPE_LABEL[q.type]}</span>
                <span className={`vd-chip vd-chip-${q.status}`}>{STATUS_LABEL[q.status]}</span>
                {!!q.has_project_term && <span className="vd-chip vd-chip-project">P</span>}
              </div>
              <div className="qitem-stem">{q.stem}</div>
            </div>
          ))}
          {!items.length && !draftNew && <div className="empty">조건에 맞는 문항이 없습니다</div>}
        </div>
        <div className="qlist-foot">
          <span>{items.length}개</span>
          <button className="vd-btn vd-btn-primary vd-btn-small" onClick={startNew}>
            + 새 문항
          </button>
        </div>
      </section>

      <section className="editor">
        {draftNew ? (
          <QuestionEditor
            key={'new:' + draftNew.id}
            initial={draftNew}
            isNew
            onDirty={setDirty}
            onSaved={(q) => {
              setDraftNew(null);
              setSelected(q.id);
              setDirty(false);
            }}
            onCancel={() => (setDraftNew(null), setDirty(false))}
            onSelect={(id) => (setDraftNew(null), setSelected(id), setDirty(false))}
          />
        ) : selected && bank.getQuestion(selected) ? (
          <QuestionEditor
            key={selected + ':' + rev}
            initial={bank.getQuestion(selected)!}
            onDirty={setDirty}
            onSaved={() => setDirty(false)}
            onDeleted={() => (setSelected(null), setDirty(false))}
            onSelect={(id) => (setSelected(id), setDirty(false))}
          />
        ) : (
          <div className="empty">왼쪽에서 문항을 고르거나 새 문항을 만드세요</div>
        )}
      </section>
    </div>
  );
}
