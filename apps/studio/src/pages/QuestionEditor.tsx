import {
  LEVEL_LABEL,
  STATUS_LABEL,
  TYPE_LABEL,
  gradeQuestion,
  validateQuestion,
  type Answer,
  type QType,
  type Question,
  type Response,
} from '@visiondrill/core';
import { askConfirm, pickAndReadFile, showMessage } from '@visiondrill/platform';
import { CIRCLED, QuestionCard, ResultCard } from '@visiondrill/ui';
import { useEffect, useMemo, useState } from 'react';
import { useBank, useStudio } from '../store';

interface Props {
  initial: Question;
  isNew?: boolean;
  onDirty: (d: boolean) => void;
  onSaved: (q: Question) => void;
  onCancel?: () => void;
  onDeleted?: () => void;
  onSelect: (id: string) => void;
}

type Tab = 'edit' | 'preview' | 'history';

const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

function IssueList({ issues }: { issues: ReturnType<typeof validateQuestion> }) {
  const [open, setOpen] = useState(false);
  if (!issues.length) return <div className="reviewed-note">✓ 검증 통과 — 검수 완료로 전환할 수 있습니다</div>;
  const e = issues.filter((i) => i.level === 'error').length;
  const w = issues.length - e;
  const show = open || issues.length <= 3;
  return (
    <div style={{ marginBottom: 12 }}>
      {!show && (
        <button type="button" className="vd-link" onClick={() => setOpen(true)} style={{ fontSize: '0.88rem' }}>
          {e > 0 && <span style={{ color: 'var(--bad)' }}>⛔ 오류 {e}개 </span>}
          {w > 0 && <span style={{ color: 'var(--warn)' }}>⚠️ 경고 {w}개 </span>}
          (자세히)
        </button>
      )}
      {show && (
        <ul className="issues">
          {issues.map((i, k) => (
            <li key={k} className={`is-${i.level}`}>
              {i.level === 'error' ? '⛔' : '⚠️'} {i.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function convertType(q: Question, type: QType): Question {
  const toShort = type === 'short';
  const fromShort = q.type === 'short';
  if (toShort === fromShort) return { ...q, type };
  if (toShort) {
    return { ...q, type, answer_choice: null, choices: [], answers: [{ lang: 'en', text: '', is_primary: true }] };
  }
  return {
    ...q,
    type,
    answer_choice: 1,
    answers: [],
    choices: [1, 2, 3, 4, 5].map((idx) => ({ idx, text: '', rationale: '' })),
  };
}

export function QuestionEditor({ initial, isNew, onDirty, onSaved, onCancel, onDeleted, onSelect }: Props) {
  const bank = useBank();
  const { mutate, mutateAsync, reviewer, rev } = useStudio();
  const [q, setQ] = useState<Question>(() => structuredClone(initial));
  const [tab, setTab] = useState<Tab>('edit');
  const [termQuery, setTermQuery] = useState('');
  const chapters = useMemo(() => bank.chapters(), [bank, rev]);
  const terms = useMemo(() => bank.terms(), [bank, rev]);
  const dirty = useMemo(() => isNew || JSON.stringify(q) !== JSON.stringify(initial), [q, initial, isNew]);
  const issues = useMemo(() => validateQuestion(q), [q]);
  const errors = issues.filter((i) => i.level === 'error');
  const locked = initial.status === 'retired';

  useEffect(() => onDirty(dirty), [dirty, onDirty]);

  const upd = (patch: Partial<Question>) => setQ((prev) => ({ ...prev, ...patch }));

  const save = () => {
    const saved = mutate((b) => b.saveQuestion(q), isNew ? '문항을 만들었습니다' : '저장했습니다');
    if (saved) onSaved(saved);
  };

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        if (!locked) save();
      }
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  });

  const setStatus = async (status: 'draft' | 'reviewed' | 'retired') => {
    if (dirty) return showMessage('먼저 변경 사항을 저장하세요.');
    if (status === 'retired' && !(await askConfirm('출제 중단으로 바꾸면 되돌릴 수 없습니다(복제는 가능). 계속할까요?', '출제 중단')))
      return;
    const msg = { draft: '초안으로 되돌렸습니다', reviewed: '검수 완료로 전환했습니다', retired: '출제 중단했습니다' }[status];
    mutate((b) => b.setStatus(q.id, status, reviewer), msg);
  };

  const attachImage = async () => {
    const f = await pickAndReadFile('문항 이미지', [{ name: '이미지', extensions: ['png', 'jpg', 'jpeg', 'webp', 'gif', 'svg'] }]);
    if (!f) return;
    const ext = f.name.split('.').pop()?.toLowerCase() ?? 'png';
    const mime = ext === 'svg' ? 'image/svg+xml' : ext === 'jpg' ? 'image/jpeg' : `image/${ext}`;
    const id = await mutateAsync((b) => b.addAsset(f.name, mime, f.bytes));
    if (id) upd({ image_asset_id: id });
  };

  const imageUrl = q.image_asset_id ? bank.assetDataUrl(q.image_asset_id) : undefined;
  const linkedTerms = q.term_ids.map((id) => terms.find((t) => t.id === id)).filter(Boolean) as typeof terms;
  const termMatches = termQuery.trim()
    ? terms
        .filter((t) => !q.term_ids.includes(t.id))
        .filter((t) => [t.id, t.term_en, t.term_ko, t.abbreviation].some((s) => s?.toLowerCase().includes(termQuery.trim().toLowerCase())))
        .slice(0, 8)
    : [];

  const setAnswer = (i: number, patch: Partial<Answer>) =>
    upd({
      answers: q.answers.map((a, j) => (j === i ? { ...a, ...patch } : patch.is_primary ? { ...a, is_primary: false } : a)),
    });

  return (
    <div>
      <div className="editor-head">
        <h2>{q.id}</h2>
        <span className={`vd-chip vd-chip-${initial.status}`}>{isNew ? '새 문항' : STATUS_LABEL[initial.status]}</span>
        {!isNew && <span className="vd-chip vd-chip-muted">v{initial.version}</span>}
        {dirty && !isNew && <span className="vd-chip vd-chip-scenario">저장 안 됨</span>}
        <div className="editor-actions">
          {!locked && (
            <button className="vd-btn vd-btn-primary vd-btn-small" onClick={save} disabled={!dirty}>
              저장 <kbd style={{ opacity: 0.7, fontSize: '0.75rem' }}>{IS_MAC ? '⌘S' : 'Ctrl+S'}</kbd>
            </button>
          )}
          {isNew && onCancel && (
            <button className="vd-btn vd-btn-small" onClick={onCancel}>
              취소
            </button>
          )}
          {!isNew && dirty && (
            <button className="vd-btn vd-btn-small" onClick={() => setQ(structuredClone(initial))}>
              되돌리기
            </button>
          )}
          {!isNew && initial.status === 'draft' && (
            <button
              className="vd-btn vd-btn-small"
              onClick={() => setStatus('reviewed')}
              disabled={errors.length > 0}
              title={errors.length ? '검증 오류를 먼저 해결하세요' : !reviewer ? '설정에서 검수자 이름을 입력하세요' : ''}
            >
              ✓ 검수 완료
            </button>
          )}
          {!isNew && initial.status === 'reviewed' && (
            <button className="vd-btn vd-btn-small" onClick={() => setStatus('draft')}>
              초안으로
            </button>
          )}
          {!isNew && (
            <button
              className="vd-btn vd-btn-small"
              onClick={() => {
                const c = mutate((b) => b.duplicateQuestion(q.id), '복제했습니다');
                if (c) onSelect(c.id);
              }}
            >
              복제
            </button>
          )}
          {!isNew && initial.status === 'draft' && (
            <button
              className="vd-btn vd-btn-small vd-btn-danger"
              onClick={async () => {
                if (!(await askConfirm(`${q.id} 초안을 삭제할까요? (변경 이력에는 남습니다)`, '삭제'))) return;
                mutate((b) => b.deleteQuestion(q.id), '삭제했습니다');
                onDeleted?.();
              }}
            >
              삭제
            </button>
          )}
          {!isNew && initial.status === 'reviewed' && (
            <button className="vd-btn vd-btn-small vd-btn-danger" onClick={() => setStatus('retired')}>
              출제 중단
            </button>
          )}
        </div>
      </div>

      {initial.status === 'reviewed' && (
        <div className="reviewed-note">
          검수 완료 ({initial.reviewed_by}, {initial.reviewed_at?.slice(0, 10)}). 수정하면 버전이 올라가고, 정답이 바뀌면 학습자의 복습
          일정이 초기화됩니다.
        </div>
      )}
      {locked && <div className="retired-note">출제 중단된 문항은 수정할 수 없습니다. 다시 쓰려면 복제하세요.</div>}

      <div className="tabs">
        {(['edit', 'preview', 'history'] as Tab[]).map((t) => (
          <button key={t} className={`tab ${tab === t ? 'is-active' : ''}`} onClick={() => setTab(t)}>
            {{ edit: '편집', preview: '미리보기', history: '변경 이력' }[t]}
          </button>
        ))}
      </div>

      {tab === 'edit' && (
        <fieldset disabled={locked} style={{ border: 0, padding: 0, margin: 0 }}>
          <IssueList issues={issues} />
          <div className="section">
            <div className="field-row">
              {isNew && (
                <label className="field">
                  <span>문항 ID</span>
                  <input className="vd-input" value={q.id} onChange={(e) => upd({ id: e.target.value.trim() })} />
                </label>
              )}
              <label className="field">
                <span>유형</span>
                <select className="vd-select" value={q.type} onChange={(e) => setQ(convertType(q, e.target.value as QType))}>
                  {Object.entries(TYPE_LABEL).map(([k, v]) => (
                    <option key={k} value={k}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>장</span>
                <select
                  className="vd-select"
                  value={q.chapter_id}
                  onChange={(e) => upd({ chapter_id: e.target.value, ...(isNew ? { id: bank.nextQuestionId(e.target.value) } : {}) })}
                >
                  {chapters.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.id} {c.name_ko}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>인지 단계</span>
                <select className="vd-select" value={q.cognitive_level} onChange={(e) => upd({ cognitive_level: Number(e.target.value) })}>
                  {[1, 2, 3].map((l) => (
                    <option key={l} value={l}>
                      {l}. {LEVEL_LABEL[l]}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field" style={{ maxWidth: 110 }}>
                <span>난이도</span>
                <select className="vd-select" value={q.difficulty} onChange={(e) => upd({ difficulty: Number(e.target.value) })}>
                  {[1, 2, 3, 4, 5].map((d) => (
                    <option key={d} value={d}>
                      {'★'.repeat(d)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <label className="field">
              <span>{q.type === 'scenario' ? '문제 본문 (지시문 + 질문)' : '문제 본문'}</span>
              <textarea
                className="vd-textarea"
                rows={q.type === 'scenario' ? 5 : 3}
                value={q.stem}
                placeholder={q.type === 'scenario' ? '지시: "…"\n\n가장 적절한 조치는?' : ''}
                onChange={(e) => upd({ stem: e.target.value })}
              />
            </label>
            <div className="field-row" style={{ alignItems: 'center' }}>
              <button type="button" className="vd-btn vd-btn-small" onClick={attachImage}>
                {q.image_asset_id ? '이미지 바꾸기' : '이미지 첨부'}
              </button>
              {q.image_asset_id && (
                <button type="button" className="vd-btn vd-btn-small" onClick={() => upd({ image_asset_id: null })}>
                  이미지 빼기
                </button>
              )}
              {imageUrl && <img src={imageUrl} alt="" style={{ maxHeight: 60, borderRadius: 6 }} />}
            </div>
          </div>

          {q.type === 'short' ? (
            <div className="section">
              <h3>허용 답 (영문 또는 한글 중 하나만 맞으면 정답)</h3>
              {q.answers.map((a, i) => (
                <div className="answer-row" key={i}>
                  <select className="vd-select" value={a.lang} onChange={(e) => setAnswer(i, { lang: e.target.value as Answer['lang'] })}>
                    <option value="en">영문</option>
                    <option value="ko">한글</option>
                  </select>
                  <input className="vd-input" value={a.text} onChange={(e) => setAnswer(i, { text: e.target.value })} />
                  <label title="결과 화면에 정답으로 보여줄 대표 답" style={{ fontSize: '0.82rem', display: 'flex', gap: 4 }}>
                    <input type="radio" checked={a.is_primary} onChange={() => setAnswer(i, { is_primary: true })} />
                    대표
                  </label>
                  <button
                    type="button"
                    className="vd-btn vd-btn-small"
                    onClick={() => upd({ answers: q.answers.filter((_, j) => j !== i) })}
                    aria-label="삭제"
                  >
                    ✕
                  </button>
                </div>
              ))}
              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  type="button"
                  className="vd-btn vd-btn-small"
                  onClick={() => upd({ answers: [...q.answers, { lang: 'en', text: '', is_primary: !q.answers.length }] })}
                >
                  + 영문
                </button>
                <button
                  type="button"
                  className="vd-btn vd-btn-small"
                  onClick={() => upd({ answers: [...q.answers, { lang: 'ko', text: '', is_primary: !q.answers.length }] })}
                >
                  + 한글
                </button>
              </div>
              <p className="vd-muted" style={{ fontSize: '0.82rem', margin: '8px 0 0' }}>
                대소문자·공백·하이픈·마침표·따옴표는 무시하고, 괄호 안/밖도 각각 인정합니다. 예) "Intersection over Union (IoU)" → "IoU"도
                정답
              </p>
            </div>
          ) : (
            <div className="section">
              <h3>보기 5개 · 정답 선택 · 오답마다 틀린 이유</h3>
              {q.choices.map((c) => {
                const isAns = c.idx === q.answer_choice;
                return (
                  <div key={c.idx} className={`choice-row ${isAns ? 'is-answer' : ''}`}>
                    <label className="correct-pick" title="정답으로 지정">
                      <span style={{ fontWeight: 600 }}>{CIRCLED[c.idx - 1]}</span>
                      <input type="radio" name={`ans-${q.id}`} checked={isAns} onChange={() => upd({ answer_choice: c.idx })} />
                    </label>
                    <div className="inputs">
                      <input
                        className="vd-input"
                        value={c.text}
                        placeholder={`보기 ${c.idx}`}
                        onChange={(e) => upd({ choices: q.choices.map((x) => (x.idx === c.idx ? { ...x, text: e.target.value } : x)) })}
                      />
                      {isAns ? (
                        <span className="vd-muted" style={{ fontSize: '0.82rem', color: 'var(--ok)' }}>
                          ✓ 정답 보기
                        </span>
                      ) : (
                        <input
                          className="vd-input"
                          value={c.rationale ?? ''}
                          placeholder="이 보기가 틀린 이유 (필수)"
                          style={{ fontSize: '0.88rem' }}
                          onChange={(e) =>
                            upd({ choices: q.choices.map((x) => (x.idx === c.idx ? { ...x, rationale: e.target.value } : x)) })
                          }
                        />
                      )}
                    </div>
                  </div>
                );
              })}
              <p className="vd-muted" style={{ fontSize: '0.82rem', margin: '8px 0 0' }}>
                학습 앱에서는 보기 순서가 섞입니다. "①번과 ②번 모두" 같은 위치 의존 보기는 피하세요.
              </p>
            </div>
          )}

          <div className="section">
            <label className="field">
              <span>짧은 해설 (오답 시 자동 표시, 2~3문장)</span>
              <textarea
                className="vd-textarea"
                rows={3}
                value={q.explanation_short}
                onChange={(e) => upd({ explanation_short: e.target.value })}
              />
            </label>
            <label className="field">
              <span>상세 해설 (선택, "더 보기"로 펼침)</span>
              <textarea
                className="vd-textarea"
                rows={4}
                value={q.explanation_full ?? ''}
                onChange={(e) => upd({ explanation_full: e.target.value || null })}
              />
            </label>
          </div>

          <div className="section">
            <h3>용어 연결</h3>
            <div className="term-picker">
              {linkedTerms.map((t) => (
                <span key={t.id} className={`vd-chip ${t.origin === 'project' ? 'vd-chip-project' : ''}`}>
                  {t.abbreviation ?? t.term_en}
                  <button type="button" onClick={() => upd({ term_ids: q.term_ids.filter((x) => x !== t.id) })} aria-label="연결 해제">
                    ×
                  </button>
                </span>
              ))}
              {q.term_ids
                .filter((id) => !terms.some((t) => t.id === id))
                .map((id) => (
                  <span key={id} className="vd-chip vd-chip-retired" title="존재하지 않는 용어">
                    {id}?
                  </span>
                ))}
              <input
                className="vd-input"
                style={{ width: 220 }}
                placeholder="용어 검색해서 추가"
                value={termQuery}
                onChange={(e) => setTermQuery(e.target.value)}
              />
            </div>
            {termMatches.length > 0 && (
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                {termMatches.map((t) => (
                  <button
                    key={t.id}
                    type="button"
                    className="vd-btn vd-btn-small"
                    onClick={() => (upd({ term_ids: [...q.term_ids, t.id] }), setTermQuery(''))}
                  >
                    + {t.abbreviation ? `${t.abbreviation} · ` : ''}
                    {t.term_en}
                  </button>
                ))}
              </div>
            )}
            <div className="field-row" style={{ marginTop: 12 }}>
              <label className="field">
                <span>태그 (쉼표로 구분)</span>
                <input
                  className="vd-input"
                  value={q.tags.join(', ')}
                  onChange={(e) =>
                    upd({
                      tags: e.target.value
                        .split(',')
                        .map((s) => s.trim())
                        .filter(Boolean),
                    })
                  }
                />
              </label>
              <label className="field">
                <span>출처</span>
                <input className="vd-input" value={q.source_ref ?? ''} onChange={(e) => upd({ source_ref: e.target.value || null })} />
              </label>
            </div>
          </div>
        </fieldset>
      )}

      {tab === 'preview' && <Preview q={q} terms={linkedTerms} imageUrl={imageUrl} />}
      {tab === 'history' && !isNew && <History id={q.id} />}
      {tab === 'history' && isNew && <div className="empty">저장한 뒤부터 이력이 쌓입니다</div>}
    </div>
  );
}

function Preview({ q, terms, imageUrl }: { q: Question; terms: any[]; imageUrl?: string }) {
  const [resp, setResp] = useState<Response | undefined>();
  const [guessed, setGuessed] = useState(false);
  const [graded, setGraded] = useState(false);
  const grade = gradeQuestion(q, resp);
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <p className="vd-muted" style={{ margin: 0, fontSize: '0.85rem' }}>
        학습 앱과 같은 화면입니다. 답을 고르고 채점해 보세요. (미리보기에서는 보기 순서를 섞지 않습니다)
      </p>
      {!graded ? (
        <>
          <QuestionCard
            question={q}
            terms={terms}
            response={resp}
            onResponse={setResp}
            guessed={guessed}
            onGuessed={setGuessed}
            imageUrl={imageUrl}
            onEnter={() => setGraded(true)}
          />
          <div>
            <button className="vd-btn vd-btn-primary" onClick={() => setGraded(true)}>
              채점해 보기
            </button>
          </div>
        </>
      ) : (
        <>
          <ResultCard question={q} terms={terms} response={resp} grade={grade} guessed={guessed} imageUrl={imageUrl} />
          <div>
            <button className="vd-btn" onClick={() => (setGraded(false), setResp(undefined), setGuessed(false))}>
              다시 풀기
            </button>
          </div>
        </>
      )}
    </div>
  );
}

const ACTION_LABEL: Record<string, string> = { create: '생성', update: '수정', status: '상태 변경', retire: '출제 중단', delete: '삭제' };

function History({ id }: { id: string }) {
  const bank = useBank();
  const { mutate, rev } = useStudio();
  const items = useMemo(() => bank.history('question', id), [bank, id, rev]);
  if (!items.length) return <div className="empty">이력이 없습니다</div>;
  return (
    <div className="section">
      <p className="vd-muted" style={{ fontSize: '0.85rem', marginTop: 0 }}>
        '생성'은 처음 저장된 내용, 나머지는 그 작업 <strong>직전</strong>의 내용입니다. 되돌려도 현재 내용은 이력으로 남습니다.
      </p>
      {items.map((h) => {
        const snap = JSON.parse(h.snapshot_json) as Question;
        return (
          <div className="history-item" key={h.id}>
            <div className="history-meta">
              <span>{new Date(h.changed_at).toLocaleString('ko-KR')}</span>
              <span className="vd-chip">{ACTION_LABEL[h.action] ?? h.action}</span>
              {h.version && <span className="vd-chip vd-chip-muted">v{h.version}</span>}
              {h.note && <span>{h.note}</span>}
              <span style={{ marginLeft: 'auto' }} />
              {
                <button
                  className="vd-btn vd-btn-small"
                  onClick={async () =>
                    (await askConfirm('이 시점의 내용으로 되돌릴까요? (초안 상태가 됩니다)', '되돌리기')) &&
                    mutate((b) => b.restoreQuestion(h.id), '되돌렸습니다')
                  }
                >
                  이 내용으로 되돌리기
                </button>
              }
            </div>
            <div style={{ fontSize: '0.9rem' }}>{snap.stem}</div>
            <div className="vd-muted" style={{ fontSize: '0.82rem' }}>
              {snap.type === 'short'
                ? `허용 답: ${snap.answers.map((a) => a.text).join(', ')}`
                : `정답 ${CIRCLED[(snap.answer_choice ?? 1) - 1]} ${snap.choices.find((c) => c.idx === snap.answer_choice)?.text ?? ''}`}
            </div>
          </div>
        );
      })}
    </div>
  );
}
