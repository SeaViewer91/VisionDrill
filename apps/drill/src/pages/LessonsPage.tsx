import { lsGet, lsSet } from '@visiondrill/platform';
import { LessonView } from '@visiondrill/ui';
import { useEffect, useMemo, useState } from 'react';
import { LESSONS, PARTS, renderedLesson } from '../lessons';
import { useDrill } from '../store';

const LAST_KEY = 'vd.lesson.last';

export function LessonsPage() {
  const { learner, mutate, rev } = useDrill();
  const [id, setId] = useState<string | null>(() => {
    const last = lsGet(LAST_KEY);
    return LESSONS.find((l) => l.meta.id === last)?.meta.id ?? LESSONS[0]?.meta.id ?? null;
  });
  const read = useMemo(() => learner.lessonsRead(), [learner, rev]);
  const idx = LESSONS.findIndex((l) => l.meta.id === id);
  const lesson = idx >= 0 ? LESSONS[idx] : undefined;
  const rendered = useMemo(() => (lesson ? renderedLesson(lesson) : undefined), [lesson]);

  useEffect(() => {
    if (id) lsSet(LAST_KEY, id);
    document.querySelector('.main')?.scrollTo({ top: 0 });
  }, [id]);

  if (!LESSONS.length)
    return (
      <div className="page">
        <h1>자습서</h1>
        <p className="vd-muted">아직 들어 있는 자습서가 없습니다.</p>
      </div>
    );

  const readCount = LESSONS.filter((l) => read[l.meta.id]).length;
  const isRead = !!(lesson && read[lesson.meta.id]);

  return (
    <div className="lessons">
      <nav className="lessons-toc">
        <div className="lessons-toc-head">
          <strong>자습서</strong>
          <span className="vd-muted">
            {readCount}/{LESSONS.length} 읽음
          </span>
        </div>
        {PARTS.map((p) => {
          const items = LESSONS.filter((l) => l.meta.part === p.part);
          if (!items.length) return null;
          return (
            <div key={p.part} className="lessons-part">
              <div className="lessons-part-title">
                {p.part > 0 && p.part < 9 ? `${p.part}부 ` : ''}
                {p.title}
              </div>
              {items.map((l) => (
                <button
                  key={l.meta.id}
                  className={`lessons-item ${l.meta.id === id ? 'is-active' : ''}`}
                  onClick={() => setId(l.meta.id)}
                >
                  <span className="lessons-num">{/^\d+$/.test(l.meta.id) ? Number(l.meta.id) : l.meta.id}</span>
                  <span className="lessons-name">{l.meta.title}</span>
                  {read[l.meta.id] && <span className="lessons-check">✓</span>}
                </button>
              ))}
            </div>
          );
        })}
      </nav>
      <div className="lessons-body">
        {lesson && rendered && (
          <>
            {lesson.meta.status === 'draft' && (
              <div className="lessons-draft">검수 전 초안입니다. 틀린 내용이 있을 수 있습니다.</div>
            )}
            <LessonView lesson={rendered} />
            <div className="lessons-foot">
              <button className="vd-btn" disabled={idx <= 0} onClick={() => setId(LESSONS[idx - 1].meta.id)}>
                ← 이전 강
              </button>
              <button
                className={`vd-btn ${isRead ? '' : 'vd-btn-primary'}`}
                onClick={() => mutate((l) => l.setLessonRead(lesson.meta.id, !isRead))}
              >
                {isRead ? '읽음 표시 취소' : '다 읽었음'}
              </button>
              <button
                className="vd-btn"
                disabled={idx >= LESSONS.length - 1}
                onClick={() => setId(LESSONS[idx + 1].meta.id)}
              >
                다음 강 →
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
