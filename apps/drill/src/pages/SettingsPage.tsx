import { saveAs } from '@visiondrill/platform';
import { useDrill } from '../store';

export function SettingsPage() {
  const { learner, mutate, importPackFile, toast, save } = useDrill();
  const s = learner.settings();
  const pack = learner.packInfo();
  const set = (patch: Parameters<typeof learner.saveSettings>[0]) => mutate((l) => l.saveSettings(patch));

  return (
    <div className="page">
      <div className="page-head">
        <h1>설정</h1>
      </div>

      <div className="section-card">
        <h2>학습</h2>
        <div className="field-row">
          <label className="field">
            <span>한 세션 문항 수</span>
            <input
              className="vd-input"
              type="number"
              min={5}
              max={100}
              value={s.sessionSize}
              onChange={(e) => set({ sessionSize: Math.max(1, Number(e.target.value)) })}
            />
          </label>
          <label className="field">
            <span>세션당 새 문항 최대</span>
            <input
              className="vd-input"
              type="number"
              min={0}
              max={100}
              value={s.newPerSession}
              onChange={(e) => set({ newPerSession: Math.max(0, Number(e.target.value)) })}
            />
          </label>
          <label className="field">
            <span>오답노트 졸업 연속 정답</span>
            <input
              className="vd-input"
              type="number"
              min={1}
              max={10}
              value={s.graduateStreak}
              onChange={(e) => set({ graduateStreak: Math.max(1, Number(e.target.value)) })}
            />
          </label>
        </div>
        <label className="check">
          <input type="checkbox" checked={s.shuffleChoices} onChange={(e) => set({ shuffleChoices: e.target.checked })} />
          보기 순서 섞기 (위치로 외우는 것 방지)
        </label>
        <label className="check">
          <input type="checkbox" checked={s.guessedToWrongNote} onChange={(e) => set({ guessedToWrongNote: e.target.checked })} />
          '찍었음'으로 맞힌 문항도 오답노트에 넣기
        </label>
        <label className="check">
          <input type="checkbox" checked={s.unlockAll} onChange={(e) => set({ unlockAll: e.target.checked })} />
          모든 트랙 열기 (사수·관리자용, 해금 조건 무시)
        </label>
      </div>

      <div className="section-card">
        <h2>문제 팩</h2>
        <p style={{ marginTop: 0 }}>
          {pack ? (
            <>
              현재 <strong>{pack.pack_version}</strong> · 문항 {pack.question_count}개 · 용어 {pack.term_count}개
              <span className="vd-muted"> ({pack.exported_at?.slice(0, 10)} 생성)</span>
            </>
          ) : (
            '설치된 문제 팩이 없습니다'
          )}
        </p>
        <p className="vd-muted" style={{ fontSize: '0.88rem' }}>
          새 팩을 가져와도 학습 기록은 그대로 유지됩니다. 정답이 바뀐 문항만 복습 일정이 처음부터 다시 시작됩니다.
        </p>
        <button className="vd-btn vd-btn-primary" onClick={importPackFile}>
          문제 팩(.vdpack) 가져오기
        </button>
      </div>

      <div className="section-card">
        <h2>학습 기록</h2>
        <p className="vd-muted" style={{ marginTop: 0, fontSize: '0.88rem' }}>
          학습 기록은 이 컴퓨터에만 저장됩니다. PC를 옮기거나 사수에게 진도를 보여줄 때 파일로 내보내세요.
        </p>
        <button
          className="vd-btn"
          onClick={async () => {
            await save();
            const d = new Date().toISOString().slice(0, 10).replace(/-/g, '');
            const p = await saveAs(`visiondrill-learner-${d}.db`, learner.export(), [{ name: '학습 기록', extensions: ['db'] }]);
            if (p) toast('학습 기록을 내보냈습니다');
          }}
        >
          학습 기록 내보내기
        </button>
      </div>
    </div>
  );
}
