import { isTauri } from '@visiondrill/platform';
import { useStudio } from '../store';

export function WelcomePage() {
  const { createBank, openBank } = useStudio();
  return (
    <div className="welcome">
      <h1>VisionDrill Studio</h1>
      <p className="vd-muted">문제은행 원본(SQLite 파일 하나)을 열어 문항·용어를 관리하고, 학습 앱용 문제 팩(.vdpack)을 만듭니다.</p>
      <div className="actions">
        <button className="vd-btn vd-btn-primary vd-btn-lg" onClick={() => openBank()}>
          기존 문제은행 열기
        </button>
        <button className="vd-btn vd-btn-lg" onClick={() => createBank(false)}>
          새 문제은행 만들기
        </button>
        <button className="vd-btn vd-btn-lg" onClick={() => createBank(true)}>
          샘플 문항으로 시작하기
        </button>
      </div>
      {!isTauri() && (
        <p className="vd-muted" style={{ marginTop: 24, fontSize: '0.85rem' }}>
          브라우저 미리보기 모드입니다. 작업 내용은 이 브라우저에 자동 보관되며, 파일은 '설정 → 다른 이름으로 저장'으로 내려받을 수
          있습니다.
        </p>
      )}
    </div>
  );
}
