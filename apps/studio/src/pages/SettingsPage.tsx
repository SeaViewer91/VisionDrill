import { isTauri, saveAs } from '@visiondrill/platform';
import { useBank, useStudio } from '../store';

export function SettingsPage() {
  const bank = useBank();
  const { reviewer, setReviewer, backupKeep, setBackupKeep, filePath, openBank, closeBank, createBank, flush, toast } = useStudio();
  return (
    <div className="page">
      <div className="page-head">
        <h1>설정</h1>
      </div>
      <div className="section">
        <h3>검수자</h3>
        <label className="field" style={{ maxWidth: 320 }}>
          <span>검수 완료 처리할 때 기록할 이름</span>
          <input className="vd-input" value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="예: 박수호" />
        </label>
      </div>
      <div className="section">
        <h3>문제은행 파일</h3>
        <p style={{ marginTop: 0 }}>
          <span className="vd-muted">현재 파일</span>
          <br />
          <code style={{ wordBreak: 'break-all' }}>{filePath ?? '브라우저 저장소 (미리보기 모드)'}</code>
        </p>
        <p className="vd-muted" style={{ fontSize: '0.88rem' }}>
          변경 사항은 저장할 때마다 자동으로 파일에 기록됩니다(임시 파일에 쓴 뒤 교체). 파일을 열 때와 앱을 닫을 때 같은 폴더의{' '}
          <code>backups</code> 폴더에 사본을 만듭니다. 이 폴더를 NAS·클라우드 동기화 폴더 안에 두면 더 안전합니다.
        </p>
        {isTauri() && (
          <label className="field" style={{ maxWidth: 240 }}>
            <span>보관할 백업 개수</span>
            <input
              className="vd-input"
              type="number"
              min={1}
              max={100}
              value={backupKeep}
              onChange={(e) => setBackupKeep(Math.max(1, Number(e.target.value)))}
            />
          </label>
        )}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button
            className="vd-btn"
            onClick={async () => {
              await flush();
              const p = await saveAs('visiondrill-bank-copy.db', bank.export(), [{ name: 'VisionDrill 문제은행', extensions: ['db'] }]);
              if (p) toast('사본을 저장했습니다');
            }}
          >
            다른 이름으로 사본 저장
          </button>
          <button className="vd-btn" onClick={() => openBank()}>
            다른 문제은행 열기
          </button>
          <button className="vd-btn" onClick={() => createBank(false)}>
            새 문제은행 만들기
          </button>
          <button className="vd-btn vd-btn-danger" onClick={() => closeBank()}>
            닫기
          </button>
        </div>
      </div>
      <div className="section vd-muted" style={{ fontSize: '0.85rem' }}>
        스키마 버전 {bank.meta('schema_version')} · 문제은행 이름 {bank.meta('bank_name')} · 만든 날 {bank.meta('created_at')?.slice(0, 10)}
      </div>
    </div>
  );
}
