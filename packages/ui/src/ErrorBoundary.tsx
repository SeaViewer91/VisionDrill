import { Component, type ReactNode } from 'react';

/** 화면 렌더링 중 오류가 나면 빈 화면 대신 오류 내용을 보여준다 (신고·디버깅용) */
export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="empty" style={{ textAlign: 'left', maxWidth: 720, margin: '10vh auto' }}>
        <h2>문제가 생겼습니다</h2>
        <p>아래 내용을 복사해 오류 신고에 붙여 주세요. 앱을 다시 시작하면 대부분 해결됩니다.</p>
        <pre className="vd-card" style={{ whiteSpace: 'pre-wrap', fontSize: '0.8rem' }}>
          {String(this.state.error?.stack ?? this.state.error)}
        </pre>
        <button className="vd-btn" onClick={() => location.reload()}>
          다시 시작
        </button>
      </div>
    );
  }
}

/** 렌더링 밖(비동기) 오류도 화면에 남긴다 */
export function installGlobalErrorOverlay() {
  const show = (msg: string) => {
    let el = document.getElementById('vd-fatal');
    if (!el) {
      el = document.createElement('pre');
      el.id = 'vd-fatal';
      el.style.cssText =
        'position:fixed;left:8px;right:8px;bottom:8px;max-height:40vh;overflow:auto;z-index:99;background:#fff4ee;color:#9a3412;border:1px solid #fdba74;border-radius:8px;padding:10px;font-size:12px;white-space:pre-wrap';
      document.body.appendChild(el);
    }
    el.textContent += msg + '\n';
  };
  window.addEventListener('error', (e) => show(`[error] ${e.message} ${e.filename ?? ''}:${e.lineno ?? ''}`));
  window.addEventListener('unhandledrejection', (e) => show(`[promise] ${e.reason?.stack ?? e.reason}`));
}
