/**
 * 데스크톱(Tauri)과 브라우저(개발·미리보기) 양쪽에서 동작하는 파일 입출력 계층.
 * Tauri: 네이티브 파일 대화상자 + 파일시스템(원자적 저장, 백업)
 * 브라우저: <input type=file> / 다운로드 + IndexedDB 자동 보관
 */

export const isTauri = (): boolean => typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

export interface FileFilter {
  name: string;
  extensions: string[];
}

export interface OpenedFile {
  path: string | null; // Tauri에서만 실제 경로
  name: string;
  bytes: Uint8Array;
}

function baseName(p: string): string {
  return p.split(/[\\/]/).pop() ?? p;
}

// ------------------------------------------------------------------ 열기
export async function pickAndReadFile(title: string, filters: FileFilter[]): Promise<OpenedFile | null> {
  if (isTauri()) {
    const { open } = await import('@tauri-apps/plugin-dialog');
    const { readFile } = await import('@tauri-apps/plugin-fs');
    const path = await open({ title, multiple: false, directory: false, filters });
    if (!path || Array.isArray(path)) return null;
    const bytes = await readFile(path as string);
    return { path: path as string, name: baseName(path as string), bytes };
  }
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = filters.flatMap((f) => f.extensions.map((e) => '.' + e)).join(',');
    input.onchange = async () => {
      const f = input.files?.[0];
      if (!f) return resolve(null);
      resolve({ path: null, name: f.name, bytes: new Uint8Array(await f.arrayBuffer()) });
    };
    input.oncancel = () => resolve(null);
    input.click();
  });
}

export async function readPath(path: string): Promise<Uint8Array> {
  const { readFile } = await import('@tauri-apps/plugin-fs');
  return readFile(path);
}

export async function pathExists(path: string): Promise<boolean> {
  if (!isTauri()) return false;
  const { exists } = await import('@tauri-apps/plugin-fs');
  return exists(path);
}

// ------------------------------------------------------------------ 저장
/** 저장 위치를 고르게 한 뒤 저장. 브라우저에서는 다운로드. 저장된 경로(또는 파일명)를 반환 */
export async function saveAs(defaultName: string, data: Uint8Array | string, filters: FileFilter[]): Promise<string | null> {
  const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data;
  if (isTauri()) {
    const { save } = await import('@tauri-apps/plugin-dialog');
    const path = await save({ defaultPath: defaultName, filters });
    if (!path) return null;
    await writeAtomic(path, bytes);
    return path;
  }
  const blob = new Blob([bytes as BlobPart], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = defaultName;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  return defaultName;
}

/** 새 파일 경로만 고르기 (Tauri 전용) */
export async function pickSavePath(defaultName: string, filters: FileFilter[]): Promise<string | null> {
  if (!isTauri()) return null;
  const { save } = await import('@tauri-apps/plugin-dialog');
  return (await save({ defaultPath: defaultName, filters })) ?? null;
}

/** 임시 파일에 먼저 쓰고 이름을 바꿔 저장 도중 문제가 생겨도 원본이 깨지지 않게 한다 */
export async function writeAtomic(path: string, bytes: Uint8Array): Promise<void> {
  const { writeFile, rename, remove, exists } = await import('@tauri-apps/plugin-fs');
  // 저장이 겹쳐도 서로의 임시 파일을 건드리지 않도록 고유한 이름 사용
  const tmp = `${path}.${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}.tmp`;
  await writeFile(tmp, bytes);
  try {
    await rename(tmp, path);
  } catch {
    // Windows에서 대상이 있으면 rename이 실패할 수 있음 → 지우고 다시 시도
    if (await exists(path)) await remove(path);
    await rename(tmp, path);
  }
}

/** 비동기 작업을 한 줄로 세워 겹치지 않게 실행 (저장 직렬화용) */
export function createQueue() {
  let tail: Promise<unknown> = Promise.resolve();
  return <T>(task: () => Promise<T>): Promise<T> => {
    const run = tail.then(task, task);
    tail = run.catch(() => undefined);
    return run;
  };
}

/**
 * 원본 옆 backups 폴더에 시각이 붙은 사본을 만들고 최근 keep개만 남긴다.
 * 예) bank.db → backups/bank-20260928-181500.db
 */
export async function backupFile(path: string, bytes: Uint8Array, keep = 10, backupDir?: string): Promise<string | null> {
  if (!isTauri()) return null;
  const { mkdir, readDir, remove, writeFile } = await import('@tauri-apps/plugin-fs');
  const { dirname, join } = await import('@tauri-apps/api/path');
  const dir = backupDir || (await join(await dirname(path), 'backups'));
  await mkdir(dir, { recursive: true });
  const name = baseName(path);
  const dot = name.lastIndexOf('.');
  const stem = dot > 0 ? name.slice(0, dot) : name;
  const ext = dot > 0 ? name.slice(dot) : '';
  const d = new Date();
  const two = (n: number) => String(n).padStart(2, '0');
  const ts = `${d.getFullYear()}${two(d.getMonth() + 1)}${two(d.getDate())}-${two(d.getHours())}${two(d.getMinutes())}${two(d.getSeconds())}`; // 현지 시각
  const target = await join(dir, `${stem}-${ts}${ext}`);
  await writeFile(target, bytes);
  const entries = (await readDir(dir))
    .filter((e) => e.isFile && e.name.startsWith(stem + '-') && e.name.endsWith(ext))
    .map((e) => e.name)
    .sort();
  for (const old of entries.slice(0, Math.max(0, entries.length - keep))) await remove(await join(dir, old));
  return target;
}

export async function appDataFile(name: string): Promise<string | null> {
  if (!isTauri()) return null;
  const { appDataDir, join } = await import('@tauri-apps/api/path');
  const { mkdir } = await import('@tauri-apps/plugin-fs');
  const dir = await appDataDir();
  await mkdir(dir, { recursive: true });
  return join(dir, name);
}

// ------------------------------------------------------------------ 브라우저 보관소 (개발·미리보기용)
const IDB_NAME = 'visiondrill';
function idb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore('files');
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function idbGet(key: string): Promise<Uint8Array | undefined> {
  try {
    const db = await idb();
    return await new Promise((resolve, reject) => {
      const r = db.transaction('files').objectStore('files').get(key);
      r.onsuccess = () => resolve(r.result as Uint8Array | undefined);
      r.onerror = () => reject(r.error);
    });
  } catch {
    return undefined;
  }
}

export async function idbSet(key: string, bytes: Uint8Array): Promise<void> {
  try {
    const db = await idb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction('files', 'readwrite');
      tx.objectStore('files').put(bytes, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    /* 사생활 보호 모드 등에서는 무시 */
  }
}

export function lsGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function lsSet(key: string, value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* 무시 */
  }
}

export async function openExternal(url: string): Promise<void> {
  if (isTauri()) {
    const { openUrl } = await import('@tauri-apps/plugin-opener');
    await openUrl(url);
    return;
  }
  window.open(url, '_blank', 'noopener');
}

/** 클립보드 복사 (웹뷰에서 Clipboard API가 막혀 있으면 execCommand로 대체) */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  }
}

// ------------------------------------------------------------------ 확인·알림 창
// 주의: window.confirm/alert를 직접 쓰지 말 것. Tauri 대화상자 플러그인이 이 둘을 비동기 함수로 바꿔치기하는데,
// 그 안에서 부르는 confirm 명령이 현재 버전에 없어 오류가 나고 확인 결과도 무시된다.

/** 예/아니요 확인 창. 확인을 누르면 true */
export async function askConfirm(message: string, okLabel = '확인', cancelLabel = '취소'): Promise<boolean> {
  if (isTauri()) {
    const { ask } = await import('@tauri-apps/plugin-dialog');
    return ask(message, { title: '확인', kind: 'warning', okLabel, cancelLabel });
  }
  return window.confirm(message);
}

/** 알림 창 */
export async function showMessage(message: string): Promise<void> {
  if (isTauri()) {
    const { message: msg } = await import('@tauri-apps/plugin-dialog');
    await msg(message, { title: '안내', kind: 'info' });
    return;
  }
  window.alert(message);
}
