import { Bank, BankError, importExchange, loadSql, parseExchange } from '@visiondrill/core';
import {
  backupFile,
  createQueue,
  idbGet,
  idbSet,
  isTauri,
  lsGet,
  lsSet,
  pathExists,
  pickAndReadFile,
  pickSavePath,
  readPath,
  writeAtomic,
} from '@visiondrill/platform';
import type { SqlJsStatic } from 'sql.js';
import wasmUrl from 'sql.js/dist/sql-wasm.wasm?url';
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';

export const BANK_FILTERS = [{ name: 'VisionDrill 문제은행', extensions: ['db', 'sqlite'] }];
const LS_PATH = 'studio.lastPath';
const LS_REVIEWER = 'studio.reviewer';
const LS_BACKUP_KEEP = 'studio.backupKeep';
const IDB_KEY = 'studio-bank';

export type SaveState = 'saved' | 'dirty' | 'saving' | 'error';

interface Toast {
  msg: string;
  error?: boolean;
}

interface Ctx {
  SQL: SqlJsStatic;
  bank: Bank | null;
  filePath: string | null;
  rev: number;
  saveState: SaveState;
  lastSaved: Date | null;
  reviewer: string;
  setReviewer: (v: string) => void;
  backupKeep: number;
  setBackupKeep: (n: number) => void;
  /** 쓰기 작업을 실행하고 자동 저장. 실패하면 오류 토스트를 띄우고 false */
  mutate: <T>(fn: (b: Bank) => T, okMsg?: string) => T | undefined;
  mutateAsync: <T>(fn: (b: Bank) => Promise<T>, okMsg?: string) => Promise<T | undefined>;
  toast: (msg: string, error?: boolean) => void;
  createBank: (withSample: boolean) => Promise<void>;
  openBank: () => Promise<void>;
  closeBank: () => Promise<void>;
  flush: () => Promise<void>;
}

const StudioContext = createContext<Ctx | null>(null);

export function useStudio(): Ctx {
  const c = useContext(StudioContext);
  if (!c) throw new Error('StudioProvider 밖에서 사용됨');
  return c;
}

export function useBank(): Bank {
  const { bank } = useStudio();
  if (!bank) throw new Error('열린 문제은행이 없습니다');
  return bank;
}

export function StudioProvider({ children }: { children: ReactNode }) {
  const [SQL, setSQL] = useState<SqlJsStatic | null>(null);
  const [bank, setBank] = useState<Bank | null>(null);
  const [filePath, setFilePath] = useState<string | null>(null);
  const [rev, setRev] = useState(0);
  const [saveState, setSaveState] = useState<SaveState>('saved');
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [reviewer, setReviewerState] = useState(lsGet(LS_REVIEWER) ?? '');
  const [backupKeep, setBackupKeepState] = useState(Number(lsGet(LS_BACKUP_KEEP) ?? 10));
  const [toastState, setToast] = useState<Toast | null>(null);
  const [booting, setBooting] = useState(true);
  const saveTimer = useRef<number | undefined>(undefined);
  const bankRef = useRef<Bank | null>(null);
  const queue = useRef(createQueue());
  const started = useRef(false);
  const pathRef = useRef<string | null>(null);
  bankRef.current = bank;
  pathRef.current = filePath;

  const toast = useCallback((msg: string, error = false) => {
    setToast({ msg, error });
    window.setTimeout(() => setToast((t) => (t?.msg === msg ? null : t)), error ? 6000 : 2500);
  }, []);

  const persist = useCallback(async () => {
    window.clearTimeout(saveTimer.current);
    await queue.current(async () => {
      const b = bankRef.current;
      if (!b) return;
      setSaveState('saving');
      try {
        const bytes = b.export();
        if (isTauri() && pathRef.current) await writeAtomic(pathRef.current, bytes);
        else await idbSet(IDB_KEY, bytes);
        setSaveState('saved');
        setLastSaved(new Date());
      } catch (e) {
        setSaveState('error');
        toast(`저장 실패: ${e instanceof Error ? e.message : String(e)}`, true);
      }
    });
  }, [toast]);

  const scheduleSave = useCallback(() => {
    setSaveState('dirty');
    window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(persist, 600);
  }, [persist]);

  const handleError = useCallback(
    (e: unknown) => {
      const msg = e instanceof BankError ? e.message : `오류: ${(e as Error)?.message ?? e}`;
      toast(msg, true);
      if (!(e instanceof BankError)) console.error(e);
    },
    [toast],
  );

  const mutate = useCallback(
    <T,>(fn: (b: Bank) => T, okMsg?: string): T | undefined => {
      const b = bankRef.current;
      if (!b) return undefined;
      try {
        const r = fn(b);
        setRev((x) => x + 1);
        scheduleSave();
        if (okMsg) toast(okMsg);
        return r;
      } catch (e) {
        handleError(e);
        return undefined;
      }
    },
    [scheduleSave, toast, handleError],
  );

  const mutateAsync = useCallback(
    async <T,>(fn: (b: Bank) => Promise<T>, okMsg?: string): Promise<T | undefined> => {
      const b = bankRef.current;
      if (!b) return undefined;
      try {
        const r = await fn(b);
        setRev((x) => x + 1);
        scheduleSave();
        if (okMsg) toast(okMsg);
        return r;
      } catch (e) {
        handleError(e);
        return undefined;
      }
    },
    [scheduleSave, toast, handleError],
  );

  const adopt = useCallback(async (b: Bank, path: string | null, backup: boolean) => {
    setBank(b);
    setFilePath(path);
    bankRef.current = b;
    pathRef.current = path;
    setRev((x) => x + 1);
    setSaveState('saved');
    setLastSaved(null);
    lsSet(LS_PATH, path);
    if (backup && path) {
      try {
        await backupFile(path, b.export(), Number(lsGet(LS_BACKUP_KEEP) ?? 10));
      } catch (e) {
        console.warn('backup failed', e);
      }
    }
  }, []);

  // 시작 시 마지막 문제은행 자동 열기
  useEffect(() => {
    if (started.current) return; // 개발 모드(StrictMode) 이중 실행 방지
    started.current = true;
    (async () => {
      const sql = await loadSql(wasmUrl);
      setSQL(sql);
      try {
        if (isTauri()) {
          const last = lsGet(LS_PATH);
          if (last && (await pathExists(last))) await adopt(Bank.open(sql, await readPath(last)), last, true);
        } else {
          const bytes = await idbGet(IDB_KEY);
          if (bytes) await adopt(Bank.open(sql, bytes), null, false);
        }
      } catch (e) {
        console.warn('자동 열기 실패', e);
      } finally {
        setBooting(false);
      }
    })();
  }, [adopt]);

  // 창 닫기 전 저장 + 백업
  useEffect(() => {
    if (!isTauri()) {
      const h = () => {
        if (bankRef.current) void idbSet(IDB_KEY, bankRef.current.export());
      };
      window.addEventListener('beforeunload', h);
      return () => window.removeEventListener('beforeunload', h);
    }
    let unlisten: (() => void) | undefined;
    (async () => {
      const { getCurrentWindow } = await import('@tauri-apps/api/window');
      const win = getCurrentWindow();
      unlisten = await win.onCloseRequested(async () => {
        const b = bankRef.current;
        const p = pathRef.current;
        if (b && p) {
          const bytes = b.export();
          await writeAtomic(p, bytes);
          await backupFile(p, bytes, Number(lsGet(LS_BACKUP_KEEP) ?? 10)).catch(() => undefined);
        }
      });
    })();
    return () => unlisten?.();
  }, []);

  const createBank = useCallback(
    async (withSample: boolean) => {
      if (!SQL) return;
      let path: string | null = null;
      if (isTauri()) {
        path = await pickSavePath('visiondrill-bank.db', BANK_FILTERS);
        if (!path) return;
      }
      try {
        const b = Bank.create(SQL);
        let sampleCount = 0;
        if (withSample) {
          const text = await (await fetch('./sample-bank.json')).text();
          const { file, errors } = parseExchange(text);
          if (!file) throw new Error(errors.join('\n'));
          sampleCount = importExchange(b, file, { reviewer: 'sample' }).created;
        }
        const bytes = b.export();
        if (path) await writeAtomic(path, bytes);
        else await idbSet(IDB_KEY, bytes);
        await adopt(b, path, false);
        toast(withSample ? `샘플 문항 ${sampleCount}개가 담긴 문제은행을 만들었습니다` : '새 문제은행을 만들었습니다');
      } catch (e) {
        handleError(e);
      }
    },
    [SQL, adopt, toast, handleError],
  );

  const openBank = useCallback(async () => {
    if (!SQL) return;
    const f = await pickAndReadFile('문제은행 열기', BANK_FILTERS);
    if (!f) return;
    try {
      const b = Bank.open(SQL, f.bytes);
      if (!isTauri()) await idbSet(IDB_KEY, f.bytes);
      await adopt(b, f.path, true);
      toast(`${f.name} 을(를) 열었습니다`);
    } catch (e) {
      handleError(e);
    }
  }, [SQL, adopt, toast, handleError]);

  const closeBank = useCallback(async () => {
    await persist();
    setBank(null);
    setFilePath(null);
    bankRef.current = null;
    pathRef.current = null;
    lsSet(LS_PATH, null);
  }, [persist]);

  const value = useMemo<Ctx | null>(
    () =>
      SQL
        ? {
            SQL,
            bank,
            filePath,
            rev,
            saveState,
            lastSaved,
            reviewer,
            setReviewer: (v) => (setReviewerState(v), lsSet(LS_REVIEWER, v)),
            backupKeep,
            setBackupKeep: (n) => (setBackupKeepState(n), lsSet(LS_BACKUP_KEEP, String(n))),
            mutate,
            mutateAsync,
            toast,
            createBank,
            openBank,
            closeBank,
            flush: persist,
          }
        : null,
    [
      SQL,
      bank,
      filePath,
      rev,
      saveState,
      lastSaved,
      reviewer,
      backupKeep,
      mutate,
      mutateAsync,
      toast,
      createBank,
      openBank,
      closeBank,
      persist,
    ],
  );

  if (!value || booting) return <div className="empty">불러오는 중…</div>;
  return (
    <StudioContext.Provider value={value}>
      {children}
      {toastState && <div className={`toast ${toastState.error ? 'is-error' : ''}`}>{toastState.msg}</div>}
    </StudioContext.Provider>
  );
}
