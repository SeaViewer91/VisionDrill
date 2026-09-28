import { BankError, Learner, loadSql } from '@visiondrill/core';
import {
  appDataFile,
  createQueue,
  idbGet,
  idbSet,
  isTauri,
  pathExists,
  pickAndReadFile,
  readPath,
  writeAtomic,
} from '@visiondrill/platform';
import type { SqlJsStatic } from 'sql.js';
import wasmUrl from 'sql.js/dist/sql-wasm.wasm?url';
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';

const LEARNER_FILE = 'visiondrill-learner.db';
const IDB_KEY = 'drill-learner';
export const REPO_URL = 'https://github.com/SeaViewer91/VisionDrill';

interface Ctx {
  SQL: SqlJsStatic;
  learner: Learner;
  rev: number;
  /** 마지막 저장 실패 메시지 (성공하면 null) */
  saveError: string | null;
  /** 쓰기 작업 후 자동 저장 */
  mutate: <T>(fn: (l: Learner) => T, okMsg?: string) => T | undefined;
  toast: (msg: string, error?: boolean) => void;
  importPackFile: () => Promise<void>;
  save: () => Promise<void>;
}

const DrillContext = createContext<Ctx | null>(null);

export function useDrill(): Ctx {
  const c = useContext(DrillContext);
  if (!c) throw new Error('DrillProvider 밖에서 사용됨');
  return c;
}

async function loadBundledPack(): Promise<Uint8Array | null> {
  try {
    const r = await fetch('./packs/default.vdpack');
    if (!r.ok) return null;
    return new Uint8Array(await r.arrayBuffer());
  } catch {
    return null;
  }
}

export function DrillProvider({ children }: { children: ReactNode }) {
  const [SQL, setSQL] = useState<SqlJsStatic | null>(null);
  const [learner, setLearner] = useState<Learner | null>(null);
  const [rev, setRev] = useState(0);
  const [toastState, setToast] = useState<{ msg: string; error?: boolean } | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const learnerRef = useRef<Learner | null>(null);
  const pathRef = useRef<string | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const queue = useRef(createQueue());
  const started = useRef(false);
  learnerRef.current = learner;

  const toast = useCallback((msg: string, error = false) => {
    setToast({ msg, error });
    window.setTimeout(() => setToast((t) => (t?.msg === msg ? null : t)), error ? 6000 : 2500);
  }, []);

  const save = useCallback(async () => {
    window.clearTimeout(timer.current);
    await queue.current(async () => {
      const l = learnerRef.current;
      if (!l) return;
      try {
        const bytes = l.export();
        if (pathRef.current) await writeAtomic(pathRef.current, bytes);
        else await idbSet(IDB_KEY, bytes);
        setSaveError(null);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        console.error('save failed', e);
        setSaveError(msg);
        toast(`학습 기록 저장 실패: ${msg}`, true);
      }
    });
  }, [toast]);

  const mutate = useCallback(
    <T,>(fn: (l: Learner) => T, okMsg?: string): T | undefined => {
      const l = learnerRef.current;
      if (!l) return undefined;
      try {
        const r = fn(l);
        setRev((x) => x + 1);
        window.clearTimeout(timer.current);
        timer.current = window.setTimeout(save, 300);
        if (okMsg) toast(okMsg);
        return r;
      } catch (e) {
        toast(e instanceof BankError ? e.message : `오류: ${(e as Error)?.message ?? e}`, true);
        if (!(e instanceof BankError)) console.error(e);
        return undefined;
      }
    },
    [save, toast],
  );

  useEffect(() => {
    if (started.current) return; // 개발 모드(StrictMode)에서 두 번 실행되는 것 방지
    started.current = true;
    (async () => {
      try {
        const sql = await loadSql(wasmUrl);
        setSQL(sql);
        let l: Learner | null = null;
        if (isTauri()) {
          const path = await appDataFile(LEARNER_FILE);
          pathRef.current = path;
          if (path && (await pathExists(path))) l = Learner.open(sql, await readPath(path));
        } else {
          const bytes = await idbGet(IDB_KEY);
          if (bytes) l = Learner.open(sql, bytes);
        }
        const firstRun = !l;
        if (!l) l = Learner.create(sql);
        // 앱에 들어 있는 기본 문제 팩: 첫 실행이면 설치, 앱 업데이트로 기본 팩이 바뀌었으면 교체 (직접 가져온 팩은 유지)
        const pack = await loadBundledPack();
        const applied = pack ? l.applyBundledPack(sql, pack) : null;
        if (applied && !firstRun)
          toast(`기본 문제 팩이 ${applied.manifest.pack_version}(으)로 업데이트됨: 문항 ${applied.manifest.question_count}개 (새 문항 ${applied.added})`);
        learnerRef.current = l;
        setLearner(l);
        await save();
      } catch (e) {
        console.error(e);
        setFatal((e as Error).message ?? String(e));
      }
    })();
  }, [save, toast]);

  useEffect(() => {
    const h = () => {
      const l = learnerRef.current;
      if (l && !pathRef.current) void idbSet(IDB_KEY, l.export());
    };
    window.addEventListener('beforeunload', h);
    return () => window.removeEventListener('beforeunload', h);
  }, []);

  const importPackFile = useCallback(async () => {
    if (!SQL) return;
    const f = await pickAndReadFile('문제 팩 가져오기', [{ name: 'VisionDrill 문제 팩', extensions: ['vdpack'] }]);
    if (!f) return;
    const r = mutate((l) => l.importPack(SQL, f.bytes));
    if (r)
      toast(
        `${r.manifest.pack_version} 팩 적용: 문항 ${r.manifest.question_count}개 (새 문항 ${r.added}` +
          (r.reset ? `, 정답 변경으로 복습 초기화 ${r.reset}` : '') +
          (r.removed ? `, 빠진 문항 ${r.removed}` : '') +
          ')',
      );
  }, [SQL, mutate, toast]);

  const value = useMemo<Ctx | null>(
    () => (SQL && learner ? { SQL, learner, rev, saveError, mutate, toast, importPackFile, save } : null),
    [SQL, learner, rev, saveError, mutate, toast, importPackFile, save],
  );

  if (fatal) return <div className="empty">시작하지 못했습니다: {fatal}</div>;
  if (!value) return <div className="empty">불러오는 중…</div>;
  return (
    <DrillContext.Provider value={value}>
      {children}
      {toastState && <div className={`toast ${toastState.error ? 'is-error' : ''}`}>{toastState.msg}</div>}
    </DrillContext.Provider>
  );
}
