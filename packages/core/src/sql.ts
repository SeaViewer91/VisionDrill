import initSqlJs, { type Database, type SqlJsStatic, type SqlValue } from 'sql.js';

export type Params = SqlValue[] | Record<string, SqlValue>;
export type Row = Record<string, SqlValue>;

let sqlPromise: Promise<SqlJsStatic> | null = null;

/**
 * sql.js(WASM SQLite) 초기화. 브라우저/Tauri에서는 wasm 파일 URL을 넘긴다.
 * Node(테스트·도구)에서는 인자 없이 호출하면 패키지 안의 wasm을 찾는다.
 */
export function loadSql(wasmUrl?: string): Promise<SqlJsStatic> {
  if (!sqlPromise) {
    sqlPromise = initSqlJs(wasmUrl ? { locateFile: () => wasmUrl } : undefined);
  }
  return sqlPromise;
}

/** sql.js Database를 감싼 얇은 래퍼: 파라미터 바인딩, 트랜잭션, 행 객체 변환 */
export class Db {
  constructor(public readonly raw: Database) {
    this.raw.exec('PRAGMA foreign_keys = ON;');
  }

  static create(SQL: SqlJsStatic, bytes?: Uint8Array): Db {
    return new Db(bytes ? new SQL.Database(bytes) : new SQL.Database());
  }

  exec(sql: string): void {
    this.raw.exec(sql);
  }

  run(sql: string, params: Params = []): void {
    this.raw.run(sql, params as any);
  }

  all<T = Row>(sql: string, params: Params = []): T[] {
    const stmt = this.raw.prepare(sql);
    try {
      stmt.bind(params as any);
      const rows: T[] = [];
      while (stmt.step()) rows.push(stmt.getAsObject() as T);
      return rows;
    } finally {
      stmt.free();
    }
  }

  get<T = Row>(sql: string, params: Params = []): T | undefined {
    return this.all<T>(sql, params)[0];
  }

  value<T = SqlValue>(sql: string, params: Params = []): T | undefined {
    const row = this.get(sql, params);
    return row ? (Object.values(row)[0] as T) : undefined;
  }

  private depth = 0;

  /** 중첩 가능한 트랜잭션 (바깥은 BEGIN, 안쪽은 SAVEPOINT) */
  tx<T>(fn: () => T): T {
    const sp = `sp_${this.depth}`;
    if (this.depth === 0) this.raw.exec('BEGIN');
    else this.raw.exec(`SAVEPOINT ${sp}`);
    this.depth++;
    try {
      const result = fn();
      this.depth--;
      if (this.depth === 0) this.raw.exec('COMMIT');
      else this.raw.exec(`RELEASE ${sp}`);
      return result;
    } catch (e) {
      this.depth--;
      if (this.depth === 0) this.raw.exec('ROLLBACK');
      else this.raw.exec(`ROLLBACK TO ${sp}; RELEASE ${sp}`);
      throw e;
    }
  }

  export(): Uint8Array {
    if (this.depth > 0) throw new Error('트랜잭션 중에는 내보낼 수 없습니다');
    const bytes = this.raw.export();
    // sql.js의 export()는 DB를 닫았다 다시 열어서 PRAGMA가 초기화된다
    this.raw.exec('PRAGMA foreign_keys = ON;');
    return bytes;
  }

  close(): void {
    this.raw.close();
  }
}

export function nowIso(): string {
  return new Date().toISOString();
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const buf = await globalThis.crypto.subtle.digest('SHA-256', bytes as unknown as ArrayBuffer);
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}
