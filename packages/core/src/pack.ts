import type { SqlJsStatic } from 'sql.js';
import { Bank, BankError } from './bank';
import { BANK_SCHEMA_SQL } from './schema.generated';
import { Db, nowIso, sha256Hex, type Params, type Row } from './sql';

export interface PackManifest {
  kind: 'pack';
  bank_name: string;
  pack_version: string;
  exported_at: string;
  question_count: number;
  term_count: number;
  schema_version: string;
}

const TRIGGERS = [
  'trg_questions_touch',
  'trg_terms_touch',
  'trg_questions_insert_draft',
  'trg_questions_review_gate',
  'trg_questions_delete_guard',
  'trg_questions_no_unretire',
  'trg_terms_delete_guard',
];

/** 문제은행 스키마로 빈 DB를 만들고, 콘텐츠를 그대로 복사할 수 있도록 트리거를 제거 */
export function createContentDb(SQL: SqlJsStatic): Db {
  const db = Db.create(SQL);
  db.exec(BANK_SCHEMA_SQL);
  for (const t of TRIGGERS) db.exec(`DROP TRIGGER IF EXISTS ${t}`);
  db.exec(`DELETE FROM track_prereqs; DELETE FROM tracks;`);
  return db;
}

export function copyRows(src: Db, dst: Db, table: string, where = '', params: Params = []): number {
  const rows = src.all<Row>(`SELECT * FROM ${table} ${where}`, params);
  if (!rows.length) return 0;
  const cols = Object.keys(rows[0]);
  const sql = `INSERT INTO ${table}(${cols.join(',')}) VALUES(${cols.map(() => '?').join(',')})`;
  for (const r of rows)
    dst.run(
      sql,
      cols.map((c) => r[c]),
    );
  return rows.length;
}

/**
 * 배포 팩(.vdpack) 생성: 검수 완료 문항과 그 보기·허용답·태그·용어·이미지만 담은 읽기 전용 SQLite 파일.
 * 생성 기록은 원본 DB의 pack_exports에 남긴다.
 */
export async function exportPack(bank: Bank, SQL: SqlJsStatic, packVersion: string, note?: string) {
  if (!/^[0-9A-Za-z.\-_]+$/.test(packVersion)) throw new BankError('팩 버전은 영문·숫자·점·하이픈만 쓸 수 있습니다 (예: 2026.10.01)');
  if (bank.db.value(`SELECT 1 FROM pack_exports WHERE pack_version=?`, [packVersion]))
    throw new BankError(`이미 내보낸 팩 버전입니다: ${packVersion}`);
  const src = bank.db;
  const qCount = src.value<number>(`SELECT COUNT(*) FROM questions WHERE status='reviewed'`) ?? 0;
  if (qCount === 0) throw new BankError('검수 완료 문항이 없습니다');

  const dst = createContentDb(SQL);
  const inQ = `question_id IN (SELECT id FROM questions WHERE status='reviewed')`;
  dst.tx(() => {
    copyRows(src, dst, 'tracks');
    copyRows(src, dst, 'track_prereqs');
    copyRows(src, dst, 'chapters');
    copyRows(src, dst, 'assets', `WHERE id IN (SELECT image_asset_id FROM questions WHERE status='reviewed')`);
    copyRows(src, dst, 'terms', `WHERE status='reviewed' OR id IN (SELECT term_id FROM question_terms WHERE ${inQ})`);
    copyRows(src, dst, 'questions', `WHERE status='reviewed'`);
    copyRows(src, dst, 'choices', `WHERE ${inQ}`);
    copyRows(src, dst, 'accepted_answers', `WHERE ${inQ}`);
    copyRows(src, dst, 'question_tags', `WHERE ${inQ}`);
    copyRows(src, dst, 'question_terms', `WHERE ${inQ}`);
  });
  const termCount = dst.value<number>(`SELECT COUNT(*) FROM terms`) ?? 0;
  const manifest: PackManifest = {
    kind: 'pack',
    bank_name: bank.meta('bank_name') ?? 'VisionDrill',
    pack_version: packVersion,
    exported_at: nowIso(),
    question_count: qCount,
    term_count: termCount,
    schema_version: bank.meta('schema_version') ?? '1',
  };
  dst.run(`DELETE FROM bank_meta`);
  for (const [k, v] of Object.entries(manifest)) dst.run(`INSERT INTO bank_meta(key,value) VALUES(?,?)`, [k, String(v)]);
  dst.exec('VACUUM');
  const bytes = dst.export();
  dst.close();
  const hash = await sha256Hex(bytes);
  src.run(`INSERT INTO pack_exports(pack_version,question_count,term_count,sha256,note) VALUES(?,?,?,?,?)`, [
    packVersion,
    qCount,
    termCount,
    hash,
    note ?? null,
  ]);
  return { bytes, manifest, sha256: hash };
}

export function readPackManifest(db: Db): PackManifest {
  const rows = db.all<{ key: string; value: string }>(`SELECT key, value FROM bank_meta`);
  const m = Object.fromEntries(rows.map((r) => [r.key, r.value]));
  if (m.kind !== 'pack') throw new BankError('VisionDrill 문제 팩(.vdpack) 파일이 아닙니다');
  return {
    kind: 'pack',
    bank_name: m.bank_name,
    pack_version: m.pack_version,
    exported_at: m.exported_at,
    question_count: Number(m.question_count),
    term_count: Number(m.term_count),
    schema_version: m.schema_version,
  };
}
