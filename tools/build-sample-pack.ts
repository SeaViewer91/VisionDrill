// 샘플 JSON → 원본 DB(samples/sample-bank.db) + 샘플 팩(samples/sample.vdpack, 테스트·시연용)
// 학습 앱 기본 팩은 tools/merge-drafts.ts --bundle 이 만든다
import { readFileSync, writeFileSync } from 'node:fs';
import { Bank, exportPack, importExchange, loadSql, parseExchange } from '../packages/core/src/index.ts';

const SQL = await loadSql();
const bank = Bank.create(SQL, 'VisionDrill Sample');
const { file, errors } = parseExchange(readFileSync('samples/sample-bank.json', 'utf8'));
if (!file) throw new Error(errors.join('\n'));
const res = importExchange(bank, file, { reviewer: 'sample' });
console.log(res);
if (res.failed) process.exit(1);
const { bytes, manifest } = await exportPack(bank, SQL, 'sample-0.1.0', '샘플 팩');
writeFileSync('samples/sample-bank.db', bank.export());
writeFileSync('samples/sample.vdpack', bytes);
console.log('pack', manifest, bytes.length, 'bytes');
