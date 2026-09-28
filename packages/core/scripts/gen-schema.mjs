// db/*.sql → src/schema.generated.ts (SQL 파일이 원본, TS는 번들용 사본)
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const bank = readFileSync(join(root, 'db/001_bank_schema.sql'), 'utf8');
const learner = readFileSync(join(root, 'db/002_learner_schema.sql'), 'utf8');
const out = `// 자동 생성 파일 — 직접 수정하지 말고 db/*.sql을 고친 뒤 \`npm run gen -w @visiondrill/core\` 실행\n` +
  `export const BANK_SCHEMA_SQL = ${JSON.stringify(bank)};\n` +
  `export const LEARNER_SCHEMA_SQL = ${JSON.stringify(learner)};\n`;
writeFileSync(join(root, 'src/schema.generated.ts'), out);
console.log('schema.generated.ts written');
