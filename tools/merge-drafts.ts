// 초안(bank/incoming/*.json) + 샘플 → 원본 문제은행 DB 하나로 합치기
// 사용: npx tsx tools/merge-drafts.ts [--out bank/visiondrill-bank.db] [--preview-pack]
//   - 장: content/curriculum.json
//   - 용어: 샘플 용어가 우선, 같은 ID가 여러 장에 있으면 커리큘럼 순서상 먼저 나온 장의 정의를 씀
//   - 다른 이름으로 참조한 용어는 ALIAS로 맞추고, 아무 데도 없는 용어는 STUB_TERMS로 새로 정의
//   - 샘플 문항은 검수 상태 유지, 새 문항은 모두 초안(draft)
//   - --preview-pack: 초안까지 전부 담은 "사람 검수 전" 미리보기 팩(bank/preview-draft.vdpack)도 만든다
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import {
  Bank,
  exportPack,
  importExchange,
  loadSql,
  parseExchange,
  type ExchangeFile,
  type XChapter,
  type XQuestion,
  type XTerm,
} from '../packages/core/src/index.ts';

const ROOT = new URL('..', import.meta.url).pathname;
const args = process.argv.slice(2);
const outArg = args.indexOf('--out');
const OUT = join(ROOT, outArg >= 0 ? args[outArg + 1] : 'bank/visiondrill-bank.db');
const PREVIEW = args.includes('--preview-pack');
const INCOMING = join(ROOT, 'bank/incoming');

/** 초안이 다른 이름으로 참조한 용어 → 실제 정의된 용어 ID */
const ALIAS: Record<string, string> = {
  'connected-component': 'connected-component-labeling',
  ece: 'expected-calibration-error',
};

/** 어느 장에도 정의되지 않았지만 여러 문항이 참조하는 용어 */
const STUB_TERMS: XTerm[] = [
  {
    id: 'speckle',
    term_en: 'Speckle',
    term_ko: '스페클',
    abbreviation: null,
    definition:
      'SAR처럼 결맞은(coherent) 파동으로 영상을 만들 때, 한 해상 셀 안 여러 산란체의 반사파가 간섭해 생기는 곱셈성 입자 모양 잡음. 평균을 내면 줄지만 해상도를 잃는다.',
    usage_example: 'Sentinel-1 VV 영상은 스페클이 심하니까 Lee 필터 한 번 걸고 수계 추출 돌려.',
    origin: 'standard',
    track: 'D',
  },
  {
    id: 'clahe',
    term_en: 'Contrast Limited Adaptive Histogram Equalization',
    term_ko: '대비 제한 적응형 히스토그램 평활화',
    abbreviation: 'CLAHE',
    definition:
      '영상을 작은 타일로 나눠 타일마다 히스토그램 평활화를 하되, 히스토그램 높이를 잘라(clip limit) 잡음이 과하게 증폭되지 않게 하고 타일 경계는 보간으로 잇는 대비 향상 기법.',
    usage_example: '그림자 진 골목 건물이 안 보이니까 표출용 영상만 CLAHE 걸어서 라벨러한테 넘겨.',
    origin: 'standard',
    track: 'D',
  },
  {
    id: 'data-augmentation',
    term_en: 'Data Augmentation',
    term_ko: '데이터 증강',
    abbreviation: null,
    definition:
      '라벨의 의미를 바꾸지 않는 변환(뒤집기·회전·밝기 변화·잘라 붙이기 등)을 학습 데이터에 적용해 겉보기 데이터 다양성을 늘리고 과적합을 줄이는 기법의 총칭.',
    usage_example: '학습 영상이 한 계절 것뿐이니까 증강에 밝기·대비 변화 좀 넣어서 다시 돌려봐.',
    origin: 'standard',
    track: 'G',
  },
  {
    id: 'cutmix',
    term_en: 'CutMix',
    term_ko: '컷믹스',
    abbreviation: null,
    definition:
      '한 이미지의 사각형 영역을 잘라 다른 이미지의 같은 위치에 붙이고, 라벨도 붙인 면적 비율만큼 섞는 증강. Mixup과 달리 픽셀을 반투명하게 섞지 않는다.',
    usage_example: '토지피복 분류기가 배경 문맥에 너무 기대는 것 같으니 CutMix 넣어서 비교해봐.',
    origin: 'standard',
    track: 'G',
  },
  {
    id: 'active-learning',
    term_en: 'Active Learning',
    term_ko: '능동 학습',
    abbreviation: null,
    definition:
      '모델이 가장 불확실해하거나 정보량이 큰 표본을 골라 사람에게 라벨링을 요청하고, 새 라벨로 다시 학습하는 과정을 반복해 라벨링 비용 대비 성능을 높이는 방법.',
    usage_example: '이번 달 라벨링 예산이 500장뿐이니까 능동 학습으로 신뢰도 낮은 타일부터 뽑아서 보내.',
    origin: 'standard',
    track: 'G',
  },
];

// ---------------------------------------------------------------- 읽기
const curriculum = JSON.parse(readFileSync(join(ROOT, 'content/curriculum.json'), 'utf8'));
const chapters: XChapter[] = curriculum.chapters;
const chapterOrder = new Map(chapters.map((c, i) => [c.id, i]));
const trackOf = new Map(chapters.map((c) => [c.id, c.track]));

const sample = JSON.parse(readFileSync(join(ROOT, 'samples/sample-bank.json'), 'utf8')) as ExchangeFile;
const draftFiles = readdirSync(INCOMING)
  .filter((f) => f.endsWith('.json'))
  .sort((a, b) => (chapterOrder.get(a.slice(0, -5)) ?? 999) - (chapterOrder.get(b.slice(0, -5)) ?? 999));

const drafts: { chapter: string; file: ExchangeFile }[] = [];
for (const f of draftFiles) {
  const { file, errors } = parseExchange(readFileSync(join(INCOMING, f), 'utf8'));
  if (!file) throw new Error(`${f}: ${errors.join('; ')}`);
  drafts.push({ chapter: f.slice(0, -5), file });
}

// ---------------------------------------------------------------- 용어 합치기
const report: string[] = [];
const terms = new Map<string, XTerm & { from: string }>();
for (const t of sample.terms ?? []) terms.set(t.id, { ...t, from: 'samples' });
const dupTerms: string[] = [];
for (const { chapter, file } of drafts) {
  for (const t of file.terms ?? []) {
    const prev = terms.get(t.id);
    if (prev) {
      dupTerms.push(`- \`${t.id}\`: ${chapter}의 정의는 버리고 ${prev.from}의 정의 사용`);
      continue;
    }
    terms.set(t.id, { ...t, track: t.track ?? trackOf.get(chapter) ?? null, status: 'draft', from: chapter });
  }
}
for (const t of STUB_TERMS) if (!terms.has(t.id)) terms.set(t.id, { ...t, status: 'draft', from: '병합 도구(STUB)' });

// ---------------------------------------------------------------- 문항 합치기
const questions: XQuestion[] = [];
const seenQ = new Set((sample.questions ?? []).map((q) => q.id));
const fixes: string[] = [];
const unresolved = new Map<string, string[]>();
for (const { chapter, file } of drafts) {
  for (const q of file.questions ?? []) {
    if (seenQ.has(q.id)) throw new Error(`문항 ID 중복: ${q.id}`);
    seenQ.add(q.id);
    const mapped: string[] = [];
    for (const tid of q.terms ?? []) {
      const real = ALIAS[tid] ?? tid;
      if (real !== tid) fixes.push(`- ${q.id}: 용어 \`${tid}\` → \`${real}\``);
      if (!terms.has(real)) {
        unresolved.set(real, [...(unresolved.get(real) ?? []), q.id]);
        continue;
      }
      if (!mapped.includes(real)) mapped.push(real);
    }
    questions.push({ ...q, chapter, terms: mapped, status: 'draft' });
  }
}
if (unresolved.size) {
  console.error('정의되지 않은 용어:', Object.fromEntries(unresolved));
  process.exit(1);
}

// ---------------------------------------------------------------- DB 만들기
const SQL = await loadSql();
const bank = Bank.create(SQL, 'VisionDrill Core');
const allTerms = [...terms.values()].map(({ from: _from, ...t }) => t);
const r1 = importExchange(
  bank,
  { format: 'visiondrill-bank-exchange', format_version: 1, chapters, terms: allTerms, questions: sample.questions },
  { reviewer: 'sample' },
);
const r2 = importExchange(bank, { format: 'visiondrill-bank-exchange', format_version: 1, questions });
const problems = [...r1.messages, ...r2.messages].filter((m) => m.level !== 'info');
if (r1.failed || r2.failed) {
  console.error(problems);
  process.exit(1);
}
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, bank.export());

// ---------------------------------------------------------------- 보고서
const stats = bank.db.all<{ track: string; type: string; status: string; n: number }>(
  `SELECT c.track_id AS track, q.type, q.status, COUNT(*) AS n
     FROM questions q JOIN chapters c ON c.id = q.chapter_id GROUP BY 1,2,3`,
);
const byTrack = new Map<string, { short: number; mcq: number; scenario: number; reviewed: number; draft: number }>();
for (const s of stats) {
  const t = byTrack.get(s.track) ?? { short: 0, mcq: 0, scenario: 0, reviewed: 0, draft: 0 };
  t[s.type as 'short'] += s.n;
  t[s.status as 'draft'] = (t[s.status as 'draft'] ?? 0) + s.n;
  byTrack.set(s.track, t);
}
const total = stats.reduce((a, s) => a + s.n, 0);
const termCount = bank.db.value<number>('SELECT COUNT(*) FROM terms') ?? 0;
const lines = [
  `# 문제은행 병합 보고서`,
  ``,
  `- 생성: ${new Date().toISOString()}`,
  `- 결과 DB: \`${OUT.replace(ROOT, '')}\``,
  `- 문항 ${total}개 (검수 완료 ${stats.filter((s) => s.status === 'reviewed').reduce((a, s) => a + s.n, 0)} · 초안 ${stats.filter((s) => s.status === 'draft').reduce((a, s) => a + s.n, 0)}) · 용어 ${termCount}개 · 장 ${chapters.length}개`,
  ``,
  `| 트랙 | 문항 | 단답 | 5지선다 | 지시문 | 검수 완료 | 초안 |`,
  `|---|---|---|---|---|---|---|`,
  ...[...byTrack.entries()]
    .sort()
    .map(([k, t]) => `| ${k} | ${t.short + t.mcq + t.scenario} | ${t.short} | ${t.mcq} | ${t.scenario} | ${t.reviewed ?? 0} | ${t.draft ?? 0} |`),
  ``,
  `## 여러 장에서 정의한 용어 (${dupTerms.length})`,
  ...dupTerms,
  ``,
  `## 병합 도구가 새로 정의한 용어 (${STUB_TERMS.length})`,
  ...STUB_TERMS.map((t) => `- \`${t.id}\` ${t.term_en} / ${t.term_ko}`),
  ``,
  `## 용어 참조 이름 맞춤 (${fixes.length})`,
  ...fixes,
  ``,
  `## 가져오기 경고 (${problems.length})`,
  ...problems.map((m) => `- ${m.id}: ${m.message}`),
];
report.push(...lines);
writeFileSync(join(dirname(OUT), 'merge-report.md'), report.join('\n') + '\n');
console.log(lines.slice(0, 18).join('\n'));

// ---------------------------------------------------------------- 미리보기 팩 (선택)
if (PREVIEW) {
  const tmp = Bank.open(SQL, bank.export());
  for (const s of tmp.listQuestions()) if (s.status === 'draft') tmp.setStatus(s.id, 'reviewed', 'Claude 교차검증(사람 검수 전)');
  tmp.db.run(`UPDATE terms SET status='reviewed'`);
  const { bytes, manifest } = await exportPack(tmp, SQL, `preview-${new Date().toISOString().slice(0, 10)}`, '미리보기 팩 (사람 검수 전)');
  const packPath = join(dirname(OUT), 'preview-draft.vdpack');
  writeFileSync(packPath, bytes);
  console.log('미리보기 팩:', packPath.replace(ROOT, ''), manifest.question_count ?? '', `${(bytes.length / 1024 / 1024).toFixed(1)} MB`);
}
if (!existsSync(OUT)) process.exit(1);
