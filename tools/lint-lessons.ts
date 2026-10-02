// 자습서 점검: 머리말, 필수 절, 수식(KaTeX), 그림, 문체, 용어 범위
// 사용: npx tsx tools/lint-lessons.ts [content/lessons/01.md ...]   (인자가 없으면 전체)
// 오류가 하나라도 있으면 종료 코드 1
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, join } from 'node:path';
import { figureName, parseLesson, renderLesson, type FigureSource } from '../packages/ui/src/lesson.ts';

const ROOT = new URL('..', import.meta.url).pathname;
const DIR = join(ROOT, 'content/lessons');
const FIG = join(DIR, 'fig');
const parts = JSON.parse(readFileSync(join(DIR, 'parts.json'), 'utf8')).parts as { part: number }[];
const curriculum = JSON.parse(readFileSync(join(ROOT, 'content/curriculum.json'), 'utf8'));
const chapterIds = new Set<string>(curriculum.chapters.map((c: { id: string }) => c.id));

/** 강 머리말의 장에 속한 용어 (문제은행 초안 기준) — 자습서가 설명하는지 확인용 */
function chapterTerms(ch: string): { id: string; names: string[] }[] {
  const f = join(ROOT, 'bank/incoming', `${ch}.json`);
  if (!existsSync(f)) return [];
  const x = JSON.parse(readFileSync(f, 'utf8'));
  return (x.terms ?? []).map((t: { id: string; term_en: string; term_ko?: string; abbreviation?: string }) => ({
    id: t.id,
    names: [t.term_ko, t.term_en, t.abbreviation]
      .filter((s): s is string => !!s)
      .flatMap((s) => [s, s.replace(/\s*\(.*?\)\s*/g, '')])
      .filter(Boolean),
  }));
}

const REQUIRED = ['이 강에서 얻는 것', '현장에서는', '헷갈리는 쌍', '이렇게 지시받음', '요약', '핵심 용어'];
/** 부록(part 9)은 강의 틀을 따르지 않음: 첫 절만 정해 둠 */
const APPENDIX_PART = 9;
const REQUIRED_APPENDIX = ['이 부록의 쓰임'];
const lessonFiles = readdirSync(DIR).filter((f) => /^[0-9A-Z]+\.md$/.test(f));
const targets = process.argv.slice(2).length ? process.argv.slice(2).map((f) => basename(f)) : lessonFiles;
const usedFigures = new Set<string>();

function resolveFigure(src: string): FigureSource | undefined {
  const name = figureName(src);
  const p = join(FIG, name);
  if (!existsSync(p)) return undefined;
  usedFigures.add(name);
  return name.endsWith('.svg') ? { kind: 'svg', svg: readFileSync(p, 'utf8') } : { kind: 'url', url: `fig/${name}` };
}

let totalErrors = 0;
let totalWarnings = 0;
for (const file of targets) {
  const errors: string[] = [];
  const warns: string[] = [];
  const raw = readFileSync(join(DIR, file), 'utf8');
  let lesson;
  try {
    lesson = parseLesson(raw);
  } catch (e) {
    console.log(`\n■ ${file}\n  ⛔ ${(e as Error).message}`);
    totalErrors++;
    continue;
  }
  const { meta, body } = lesson;
  if (`${meta.id}.md` !== file) errors.push(`머리말 id(${meta.id})와 파일 이름이 다름`);
  if (!parts.some((p) => p.part === meta.part)) errors.push(`part ${meta.part}가 parts.json에 없음`);
  for (const c of meta.chapters) if (!chapterIds.has(c)) errors.push(`없는 장: ${c}`);
  if (!['draft', 'reviewed'].includes(meta.status)) errors.push(`status는 draft 또는 reviewed`);

  // 구조
  const h1 = body.match(/^# (.+)$/m);
  if (!h1) errors.push('제목(# N강 … 또는 # 부록 X …)이 없음');
  const h2 = [...body.matchAll(/^## (.+)$/gm)].map((m) => m[1].trim());
  const isAppendix = meta.part === APPENDIX_PART;
  const req = isAppendix ? REQUIRED_APPENDIX : REQUIRED;
  const pos = req.map((r) => h2.indexOf(r));
  req.forEach((r, i) => pos[i] < 0 && errors.push(`필수 절 없음: ## ${r}`));
  if (pos.every((p) => p >= 0)) {
    if (pos[0] !== 0) errors.push(`첫 절은 '## ${req[0]}'`);
    if (!isAppendix && pos[pos.length - 1] !== h2.length - 1) errors.push(`마지막 절은 '## ${req[req.length - 1]}'`);
    for (let i = 1; i < pos.length; i++) if (pos[i] < pos[i - 1]) errors.push(`필수 절 순서: ${req.join(' → ')}`);
  }

  // 렌더링 (수식·그림)
  const r = renderLesson(body, resolveFigure);
  for (const i of r.issues) (i.kind === 'math' || /찾을 수 없음/.test(i.message) ? errors : warns).push(i.message);
  const leftover = r.html.replace(/<span class="vd-math">[\s\S]*?<!--\/vd-math-->/g, '').match(/\$/g);
  if (leftover) errors.push(`렌더링 후에도 $ 기호가 ${leftover.length}개 남음 (수식 구분자가 짝이 안 맞거나, 표 안 수식에 | 사용 → \\lvert·\\vert로 바꿀 것)`);

  const strayStars = r.html.replace(/<span class="vd-math">[\s\S]*?<!--\/vd-math-->/g, '').replace(/<code>[\s\S]*?<\/code>/g, '').match(/\*\*|__/g);
  if (strayStars) errors.push(`굵게(**) 표시가 적용되지 않고 ${strayStars.length}곳에 그대로 남음`);

  // 문체 (음슴체)
  const prose = body
    .replace(/\$\$[\s\S]*?\$\$/g, '')
    .replace(/\$[^$\n]*\$/g, '')
    .replace(/`[^`]*`/g, '');
  const polite = prose.match(/[^\n]{0,20}(습니다|합니다|입니다|세요)[.\s]/g);
  if (polite) warns.push(`존댓말 어미 ${polite.length}곳 (음슴체로): ${polite.slice(0, 3).join(' / ')}`);
  const plain = prose.match(/[^\n]{0,15}[가-힣](한다|된다|이다|있다|없다|하자)\.(\s|$)/g);
  if (plain) warns.push(`'~다.' 어미 ${plain.length}곳 (음슴체로): ${plain.slice(0, 3).join(' / ')}`);

  // 분량
  const chars = prose.replace(/\s+/g, '').length;
  if (chars < 2000) warns.push(`분량이 적음: ${chars}자 (권장 3,000~5,000)`);
  if (chars > (isAppendix ? 40000 : 7000)) warns.push(`분량이 많음: ${chars}자 (강을 나누는 것 검토)`);

  // 용어 범위
  const text = prose.toLowerCase();
  for (const c of meta.chapters)
    for (const t of chapterTerms(c))
      if (!t.names.some((n) => text.includes(n.toLowerCase()))) warns.push(`${c} 용어가 본문에 안 나옴: ${t.names[0]} (${t.id})`);

  totalErrors += errors.length;
  totalWarnings += warns.length;
  console.log(`\n■ ${file}  ${meta.title} · ${chars}자 · 수식 ${(r.html.match(/class="vd-math"/g) ?? []).length}개 · 그림 ${(r.html.match(/class="lesson-fig"/g) ?? []).length}개`);
  for (const e of errors) console.log(`  ⛔ ${e}`);
  for (const w of warns) console.log(`  ⚠️  ${w}`);
}

// 그림 파일 자체 점검
if (!process.argv.slice(2).length && existsSync(FIG)) {
  const errors: string[] = [];
  const warns: string[] = [];
  for (const f of readdirSync(FIG)) {
    const p = join(FIG, f);
    const size = statSync(p).size;
    if (!usedFigures.has(f)) warns.push(`어느 강에서도 쓰지 않는 그림: ${f}`);
    if (f.endsWith('.svg')) {
      const svg = readFileSync(p, 'utf8');
      const noStyle = svg.replace(/<style[\s\S]*?<\/style>/g, '');
      if (!/class="vdfig"/.test(svg)) errors.push(`${f}: <svg class="vdfig"> 필요 (앱 테마 색 적용)`);
      if (!/viewBox=/.test(svg)) errors.push(`${f}: viewBox 필요`);
      if (/\sid="/.test(svg)) errors.push(`${f}: id 속성 금지 (한 화면에 여러 그림이 들어갈 때 충돌)`);
      if (/\s(fill|stroke)="#|\s(fill|stroke)="(black|white|rgb)/i.test(noStyle))
        errors.push(`${f}: 색을 직접 지정함. s-*/f-* 클래스를 쓸 것 (다크 모드에서 안 보임)`);
      if (size > 200 * 1024) warns.push(`${f}: ${Math.round(size / 1024)}KB (SVG 200KB 이하 권장)`);
    } else if (/\.(png|jpe?g|webp)$/i.test(f)) {
      if (size > 800 * 1024) warns.push(`${f}: ${Math.round(size / 1024)}KB (참고 이미지 800KB 이하 권장)`);
    } else warns.push(`${f}: 지원하지 않는 형식 (svg, png, jpg, webp)`);
  }
  totalErrors += errors.length;
  totalWarnings += warns.length;
  console.log(`\n■ 그림 폴더  ${readdirSync(FIG).length}개`);
  for (const e of errors) console.log(`  ⛔ ${e}`);
  for (const w of warns) console.log(`  ⚠️  ${w}`);
}

console.log(`\n합계: 강 ${targets.length} · 오류 ${totalErrors} · 경고 ${totalWarnings}`);
process.exit(totalErrors ? 1 : 0);
