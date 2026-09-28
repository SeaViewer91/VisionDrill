// 문항 초안 자동 점검
// 사용: npx tsx tools/lint-drafts.ts [파일...]   (인자가 없으면 bank/incoming/*.json 전체)
// 오류가 하나라도 있으면 종료 코드 1
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { basename, join } from 'node:path';
import { fromXQuestion, normalize, parseExchange, validateQuestion, type XQuestion, type XTerm } from '../packages/core/src/index.ts';

const ROOT = new URL('..', import.meta.url).pathname;
const INCOMING = join(ROOT, 'bank/incoming');
const curriculum = JSON.parse(readFileSync(join(ROOT, 'content/curriculum.json'), 'utf8'));
const chapters = new Map<string, { track: string; code: string }>(curriculum.chapters.map((c: any) => [c.id, c]));
const targets: Record<string, number> = JSON.parse(readFileSync(join(ROOT, 'content/targets.json'), 'utf8'));

const sample = JSON.parse(readFileSync(join(ROOT, 'samples/sample-bank.json'), 'utf8'));
const allFiles = existsSync(INCOMING) ? readdirSync(INCOMING).filter((f) => f.endsWith('.json')).map((f) => join(INCOMING, f)) : [];
const targetFiles = process.argv.slice(2).length ? process.argv.slice(2).map((f) => (f.startsWith('/') ? f : join(process.cwd(), f))) : allFiles;

type Loaded = { file: string; qs: XQuestion[]; terms: XTerm[]; parseErrors: string[] };
function load(file: string): Loaded {
  const text = readFileSync(file, 'utf8');
  const { file: x, errors } = parseExchange(text);
  return { file, qs: x?.questions ?? [], terms: x?.terms ?? [], parseErrors: errors };
}
const loaded = new Map(allFiles.map((f) => [f, load(f)]));
for (const f of targetFiles) if (!loaded.has(f)) loaded.set(f, load(f));

// 전역 색인: 문항 ID, 용어 ID, 본문
const qidOwners = new Map<string, string[]>();
const termIds = new Set<string>((sample.terms as XTerm[]).map((t) => t.id));
for (const q of sample.questions as XQuestion[]) qidOwners.set(q.id, ['samples']);
for (const [f, l] of loaded) {
  for (const q of l.qs) qidOwners.set(q.id, [...(qidOwners.get(q.id) ?? []), basename(f)]);
  for (const t of l.terms) termIds.add(t.id);
}
const trigrams = (s: string) => {
  const n = normalize(s);
  const set = new Set<string>();
  for (let i = 0; i < n.length - 2; i++) set.add(n.slice(i, i + 3));
  return set;
};
const allStems: { id: string; tg: Set<string> }[] = [];
for (const q of sample.questions as XQuestion[]) allStems.push({ id: q.id, tg: trigrams(q.stem) });
for (const [, l] of loaded) for (const q of l.qs) allStems.push({ id: q.id, tg: trigrams(q.stem) });
const jaccard = (a: Set<string>, b: Set<string>) => {
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  return inter / (a.size + b.size - inter || 1);
};

const BANNED = /모두\s*(정답|맞|옳)|위의?\s*(보기|모든)|보기\s*모두|①|②|③|④|⑤/;
let totalErrors = 0;
let totalWarnings = 0;
let totalQ = 0;

for (const file of targetFiles) {
  const l = loaded.get(file)!;
  const errors: string[] = [...l.parseErrors];
  const warns: string[] = [];
  const chapterId = basename(file).replace(/\.json$/, '');
  if (!chapters.has(chapterId)) errors.push(`파일 이름이 장 ID가 아닙니다: ${chapterId} (예: G.aug.json)`);
  const ch = chapters.get(chapterId);

  // 용어
  const seenTerm = new Set<string>();
  for (const t of l.terms) {
    if (!/^[a-z0-9][a-z0-9-]*$/.test(t.id)) errors.push(`용어 ID 형식: ${t.id}`);
    if (seenTerm.has(t.id)) errors.push(`용어 ID 중복: ${t.id}`);
    seenTerm.add(t.id);
    if ((sample.terms as XTerm[]).some((s) => s.id === t.id)) warns.push(`기존 용어를 다시 정의함(병합 시 기존 것 유지): ${t.id}`);
    if (!t.definition?.trim()) errors.push(`용어 정의 없음: ${t.id}`);
    if (!t.usage_example?.trim()) warns.push(`용어 실무 예문 없음: ${t.id}`);
    if (t.origin && !['standard', 'project'].includes(t.origin)) errors.push(`용어 origin 값 오류: ${t.id}`);
  }

  // 문항
  const byLevel: Record<number, number> = {};
  const byType: Record<string, number> = {};
  const pos = [0, 0, 0, 0, 0];
  let choiceQ = 0;
  let longest = 0;
  const termLevels = new Map<string, Set<number>>();
  for (const x of l.qs) {
    totalQ++;
    const tag = x.id ?? '(id 없음)';
    if (x.chapter !== chapterId) errors.push(`${tag}: chapter가 파일과 다름 (${x.chapter})`);
    if (ch && !new RegExp(`^${ch.track}-${ch.code}-\\d{3}$`).test(x.id)) errors.push(`${tag}: ID 형식은 ${ch.track}-${ch.code}-NNN`);
    if ((qidOwners.get(x.id) ?? []).length > 1) errors.push(`${tag}: 문항 ID 중복 (${qidOwners.get(x.id)!.join(', ')})`);
    if (x.status && x.status !== 'draft') errors.push(`${tag}: status는 draft여야 함`);
    let q;
    try {
      q = fromXQuestion(x);
    } catch (e) {
      errors.push(`${tag}: 변환 실패 ${(e as Error).message}`);
      continue;
    }
    for (const i of validateQuestion(q)) {
      if (i.level === 'error') errors.push(`${tag}: ${i.message}`);
      else if (!/출처가 비어/.test(i.message)) warns.push(`${tag}: ${i.message}`);
    }
    if (!x.source_ref) errors.push(`${tag}: source_ref 없음`);
    if (!x.explanation_full) warns.push(`${tag}: 상세 해설 없음`);
    byType[x.type] = (byType[x.type] ?? 0) + 1;
    byLevel[x.cognitive_level ?? 1] = (byLevel[x.cognitive_level ?? 1] ?? 0) + 1;
    if (x.type === 'scenario') {
      if (!/^\s*지시\s*:/.test(x.stem)) errors.push(`${tag}: 지시문 해석형 본문은 '지시:'로 시작`);
      if (x.cognitive_level !== 3) errors.push(`${tag}: 지시문 해석형은 cognitive_level 3`);
    }
    if (x.type !== 'short') {
      choiceQ++;
      if (x.answer) pos[x.answer - 1]++;
      const lens = (x.choices ?? []).map((c) => [...c.text].length);
      const ans = (x.answer ?? 1) - 1;
      if (lens.length === 5 && lens[ans] > Math.max(...lens.filter((_, i) => i !== ans))) longest++;
      if ((x.choices ?? []).some((c) => BANNED.test(c.text))) errors.push(`${tag}: 금지된 보기 표현(모두 정답/위 보기/번호 참조)`);
    } else {
      const en = x.accepted_answers?.en ?? [];
      const ko = x.accepted_answers?.ko ?? [];
      if (!en.length || !ko.length) warns.push(`${tag}: 단답형은 영문·한글 허용 답을 모두 권장`);
    }
    for (const tid of x.terms ?? []) {
      if (!termIds.has(tid)) warns.push(`${tag}: 정의되지 않은 용어 참조 ${tid} (다른 장 소관이면 병합 때 맞춤)`);
      if (!termLevels.has(tid)) termLevels.set(tid, new Set());
      termLevels.get(tid)!.add(x.cognitive_level ?? 1);
    }
    if (!(x.terms ?? []).length) warns.push(`${tag}: 연결 용어 없음`);
    // 유사 본문
    const tg = trigrams(x.stem);
    for (const o of allStems) {
      if (o.id === x.id) continue;
      if (tg.size > 10 && jaccard(tg, o.tg) >= 0.8) warns.push(`${tag}: ${o.id}와 본문이 매우 비슷함`);
    }
  }
  // 분포 점검
  if (choiceQ >= 8) {
    const maxShare = Math.max(...pos) / choiceQ;
    if (maxShare > 0.35) errors.push(`정답 위치 쏠림: ${pos.join('/')} (한 번호 최대 35%)`);
    if (longest / choiceQ > 0.4) errors.push(`정답이 가장 긴 보기인 비율 ${Math.round((longest / choiceQ) * 100)}% (최대 40%)`);
  }
  for (const t of l.terms) {
    const lv = termLevels.get(t.id);
    if (!lv) warns.push(`용어 ${t.id}: 연결된 문항 없음`);
    else if (!lv.has(1) || !lv.has(2)) warns.push(`용어 ${t.id}: 1단계·2단계 문항을 모두 갖추지 못함 (${[...lv].sort().join(',')})`);
  }
  const target = targets[chapterId];
  if (target && l.qs.length < target) warns.push(`목표 문항 수 미달: ${l.qs.length}/${target}`);

  totalErrors += errors.length;
  totalWarnings += warns.length;
  console.log(`\n■ ${basename(file)}  문항 ${l.qs.length}${target ? `/${target}` : ''} · 용어 ${l.terms.length}`);
  console.log(`  유형 ${JSON.stringify(byType)} · 단계 ${JSON.stringify(byLevel)} · 정답위치 ${pos.join('/')} · 정답최장 ${choiceQ ? Math.round((longest / choiceQ) * 100) : 0}%`);
  for (const e of errors) console.log(`  ⛔ ${e}`);
  for (const w of warns) console.log(`  ⚠️  ${w}`);
}
console.log(`\n합계: 파일 ${targetFiles.length} · 문항 ${totalQ} · 오류 ${totalErrors} · 경고 ${totalWarnings}`);
process.exit(totalErrors ? 1 : 0);
