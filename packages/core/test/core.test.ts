import { beforeAll, describe, expect, it } from 'vitest';
import {
  Bank,
  Learner,
  emptyQuestion,
  exportExchange,
  exportPack,
  gradeShort,
  importExchange,
  loadSql,
  normalize,
  parseExchange,
  validateQuestion,
  type Question,
} from '../src/index';

let SQL: Awaited<ReturnType<typeof loadSql>>;
beforeAll(async () => {
  SQL = await loadSql();
});

function seedBank(): Bank {
  const bank = Bank.create(SQL);
  bank.saveChapter({ id: 'F.nms', track_id: 'F', code: 'NMS', name_ko: 'NMS', name_en: 'NMS', sort_order: 1, stage: 1, source_ref: null });
  bank.saveTerm({
    id: 'nms',
    term_en: 'Non-Maximum Suppression',
    term_ko: '비최대 억제',
    abbreviation: 'NMS',
    definition: 'def',
    usage_example: null,
    origin: 'standard',
    track_id: 'F',
    status: 'reviewed',
  });
  bank.saveTerm({
    id: 'e-nms-kill',
    term_en: 'NMS kill error',
    term_ko: null,
    abbreviation: null,
    definition: 'def',
    usage_example: null,
    origin: 'project',
    track_id: 'H',
    status: 'reviewed',
  });
  return bank;
}

function mcq(id: string): Question {
  const q = emptyQuestion(id, 'F.nms', 'mcq');
  q.stem = 'NMS의 역할은?';
  q.answer_choice = 2;
  q.choices = [1, 2, 3, 4, 5].map((idx) => ({ idx, text: `보기${idx}`, rationale: idx === 2 ? null : `이유${idx}` }));
  q.explanation_short = '짧은 해설';
  q.term_ids = ['nms'];
  q.source_ref = 'src';
  return q;
}

function short(id: string): Question {
  const q = emptyQuestion(id, 'F.nms', 'short');
  q.stem = '겹치는 박스 중 하나만 남기는 후처리?';
  q.answers = [
    { lang: 'en', text: 'Non-Maximum Suppression', is_primary: true },
    { lang: 'en', text: 'NMS', is_primary: false },
    { lang: 'ko', text: '비최대 억제', is_primary: false },
  ];
  q.explanation_short = '해설';
  q.term_ids = ['nms'];
  return q;
}

describe('grading', () => {
  it('normalizes', () => {
    expect(normalize(' Non-Maximum  Suppression ')).toBe('nonmaximumsuppression');
    expect(normalize('비최대 억제')).toBe('비최대억제');
  });
  it('accepts en or ko, parentheses variants, near miss', () => {
    const acc = [{ text: 'Intersection over Union (IoU)' }, { text: '교집합 대 합집합 비' }];
    expect(gradeShort('iou', acc).correct).toBe(true);
    expect(gradeShort('intersection over union', acc).correct).toBe(true);
    expect(gradeShort('교집합대합집합비', acc).correct).toBe(true);
    expect(gradeShort('intersection ovr union', acc)).toMatchObject({ correct: false, nearMiss: true });
    expect(gradeShort('iuo', acc)).toMatchObject({ correct: false, nearMiss: false }); // 짧은 약어는 오타 판정 안 함
    expect(gradeShort('', acc).correct).toBe(false);
  });
});

describe('bank', () => {
  it('creates, versions, detects breaking change, logs history', () => {
    const bank = seedBank();
    const q = bank.saveQuestion(mcq('F-NMS-001'));
    expect(q.version).toBe(1);
    const q2 = bank.saveQuestion({ ...q, stem: '바뀐 본문' });
    expect(q2.version).toBe(2);
    expect(q2.breaking_version).toBe(1);
    const q3 = bank.saveQuestion({
      ...q2,
      answer_choice: 3,
      choices: q2.choices.map((c) => ({ ...c, rationale: c.idx === 3 ? null : (c.rationale ?? '이유2') })),
    });
    expect(q3.version).toBe(3);
    expect(q3.breaking_version).toBe(3);
    expect(bank.saveQuestion(q3).version).toBe(3); // 변경 없으면 버전 유지
    expect(bank.history('question', 'F-NMS-001').map((h) => h.action)).toEqual(['update', 'update', 'create']);
  });

  it('review gate, delete/retire rules, restore', () => {
    const bank = seedBank();
    const bad = mcq('F-NMS-002');
    bad.choices[0].rationale = '';
    bank.saveQuestion(bad);
    expect(() => bank.setStatus('F-NMS-002', 'reviewed', '수호')).toThrow(/틀린 이유/);
    bank.saveQuestion(mcq('F-NMS-002'));
    expect(() => bank.setStatus('F-NMS-002', 'reviewed')).toThrow(/검수자/);
    const r = bank.setStatus('F-NMS-002', 'reviewed', '수호');
    expect(r.status).toBe('reviewed');
    expect(r.reviewed_by).toBe('수호');
    expect(() => bank.deleteQuestion('F-NMS-002')).toThrow(/삭제할 수 없습니다/);
    // 검수 완료 문항을 오류 있게 수정하면 거부
    expect(() => bank.saveQuestion({ ...r, stem: '' })).toThrow(/검증 오류/);
    // 정상 수정은 허용
    expect(bank.saveQuestion({ ...r, stem: '수정' }).status).toBe('reviewed');
    bank.setStatus('F-NMS-002', 'retired');
    expect(() => bank.setStatus('F-NMS-002', 'draft')).toThrow();
    expect(() => bank.saveQuestion({ ...r, stem: 'x' })).toThrow(/출제 중단/);
    // 초안 삭제 가능
    bank.saveQuestion(mcq('F-NMS-003'));
    bank.deleteQuestion('F-NMS-003');
    expect(bank.getQuestion('F-NMS-003')).toBeUndefined();
    // 복원
    const q = bank.saveQuestion(mcq('F-NMS-004'));
    bank.saveQuestion({ ...q, stem: '두 번째' });
    const firstLog = bank.history('question', 'F-NMS-004').find((h) => h.action === 'create')!;
    expect(bank.restoreQuestion(firstLog.id).stem).toBe('NMS의 역할은?');
  });

  it('search, filter, next id, badges, term guard', () => {
    const bank = seedBank();
    bank.saveQuestion(mcq('F-NMS-001'));
    const s = short('F-NMS-002');
    s.term_ids = ['nms', 'e-nms-kill'];
    bank.saveQuestion(s);
    expect(bank.nextQuestionId('F.nms')).toBe('F-NMS-003');
    expect(bank.listQuestions({ text: '비최대' }).map((q) => q.id)).toEqual(['F-NMS-001', 'F-NMS-002']);
    expect(bank.listQuestions({ text: '보기3' }).map((q) => q.id)).toEqual(['F-NMS-001']);
    expect(bank.listQuestions({ origin: 'project' }).map((q) => q.id)).toEqual(['F-NMS-002']);
    expect(bank.listQuestions({ type: 'short' }).length).toBe(1);
    expect(() => bank.deleteTerm('nms')).toThrow(/연결된 용어/);
    expect(validateQuestion(bank.getQuestion('F-NMS-002')!).filter((i) => i.level === 'error')).toEqual([]);
  });

  it('survives export/reopen with foreign keys still enforced', () => {
    const bank = seedBank();
    bank.saveQuestion(mcq('F-NMS-001'));
    const bytes = bank.export();
    const reopened = Bank.open(SQL, bytes);
    expect(reopened.getQuestion('F-NMS-001')?.choices.length).toBe(5);
    expect(() => bank.saveQuestion({ ...mcq('F-NMS-009'), chapter_id: 'X.none' })).toThrow(/장/);
  });
});

describe('exchange', () => {
  it('imports LLM-style JSON with reviewed status and round-trips', () => {
    const bank = seedBank();
    const text = JSON.stringify({
      format: 'visiondrill-bank-exchange',
      format_version: 1,
      chapters: [{ id: 'B.vif', track: 'B', code: 'VIF', name_ko: '다중공선성' }],
      terms: [{ id: 'vif', term_en: 'Variance Inflation Factor', abbreviation: 'VIF', definition: 'd' }],
      questions: [
        {
          id: 'B-VIF-001',
          type: 'mcq',
          chapter: 'B.vif',
          stem: 'VIF?',
          answer: 1,
          choices: [
            { text: 'a' },
            { text: 'b', rationale: 'x' },
            { text: 'c', rationale: 'x' },
            { text: 'd', rationale: 'x' },
            { text: 'e', rationale: 'x' },
          ],
          explanation_short: 'e',
          terms: ['vif'],
          status: 'reviewed',
        },
        {
          id: 'B-VIF-002',
          type: 'short',
          chapter: 'B.vif',
          stem: 'VIF 풀네임?',
          accepted_answers: { en: ['Variance Inflation Factor', 'VIF'], ko: ['분산팽창계수'] },
          explanation_short: 'e',
          terms: ['vif'],
          status: 'reviewed',
        },
        {
          id: 'B-VIF-003',
          type: 'mcq',
          chapter: 'B.vif',
          stem: 'bad',
          answer: 1,
          choices: [{ text: 'a' }],
          explanation_short: 'e',
          status: 'reviewed',
        },
        { id: 'B-VIF-004', type: 'mcq', chapter: 'B.none', stem: 'x', answer: 1, choices: [{ text: 'a' }], explanation_short: 'e' },
      ],
    });
    const { file, errors } = parseExchange(text);
    expect(errors).toEqual([]);
    const res = importExchange(bank, file!, { reviewer: '수호' });
    expect(res).toMatchObject({ chapters: 1, terms: 1, created: 3, failed: 1, reviewed: 2 });
    expect(bank.getQuestion('B-VIF-003')?.status).toBe('draft');
    expect(bank.getQuestion('B-VIF-002')?.answers.find((a) => a.is_primary)?.text).toBe('Variance Inflation Factor');

    const out = exportExchange(bank);
    const bank2 = seedBank();
    const res2 = importExchange(bank2, parseExchange(JSON.stringify(out)).file!, { reviewer: '수호' });
    expect(res2.failed).toBe(0);
    expect(bank2.getQuestion('B-VIF-002')).toMatchObject({ status: 'reviewed', stem: 'VIF 풀네임?' });
    // 재가져오기: 변경 없음
    const res3 = importExchange(bank2, parseExchange(JSON.stringify(out)).file!, { reviewer: '수호' });
    expect(res3).toMatchObject({ created: 0, updated: 0, unchanged: 3 });
  });

  it('accepts bare question arrays and reports structural errors', () => {
    expect(parseExchange('[{"id":"x"}]').errors.length).toBeGreaterThan(0);
    expect(parseExchange('{oops').errors[0]).toMatch(/JSON/);
  });
});

describe('pack + learner', () => {
  async function makePack(bank: Bank, ver: string) {
    return (await exportPack(bank, SQL, ver)).bytes;
  }

  it('exports only reviewed questions and learner flow works', async () => {
    const bank = seedBank();
    for (const id of ['F-NMS-001', 'F-NMS-002', 'F-NMS-003']) {
      bank.saveQuestion(mcq(id));
      bank.setStatus(id, 'reviewed', '수호');
    }
    bank.saveQuestion(short('F-NMS-004'));
    bank.setStatus('F-NMS-004', 'reviewed', '수호');
    bank.saveQuestion(mcq('F-NMS-005')); // draft → 팩 제외
    const pack = await makePack(bank, '2026.10.01');
    await expect(makePack(bank, '2026.10.01')).rejects.toThrow(/이미/);
    expect(bank.stats().exports.length).toBe(1);

    const L = Learner.create(SQL);
    L.saveSettings({ unlockAll: true, shuffleChoices: false });
    const imp = L.importPack(SQL, pack);
    expect(imp).toMatchObject({ added: 4, reset: 0 });
    expect(L.packInfo()?.pack_version).toBe('2026.10.01');

    const items = L.buildSession('daily');
    expect(items.length).toBe(4);
    const started = new Date(Date.now() - 60000).toISOString();
    const responses: any = {};
    for (const it of items) {
      responses[it.question.id] =
        it.question.id === 'F-NMS-004'
          ? { kind: 'text', text: 'non maximum supression' } // 오타 → near miss
          : it.question.id === 'F-NMS-001'
            ? { kind: 'choice', idx: 1 } // 오답
            : { kind: 'choice', idx: 2 };
    }
    const res = L.submitSession('daily', items, responses, { 'F-NMS-003': true }, started);
    expect(res.correct).toBe(2);
    const nm = res.items.find((r) => r.item.question.id === 'F-NMS-004')!;
    expect(nm.grade.nearMiss).toBe(true);
    expect(
      L.wrongNotes()
        .map((w) => w.question.id)
        .sort(),
    ).toEqual(['F-NMS-001', 'F-NMS-003', 'F-NMS-004']);
    L.acceptNearMiss(nm.attemptId);
    expect(
      L.wrongNotes()
        .map((w) => w.question.id)
        .sort(),
    ).toEqual(['F-NMS-001', 'F-NMS-003']);
    expect(L.db.value(`SELECT correct FROM sessions WHERE id=?`, [res.sessionId])).toBe(3);
    expect(L.db.value(`SELECT box FROM progress WHERE question_id='F-NMS-004'`)).toBe(2);
    expect(L.stats()).toMatchObject({ sessions: 1, streakDays: 1, wrongNotes: 2 });

    // 오답 문항은 바로 다음 세션에 재출제
    const next = L.buildSession('daily').map((i) => i.question.id);
    expect(next).toContain('F-NMS-001');
    expect(
      L.buildSession('wrong_note')
        .map((i) => i.question.id)
        .sort(),
    ).toEqual(['F-NMS-001', 'F-NMS-003']);

    // 정답 변경(breaking) → 새 팩 가져오면 해당 문항 일정 초기화, 기록은 유지
    const q2 = bank.getQuestion('F-NMS-002')!;
    bank.saveQuestion({
      ...q2,
      answer_choice: 3,
      choices: q2.choices.map((c) => ({ ...c, rationale: c.idx === 3 ? null : (c.rationale ?? '이유') })),
    });
    const pack2 = await makePack(bank, '2026.10.02');
    const imp2 = L.importPack(SQL, pack2);
    expect(imp2).toMatchObject({ added: 0, reset: 1 });
    expect(L.db.value(`SELECT box FROM progress WHERE question_id='F-NMS-002'`)).toBe(1);
    expect(L.db.value(`SELECT COUNT(*) FROM attempts`)).toBe(4);

    // 학습 기록 파일 저장/열기
    const reopened = Learner.open(SQL, L.export());
    expect(reopened.stats().sessions).toBe(1);
  });

  it('locks tracks until prereq mastery', async () => {
    const bank = seedBank();
    bank.saveChapter({ id: 'E.cnn', track_id: 'E', code: 'CNN', name_ko: 'CNN', name_en: null, sort_order: 1, stage: 1, source_ref: null });
    const e = mcq('E-CNN-001');
    e.chapter_id = 'E.cnn';
    bank.saveQuestion(e);
    bank.setStatus('E-CNN-001', 'reviewed', '수호');
    bank.saveQuestion(mcq('F-NMS-001'));
    bank.setStatus('F-NMS-001', 'reviewed', '수호');
    const L = Learner.create(SQL);
    L.importPack(SQL, await makePack(bank, '1'));
    const st = L.trackStatus();
    expect(st.find((t) => t.id === 'E')?.unlocked).toBe(true); // 선수 D는 문항이 없으므로 통과
    expect(st.find((t) => t.id === 'F')?.unlocked).toBe(false); // 선수 E 숙련도 0
    expect(L.buildSession('daily').map((i) => i.question.id)).toEqual(['E-CNN-001']);
  });
  it('bundled pack: install, upgrade from bundled, keep user-imported pack', async () => {
    const bank = seedBank();
    bank.saveQuestion(mcq('F-NMS-001'));
    bank.setStatus('F-NMS-001', 'reviewed', '수호');
    const sample = await makePack(bank, 'sample-0.1.0');
    bank.saveQuestion(mcq('F-NMS-002'));
    bank.setStatus('F-NMS-002', 'reviewed', '수호');
    const v1 = await makePack(bank, '0.1.1-draft');
    bank.saveQuestion(mcq('F-NMS-003'));
    bank.setStatus('F-NMS-003', 'reviewed', '수호');
    const mine = await makePack(bank, '2026.10.05');
    const v2 = await makePack(bank, '0.1.2-draft');

    // 새 설치
    const A = Learner.create(SQL);
    expect(A.applyBundledPack(SQL, v1)?.added).toBe(2);
    expect(A.applyBundledPack(SQL, v1)).toBeNull(); // 같은 버전이면 그대로
    expect(A.applyBundledPack(SQL, v2)?.added).toBe(1); // 앱 업데이트
    expect(A.packInfo()?.pack_version).toBe('0.1.2-draft');

    // 0.1.0 샘플 팩 설치본 → 새 기본 팩으로 교체
    const B = Learner.create(SQL);
    B.importPack(SQL, sample);
    expect(B.applyBundledPack(SQL, v1)?.added).toBe(1);
    expect(B.packInfo()?.pack_version).toBe('0.1.1-draft');

    // 직접 가져온 팩은 유지
    const C = Learner.create(SQL);
    C.applyBundledPack(SQL, v1);
    C.importPack(SQL, mine);
    expect(C.applyBundledPack(SQL, v2)).toBeNull();
    expect(C.packInfo()?.pack_version).toBe('2026.10.05');
  });
});
