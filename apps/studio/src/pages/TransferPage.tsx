import { exportExchange, exportPack, importExchange, parseExchange, type ImportResult } from '@visiondrill/core';
import { copyText, pickAndReadFile, saveAs } from '@visiondrill/platform';
import { useMemo, useState } from 'react';
import { useBank, useStudio } from '../store';

const today = () => {
  const d = new Date();
  return `${d.getFullYear()}.${String(d.getMonth() + 1).padStart(2, '0')}.${String(d.getDate()).padStart(2, '0')}`;
};

export const LLM_PROMPT = `아래 형식의 JSON으로 VisionDrill 문항 초안을 만들어 주세요.

[조건]
- 대상: 컴퓨터 비전 분야 신입·비전공자. 목표는 전문용어로 된 업무 지시를 이해하는 것.
- 유형: mcq(5지선다), short(단답형), scenario(지시문 해석형, 실무 지시 한 줄 + "가장 적절한 조치는?")
- cognitive_level: 1 용어 인지 / 2 개념 구분 / 3 지시 해석 (scenario는 3)
- 객관식은 보기 정확히 5개, answer는 정답 번호(1~5), 오답 보기마다 rationale(왜 틀렸는지) 필수, 정답 보기의 rationale은 null
- 오답 보기는 신입이 실제로 헷갈릴 만한 인접 개념으로 구성. "모두 정답" 류 보기 금지(보기 순서가 섞임)
- 단답형은 accepted_answers에 영문(en)·한글(ko) 인정 답을 모두 넣기. 대소문자·공백·하이픈 차이는 자동 무시되므로 그런 변형은 넣지 않기
- explanation_short: 오답 시 보여줄 2~3문장, explanation_full: 더 깊은 설명(선택)
- terms: 연결할 용어 ID 목록(없으면 terms 배열에 새 용어도 함께 정의)
- 업계 표준이 아닌 자체 정의 지표는 용어의 origin을 "project"로
- status는 "draft"

[형식]
{
  "format": "visiondrill-bank-exchange",
  "format_version": 1,
  "terms": [
    { "id": "nms", "term_en": "Non-Maximum Suppression", "term_ko": "비최대 억제", "abbreviation": "NMS",
      "definition": "한 줄 정의", "usage_example": "실무에서 이렇게 말한다", "origin": "standard", "track": "F" }
  ],
  "questions": [
    { "id": "F-NMS-010", "type": "mcq", "chapter": "F.nms", "cognitive_level": 2, "difficulty": 2,
      "stem": "문제", "answer": 3,
      "choices": [ { "text": "보기1", "rationale": "틀린 이유" }, { "text": "보기2", "rationale": "틀린 이유" },
                   { "text": "보기3(정답)", "rationale": null }, { "text": "보기4", "rationale": "틀린 이유" },
                   { "text": "보기5", "rationale": "틀린 이유" } ],
      "explanation_short": "…", "explanation_full": "…", "tags": ["detection"], "terms": ["nms"],
      "source_ref": "출처", "status": "draft" },
    { "id": "F-NMS-011", "type": "short", "chapter": "F.nms", "cognitive_level": 1, "stem": "…",
      "accepted_answers": { "en": ["Non-Maximum Suppression", "NMS"], "ko": ["비최대 억제"] },
      "primary": "Non-Maximum Suppression", "explanation_short": "…", "terms": ["nms"], "status": "draft" }
  ]
}`;

export function TransferPage() {
  const bank = useBank();
  const { SQL, rev, mutate, mutateAsync, reviewer, toast } = useStudio();
  const [text, setText] = useState('');
  const [skipExisting, setSkipExisting] = useState(false);
  const [result, setResult] = useState<ImportResult | null>(null);
  const [parseErrors, setParseErrors] = useState<string[]>([]);
  const [packVer, setPackVer] = useState(today());
  const [packNote, setPackNote] = useState('');
  const reviewedCount = useMemo(() => bank.db.value<number>(`SELECT COUNT(*) FROM questions WHERE status='reviewed'`) ?? 0, [bank, rev]);
  const exports = useMemo(() => bank.stats().exports, [bank, rev]);

  const runImport = (json: string) => {
    setResult(null);
    const { file, errors } = parseExchange(json);
    setParseErrors(errors);
    if (!file) return;
    const r = mutate((b) => importExchange(b, file, { reviewer, skipExisting }));
    if (r) {
      setResult(r);
      toast(`가져오기 완료: 새 문항 ${r.created}, 갱신 ${r.updated}, 실패 ${r.failed}`, r.failed > 0);
    }
  };

  const exportJson = async (onlyReviewed: boolean) => {
    const data = exportExchange(bank, onlyReviewed ? ['reviewed'] : undefined);
    const name = `visiondrill-${onlyReviewed ? 'reviewed' : 'all'}-${today().replace(/\./g, '')}.json`;
    const p = await saveAs(name, JSON.stringify(data, null, 2), [{ name: 'JSON', extensions: ['json'] }]);
    if (p) toast(`문항 ${data.questions?.length ?? 0}개를 내보냈습니다`);
  };

  const buildPack = async () => {
    const out = await mutateAsync((b) => exportPack(b, SQL, packVer.trim(), packNote || undefined));
    if (!out) return;
    const p = await saveAs(`visiondrill-${packVer.trim()}.vdpack`, out.bytes, [{ name: 'VisionDrill 문제 팩', extensions: ['vdpack'] }]);
    if (p) toast(`팩을 만들었습니다: 문항 ${out.manifest.question_count}개, 용어 ${out.manifest.term_count}개`);
  };

  return (
    <div className="page">
      <div className="page-head">
        <h1>가져오기 · 내보내기</h1>
      </div>

      <div className="section">
        <h3>문제 팩 만들기 (학습 앱 배포용 .vdpack)</h3>
        <p className="vd-muted" style={{ marginTop: 0, fontSize: '0.9rem' }}>
          검수 완료 문항 <strong>{reviewedCount}</strong>개와 연결된 용어·이미지만 담긴 읽기 전용 파일을 만듭니다. 이 파일을 GitHub
          Releases에 올리거나 신입에게 전달하세요.
        </p>
        <div className="field-row" style={{ alignItems: 'end' }}>
          <label className="field" style={{ maxWidth: 180 }}>
            <span>팩 버전</span>
            <input className="vd-input" value={packVer} onChange={(e) => setPackVer(e.target.value)} />
          </label>
          <label className="field">
            <span>메모 (선택)</span>
            <input className="vd-input" value={packNote} onChange={(e) => setPackNote(e.target.value)} />
          </label>
          <div className="field" style={{ flex: 0 }}>
            <button className="vd-btn vd-btn-primary" onClick={buildPack} disabled={!reviewedCount}>
              팩 만들기
            </button>
          </div>
        </div>
        {exports.length > 0 && (
          <table className="table" style={{ marginTop: 8 }}>
            <thead>
              <tr>
                <th>버전</th>
                <th>문항</th>
                <th>용어</th>
                <th>만든 시각</th>
                <th>메모</th>
              </tr>
            </thead>
            <tbody>
              {exports.map((e: any) => (
                <tr key={e.id}>
                  <td className="qitem-id">{e.pack_version}</td>
                  <td>{e.question_count}</td>
                  <td>{e.term_count}</td>
                  <td>{new Date(e.exported_at).toLocaleString('ko-KR')}</td>
                  <td className="vd-muted">{e.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="section">
        <h3>JSON 가져오기 (LLM 초안·백업 파일)</h3>
        <p className="vd-muted" style={{ marginTop: 0, fontSize: '0.9rem' }}>
          같은 ID가 있으면 내용을 갱신합니다(이전 내용은 변경 이력에 남음). 파일에 <code>"status": "reviewed"</code>가 있으면 현재 검수자 (
          {reviewer || '미설정'}) 이름으로 검수 완료 처리하고, 검증 오류가 있으면 초안으로 둡니다.
        </p>
        <textarea
          className="vd-textarea"
          rows={8}
          style={{ fontFamily: 'var(--mono)', fontSize: '0.82rem' }}
          placeholder="여기에 JSON을 붙여넣거나 파일을 고르세요. 문항 배열만 붙여넣어도 됩니다."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 8, flexWrap: 'wrap' }}>
          <button className="vd-btn vd-btn-primary" disabled={!text.trim()} onClick={() => runImport(text)}>
            붙여넣은 JSON 가져오기
          </button>
          <button
            className="vd-btn"
            onClick={async () => {
              const f = await pickAndReadFile('JSON 가져오기', [{ name: 'JSON', extensions: ['json'] }]);
              if (f) runImport(new TextDecoder().decode(f.bytes));
            }}
          >
            파일에서 가져오기
          </button>
          <label style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: '0.9rem' }}>
            <input type="checkbox" checked={skipExisting} onChange={(e) => setSkipExisting(e.target.checked)} />
            이미 있는 문항은 건너뛰기
          </label>
        </div>
        {parseErrors.length > 0 && (
          <ul className="msg-list">
            {parseErrors.map((e, i) => (
              <li key={i} className="is-error">
                ⛔ {e}
              </li>
            ))}
          </ul>
        )}
        {result && (
          <div style={{ marginTop: 10 }}>
            <div className="field-row" style={{ fontSize: '0.9rem' }}>
              <span>장 {result.chapters}</span>
              <span>용어 {result.terms}</span>
              <span>새 문항 {result.created}</span>
              <span>갱신 {result.updated}</span>
              <span>변경 없음 {result.unchanged}</span>
              <span>검수 완료 처리 {result.reviewed}</span>
              <span style={{ color: result.failed ? 'var(--bad)' : undefined }}>실패 {result.failed}</span>
            </div>
            {result.messages.length > 0 && (
              <ul className="msg-list">
                {result.messages.map((m, i) => (
                  <li key={i} className={`is-${m.level}`}>
                    <span className="qitem-id">{m.id}</span> {m.message}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      <div className="section">
        <h3>JSON 내보내기</h3>
        <p className="vd-muted" style={{ marginTop: 0, fontSize: '0.9rem' }}>
          백업하거나 LLM에게 기존 문항을 보여 주고 검토·확장을 맡길 때 씁니다. 다시 가져오면 ID 기준으로 갱신됩니다.
        </p>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="vd-btn" onClick={() => exportJson(false)}>
            전체 내보내기
          </button>
          <button className="vd-btn" onClick={() => exportJson(true)}>
            검수 완료만 내보내기
          </button>
        </div>
      </div>

      <div className="section">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <h3 style={{ margin: 0 }}>LLM 초안 요청 프롬프트</h3>
          <button
            className="vd-btn vd-btn-small"
            style={{ marginLeft: 'auto' }}
            onClick={async () =>
              toast((await copyText(LLM_PROMPT)) ? '복사했습니다' : '복사하지 못했습니다. 직접 선택해 복사하세요', false)
            }
          >
            복사
          </button>
        </div>
        <p className="vd-muted" style={{ fontSize: '0.9rem' }}>
          이 프롬프트 뒤에 "B 트랙 잔차 진단 장에서 5문항, 원문: …"처럼 요청을 붙여 LLM에 주고, 받은 JSON을 위에 붙여넣으세요.
        </p>
        <pre className="code">{LLM_PROMPT}</pre>
      </div>
    </div>
  );
}
