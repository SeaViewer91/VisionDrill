/// <reference path="./texmath.d.ts" />
// 자습서(Markdown) → HTML. 학습 앱 화면과 tools/lint-lessons.ts가 같은 코드를 씀
// - 수식: $...$ (인라인), $$...$$ (블록) → KaTeX
// - 그림: ![캡션](fig/파일.svg) → SVG는 본문에 직접 넣음(다크 모드 색 적용), 그 밖의 이미지는 <img>
import katex from 'katex';
import MarkdownIt from 'markdown-it';
import cjkFriendly from 'markdown-it-cjk-friendly';
import texmath from 'markdown-it-texmath';

export interface LessonMeta {
  id: string;
  part: number;
  title: string;
  chapters: string[];
  version: number;
  updated: string;
  status: 'draft' | 'reviewed';
}

export interface Lesson {
  meta: LessonMeta;
  body: string;
}

export type FigureSource = { kind: 'svg'; svg: string } | { kind: 'url'; url: string };

export interface RenderIssue {
  kind: 'math' | 'figure';
  message: string;
}

export interface RenderedLesson {
  html: string;
  headings: { id: string; text: string; level: number }[];
  issues: RenderIssue[];
}

/** 머리말: --- 로 감싼 key: value 줄. 값은 문자열, 숫자, [a, b] 목록만 허용 */
export function parseLesson(raw: string): Lesson {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(raw);
  if (!m) throw new Error('머리말(---)이 없습니다');
  const meta: Record<string, unknown> = {};
  for (const line of m[1].split(/\r?\n/)) {
    if (!line.trim() || line.trim().startsWith('#')) continue;
    const kv = /^([a-z_]+):\s*(.*)$/.exec(line);
    if (!kv) throw new Error(`머리말 형식 오류: ${line}`);
    const v = kv[2].trim();
    if (/^\[.*\]$/.test(v))
      meta[kv[1]] = v
        .slice(1, -1)
        .split(',')
        .map((s) => s.trim().replace(/^["']|["']$/g, ''))
        .filter(Boolean);
    else if (/^-?\d+(\.\d+)?$/.test(v) && kv[1] !== 'id') meta[kv[1]] = Number(v);
    else meta[kv[1]] = v.replace(/^["']|["']$/g, '');
  }
  for (const k of ['id', 'part', 'title', 'chapters', 'version', 'updated', 'status'])
    if (meta[k] === undefined) throw new Error(`머리말에 ${k}가 없습니다`);
  return { meta: meta as unknown as LessonMeta, body: raw.slice(m[0].length) };
}

const slug = (s: string) =>
  s
    .trim()
    .toLowerCase()
    .replace(/<[^>]+>/g, '')
    .replace(/[^\p{L}\p{N}]+/gu, '-')
    .replace(/^-|-$/g, '');

function escapeHtml(s: string) {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
}

/** 그림 SVG에서 XML 선언·주석 제거, 크기는 viewBox 기준으로 반응형 */
function prepareSvg(svg: string) {
  return svg
    .replace(/<\?xml[^>]*\?>/g, '')
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<svg\b([^>]*)>/, (_all, attrs: string) => {
      const cleaned = attrs.replace(/\s(width|height)="[^"]*"/g, '');
      return `<svg${cleaned} width="100%" preserveAspectRatio="xMidYMid meet">`;
    })
    .trim();
}

export function renderLesson(body: string, resolveFigure: (path: string) => FigureSource | undefined): RenderedLesson {
  const issues: RenderIssue[] = [];
  const engine = {
    renderToString(tex: string, opts: katex.KatexOptions) {
      try {
        return `<span class="vd-math">${katex.renderToString(tex, { ...opts, throwOnError: true, strict: 'error' })}</span><!--/vd-math-->`;
      } catch (e) {
        issues.push({ kind: 'math', message: `${(e as Error).message} ← ${tex}` });
        return `<span class="lesson-math-error">${escapeHtml(tex)}</span>`;
      }
    },
  };
  const md = new MarkdownIt({ html: false, linkify: false, typographer: false });
  // 한글 조사가 바로 붙은 강조(**구역 통계(zonal statistics)**라고)도 굵게 처리되도록
  md.use(cjkFriendly);
  // texmath는 수식 엔진을 전역 하나로 기억하므로 렌더할 때마다 이번 호출의 엔진으로 바꿔 끼움
  (texmath as unknown as { katex: typeof engine }).katex = engine;
  md.use(texmath, { engine, delimiters: 'dollars', katexOptions: { output: 'html' } });

  const headings: RenderedLesson['headings'] = [];
  md.core.ruler.push('heading_ids', (state) => {
    const used = new Set<string>();
    state.tokens.forEach((t, i) => {
      if (t.type !== 'heading_open') return;
      const text = state.tokens[i + 1]?.content ?? '';
      let id = slug(text) || `h-${i}`;
      while (used.has(id)) id += '-';
      used.add(id);
      t.attrSet('id', id);
      headings.push({ id, text, level: Number(t.tag.slice(1)) });
    });
  });

  // 문단 안에 그림 하나만 있으면 <figure>로 바꿈
  md.renderer.rules.image = (tokens, idx) => {
    const t = tokens[idx];
    const src = String(t.attrGet('src') ?? '');
    const caption = t.content;
    const fig = resolveFigure(src);
    if (!fig) {
      issues.push({ kind: 'figure', message: `그림 파일을 찾을 수 없음: ${src}` });
      return `<figure class="lesson-fig lesson-fig-missing"><figcaption>그림 없음: ${escapeHtml(src)}</figcaption></figure>`;
    }
    if (!caption.trim()) issues.push({ kind: 'figure', message: `그림 캡션 없음: ${src}` });
    const inner =
      fig.kind === 'svg'
        ? `<div class="lesson-svg" role="img" aria-label="${escapeHtml(caption)}">${prepareSvg(fig.svg)}</div>`
        : `<img src="${escapeHtml(fig.url)}" alt="${escapeHtml(caption)}" loading="lazy">`;
    return `<figure class="lesson-fig">${inner}<figcaption>${md.renderInline(caption)}</figcaption></figure>`;
  };
  const defaultParagraphOpen = md.renderer.rules.paragraph_open;
  md.renderer.rules.paragraph_open = (tokens, idx, opts, env, self) => {
    const inline = tokens[idx + 1];
    if (inline?.children?.length === 1 && inline.children[0].type === 'image') return '';
    return defaultParagraphOpen ? defaultParagraphOpen(tokens, idx, opts, env, self) : self.renderToken(tokens, idx, opts);
  };
  const defaultParagraphClose = md.renderer.rules.paragraph_close;
  md.renderer.rules.paragraph_close = (tokens, idx, opts, env, self) => {
    const inline = tokens[idx - 1];
    if (inline?.children?.length === 1 && inline.children[0].type === 'image') return '';
    return defaultParagraphClose ? defaultParagraphClose(tokens, idx, opts, env, self) : self.renderToken(tokens, idx, opts);
  };
  // 표는 가로 스크롤 가능한 틀로 감쌈
  md.renderer.rules.table_open = () => '<div class="lesson-table"><table>';
  md.renderer.rules.table_close = () => '</table></div>';

  const html = md.render(body);
  return { html, headings, issues };
}

/** 그림 파일 경로(fig/xx.svg 또는 ./fig/xx.svg)에서 파일 이름만 */
export function figureName(src: string) {
  return src.replace(/^\.?\/?/, '').replace(/^fig\//, '');
}
