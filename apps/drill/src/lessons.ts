// 앱에 내장하는 자습서: content/lessons/*.md + fig/*. 문제 팩과 별개로 앱 빌드에 들어감
import { figureName, parseLesson, renderLesson, type FigureSource, type Lesson, type RenderedLesson } from '@visiondrill/ui';
import partsIndex from '../../../content/lessons/parts.json';

const raws = import.meta.glob('../../../content/lessons/*.md', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;
const svgs = import.meta.glob('../../../content/lessons/fig/*.svg', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;
const imgs = import.meta.glob('../../../content/lessons/fig/*.{png,jpg,jpeg,webp}', {
  query: '?url',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const base = (p: string) => p.slice(p.lastIndexOf('/') + 1);
const svgByName = new Map(Object.entries(svgs).map(([p, v]) => [base(p), v]));
const imgByName = new Map(Object.entries(imgs).map(([p, v]) => [base(p), v]));

export interface Part {
  part: number;
  title: string;
  subtitle?: string;
}
export const PARTS: Part[] = partsIndex.parts;

export const LESSONS: Lesson[] = Object.values(raws)
  .map((raw) => {
    try {
      return parseLesson(raw);
    } catch (e) {
      console.error('자습서 머리말 오류', e);
      return null;
    }
  })
  .filter((l): l is Lesson => !!l)
  .sort((a, b) => a.meta.part - b.meta.part || a.meta.id.localeCompare(b.meta.id, 'en', { numeric: true }));

function resolveFigure(src: string): FigureSource | undefined {
  const name = figureName(src);
  const svg = svgByName.get(name);
  if (svg) return { kind: 'svg', svg };
  const url = imgByName.get(name);
  return url ? { kind: 'url', url } : undefined;
}

const cache = new Map<string, RenderedLesson>();
export function renderedLesson(l: Lesson): RenderedLesson {
  let r = cache.get(l.meta.id);
  if (!r) {
    r = renderLesson(l.body, resolveFigure);
    if (r.issues.length) console.warn(`자습서 ${l.meta.id} 렌더링 문제`, r.issues);
    cache.set(l.meta.id, r);
  }
  return r;
}
