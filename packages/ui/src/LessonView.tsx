import 'katex/dist/katex.min.css';
import type { RenderedLesson } from './lesson';

/** 렌더된 자습서 본문. html은 저장소 안의 Markdown에서만 만들어짐(원문 HTML 태그는 막혀 있음) */
export function LessonView({ lesson }: { lesson: RenderedLesson }) {
  return <article className="lesson" dangerouslySetInnerHTML={{ __html: lesson.html }} />;
}
