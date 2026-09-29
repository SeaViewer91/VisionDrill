declare module 'markdown-it-texmath' {
  import type MarkdownIt from 'markdown-it';
  const texmath: MarkdownIt.PluginWithOptions<Record<string, unknown>>;
  export default texmath;
}
