export type QType = 'mcq' | 'short' | 'scenario';
export type Status = 'draft' | 'reviewed' | 'retired';
export type Lang = 'en' | 'ko';
export type TermOrigin = 'standard' | 'project';

export const TYPE_LABEL: Record<QType, string> = {
  mcq: '5지선다',
  short: '단답형',
  scenario: '지시문 해석형',
};
export const STATUS_LABEL: Record<Status, string> = {
  draft: '초안',
  reviewed: '검수 완료',
  retired: '출제 중단',
};
export const LEVEL_LABEL: Record<number, string> = {
  1: '용어 인지',
  2: '개념 구분',
  3: '지시 해석',
};

export interface Track {
  id: string;
  name_ko: string;
  name_en: string;
  description: string | null;
  sort_order: number;
}

export interface TrackPrereq {
  track_id: string;
  prereq_track_id: string;
  min_mastery: number;
}

export interface Chapter {
  id: string;
  track_id: string;
  code: string;
  name_ko: string;
  name_en: string | null;
  sort_order: number;
  stage: number;
  source_ref: string | null;
}

export interface Term {
  id: string;
  term_en: string;
  term_ko: string | null;
  abbreviation: string | null;
  definition: string;
  usage_example: string | null;
  origin: TermOrigin;
  track_id: string | null;
  status: Status;
  created_at?: string;
  updated_at?: string;
}

export interface Choice {
  idx: number; // 1..5
  text: string;
  rationale: string | null;
}

export interface Answer {
  lang: Lang;
  text: string;
  is_primary: boolean;
}

export interface Question {
  id: string;
  type: QType;
  chapter_id: string;
  cognitive_level: number;
  difficulty: number;
  stem: string;
  image_asset_id: number | null;
  answer_choice: number | null;
  explanation_short: string;
  explanation_full: string | null;
  source_ref: string | null;
  status: Status;
  version: number;
  breaking_version: number;
  reviewed_by: string | null;
  reviewed_at: string | null;
  created_at?: string;
  updated_at?: string;
  choices: Choice[];
  answers: Answer[];
  tags: string[];
  term_ids: string[];
}

export interface QuestionSummary {
  id: string;
  type: QType;
  chapter_id: string;
  track_id: string;
  cognitive_level: number;
  difficulty: number;
  stem: string;
  status: Status;
  version: number;
  updated_at: string;
  has_project_term: number;
}

export interface QuestionFilter {
  track?: string;
  chapter?: string;
  type?: QType;
  status?: Status;
  level?: number;
  tag?: string;
  origin?: TermOrigin;
  text?: string;
}

export interface ChangeLogEntry {
  id: number;
  entity_type: 'question' | 'term';
  entity_id: string;
  version: number | null;
  action: 'create' | 'update' | 'status' | 'retire' | 'delete';
  snapshot_json: string;
  note: string | null;
  changed_at: string;
}

export function emptyQuestion(id: string, chapter_id: string, type: QType = 'mcq'): Question {
  return {
    id,
    type,
    chapter_id,
    cognitive_level: 1,
    difficulty: 2,
    stem: '',
    image_asset_id: null,
    answer_choice: type === 'short' ? null : 1,
    explanation_short: '',
    explanation_full: null,
    source_ref: null,
    status: 'draft',
    version: 1,
    breaking_version: 1,
    reviewed_by: null,
    reviewed_at: null,
    choices: type === 'short' ? [] : [1, 2, 3, 4, 5].map((idx) => ({ idx, text: '', rationale: '' })),
    answers: type === 'short' ? [{ lang: 'en', text: '', is_primary: true }] : [],
    tags: [],
    term_ids: [],
  };
}
