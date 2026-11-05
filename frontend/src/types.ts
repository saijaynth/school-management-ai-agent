export type SourceType = "database" | "manual" | "hybrid" | "clarification" | "error";

export interface SqlQueryRecord {
  sql: string;
  row_count: number;
  columns: string[];
  rows: Record<string, unknown>[];
  execution_time_ms: number;
}

export interface ManualReference {
  section: string;
  summary: string;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  source_type: SourceType;
  sql_queries: SqlQueryRecord[];
  manual_references: ManualReference[];
  suggested_followups: string[];
  llm_ready: boolean;
  thinking?: string;
  elapsed_ms?: number;
}

export interface HealthResponse {
  status: string;
  database: {
    connected: boolean;
    schema?: string;
    tables?: number | null;
    views?: number | null;
  };
  llm: { ready: boolean; provider: string; model: string; key_configured: boolean };
  counts: { tables: number; views: number; users: number };
}

export interface SchemaSummaryResponse {
  schema_name: string;
  modules: { module: string; tables: string[] }[];
  views: { name: string; description: string }[];
  entities: { kind: string; label: string; detail: string }[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  anytime?: {
    source_type?: SourceType;
    sql_queries?: SqlQueryRecord[];
    manual_references?: ManualReference[];
    suggested_followups?: string[];
  };
  failed?: boolean;
}

export const STARTER_PROMPTS: { group: string; items: string[] }[] = [
  {
    group: "Attendance & Leaves",
    items: [
      "Student attendance report - who is below 75%?",
      "Which leave requests are pending approval?",
      "Faculty attendance on November 2, 2026",
      "Show all student attendance percentages",
    ],
  },
  {
    group: "Classes & Timetable",
    items: [
      "What is Grade 10-A's Monday timetable?",
      "Who teaches Science?",
      "What is Sara Thomas's room and schedule on Monday?",
      "How many students are in Grade 9-A?",
    ],
  },
  {
    group: "Exams & Marks",
    items: [
      "Compare Kiran Patel and Divya Nair's marks",
      "Which subject has Kiran scored the lowest?",
      "Grade 10 Mid Term rank list with totals and grades",
      "Pending homework and assignment submissions",
    ],
  },
  {
    group: "Campus Guidelines",
    items: [
      "According to the manual, how do I approve a leave request?",
      "How can a student apply for leave?",
      "Why shouldn't records be deleted from the system?",
      "What is the year-end student promotion procedure?",
    ],
  },
];
