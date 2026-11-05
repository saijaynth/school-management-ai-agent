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

export const STARTER_PROMPTS: { group: "Admin" | "Faculty" | "Student"; items: string[] }[] = [
  {
    group: "Admin",
    items: [
      "Show pending leave requests",
      "Student attendance report - who is below 75%?",
      "Grade 10 Mid Term rank list with totals and grades",
      "According to the manual, how do I approve a leave request?",
    ],
  },
  {
    group: "Faculty",
    items: [
      "Anita Sharma's timetable and assigned courses",
      "Faculty attendance on November 2, 2026",
      "Pending homework & assignment submissions",
      "How do I enter exam marks according to the manual?",
    ],
  },
  {
    group: "Student",
    items: [
      "Kiran Patel's Mid Term marks and progress report",
      "Arjun Rao's timetable and homework",
      "Show my unread notifications for stu_arjun",
      "How can a student apply for leave?",
    ],
  },
];
