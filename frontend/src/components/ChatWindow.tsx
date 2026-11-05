import { useEffect, useRef, useState } from "react";
import type { ChatMessage, HealthResponse } from "../types";
import MessageBubble from "./MessageBubble";
import SchemaModal from "./SchemaModal";

const LOADING_STEPS = [
  "Analyzing question…",
  "Selecting tables & views on the school schema…",
  "Executing read-only SQL on school schema…",
  "Cross-checking the User Manual…",
  "Synthesizing answer…",
];

function LoadingSteps({ active }: { active: boolean }) {
  const [step, setStep] = useState(0);
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setStep((s) => Math.min(s + 1, LOADING_STEPS.length - 1)), 900);
    return () => clearInterval(t);
  }, [active]);
  if (!active) return null;
  return (
    <div className="flex justify-start">
      <div className="rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3 text-sm shadow-sm">
        <div className="space-y-1.5">
          {LOADING_STEPS.slice(0, step + 1).map((s, i) => (
            <div key={s} className="flex items-center gap-2">
              {i < step ? (
                <span className="text-emerald-600">✓</span>
              ) : (
                <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
              )}
              <span className={i < step ? "text-slate-400" : "text-slate-700"}>{s}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function ChatWindow({
  sessionId,
  messages,
  loading,
  error,
  onSend,
  onRetry,
  health,
  summary,
}: {
  sessionId: string;
  messages: ChatMessage[];
  loading: boolean;
  error: string | null;
  onSend: (q: string) => void;
  onRetry: () => void;
  health: HealthResponse | null;
  summary: import("../types").SchemaSummaryResponse | null;
}) {
  const [input, setInput] = useState("");
  const [showSql, setShowSql] = useState(true);
  const [schemaOpen, setSchemaOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const submit = () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    onSend(text);
  };

  return (
    <main className="flex h-screen min-w-0 flex-1 flex-col bg-slate-50">
      {/* Top bar */}
      <div className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-2.5">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-slate-900">
            Session {sessionId ? sessionId.slice(0, 10) : "(new)"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="hidden items-center gap-1.5 text-xs text-slate-600 sm:flex">
            <input type="checkbox" checked={showSql} onChange={(e) => setShowSql(e.target.checked)} className="accent-indigo-600" />
            Show Executed SQL &amp; Data Tables
          </label>
          <span
            className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium ${
              health?.status === "ok" ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"
            }`}
          >
            {health ? `System: ${health.status}` : "System: checking…"}
          </span>
          <button onClick={() => setSchemaOpen(true)} className="rounded-md border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50">
            Schema
          </button>
        </div>
      </div>

      {/* Show/hide data panels globally */}
      <div className="hidden" data-show-sql={String(showSql)} />

      {/* Stream */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.map((m, i) => (
            <MessageBubble
              key={i}
              role={m.role}
              content={m.content}
              failed={m.failed}
              sourceType={showSql ? m.anytime?.source_type : "clarification"}
              sqlQueries={showSql ? m.anytime?.sql_queries ?? [] : []}
              manualReferences={m.anytime?.manual_references ?? []}
              followups={m.anytime?.suggested_followups}
              onFollowup={(q) => onSend(displayFollowup(q))}
            />
          ))}
          <LoadingSteps active={loading} />
          {error && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 p-4">
              <p className="text-sm font-medium text-rose-800">Something went wrong reaching the assistant</p>
              <p className="mt-1 text-xs text-rose-600">{error}</p>
              <button onClick={onRetry} className="mt-2 rounded-md bg-rose-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-rose-700">
                Retry
              </button>
              <p className="mt-2 text-[11px] text-rose-500/80">
                Hint: make sure the FastAPI backend is running on port 8000 and the PostgreSQL server is started.
              </p>
            </div>
          )}
          {messages.length === 0 && !loading && <Welcome health={health} onPick={onSend} />}
        </div>
      </div>

      {/* Composer */}
      <div className="border-t border-slate-200 bg-white px-4 py-3">
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder="Ask about students, attendance, exams, timetables, manual workflows… (Enter to send, Shift+Enter for newline)"
            className="max-h-40 flex-1 resize-none rounded-xl border border-slate-300 px-3.5 py-2.5 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
            style={{ height: `${Math.min(2 + input.split("\n").length, 7) * 1.4}rem` }}
          />
          <button
            onClick={submit}
            disabled={loading || !input.trim()}
            className="rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? "…" : "Send"}
          </button>
        </div>
      </div>

      {schemaOpen && <SchemaModal summary={summary} onClose={() => setSchemaOpen(false)} />}
    </main>
  );
}

function displayFollowup(q: string): string {
  /* Follow-up chips pass the original suggested text unchanged; the agent
     resolves references from its own conversation memory. */
  return q;
}

function Welcome({ health, onPick }: { health: HealthResponse | null; onPick: (q: string) => void }) {
  const cards = [
    { n: "8", label: "Users", sub: "1 Admin, 3 Faculty, 4 Students" },
    { n: "2", label: "Classes", sub: "Grade 9 & Grade 10, 1 section each" },
    { n: "3", label: "Subjects", sub: "Mathematics, Science, English" },
    { n: "4", label: "Built-in Views", sub: "attendance • exam • leaves • timetable" },
  ];
  const starters = [
    "Which leave requests are pending approval?",
    "Show all student attendance percentages",
    "What is Grade 10-A's Monday timetable?",
    "Compare Kiran Patel and Divya Nair's marks",
  ];
  return (
    <div className="pt-8">
      <div className="text-center">
        <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-600 text-2xl text-white shadow">🎓</div>
        <h2 className="text-xl font-semibold text-slate-900">School Management AI Assistant</h2>
        <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
          Ask anything about the <span className="font-medium text-slate-700">school</span> PostgreSQL test database or the
          School Management Database User Manual. Every data answer is grounded in live read-only SQL.
        </p>
        {health && !health.llm.ready && (
          <p className="mx-auto mt-2 max-w-md rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700">
            No LLM API key configured yet - add <span className="font-mono">OPENAI_API_KEY</span> (or Anthropic / Gemini /
            Ollama) to <span className="font-mono">backend/.env</span> or the project <span className="font-mono">.env</span>{" "}
            and restart the backend.
          </p>
        )}
      </div>
      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {cards.map((c) => (
          <div key={c.label} className="rounded-xl border border-slate-200 bg-white p-3 text-center">
            <p className="text-2xl font-bold text-indigo-600">{c.n}</p>
            <p className="text-xs font-semibold text-slate-700">{c.label}</p>
            <p className="mt-0.5 text-[10px] text-slate-400">{c.sub}</p>
          </div>
        ))}
      </div>
      <div className="mx-auto mt-5 max-w-xl space-y-1.5">
        {starters.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="block w-full rounded-lg border border-slate-200 bg-white px-3.5 py-2.5 text-left text-sm text-slate-700 transition hover:border-indigo-300 hover:bg-indigo-50"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
