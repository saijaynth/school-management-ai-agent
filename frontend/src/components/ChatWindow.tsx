import { useEffect, useRef, useState } from "react";
import type { ChatMessage, HealthResponse } from "../types";
import MessageBubble from "./MessageBubble";

const SUGGESTIONS = [
  "Who is below 75% student attendance?",
  "Who teaches Science?",
  "What is Grade 10-A's Monday timetable?",
  "Compare Kiran Patel and Divya Nair's marks",
  "What is Sara Thomas's room and schedule on Monday?",
  "Which leave requests are pending approval?",
];

function TypingIndicator({ active }: { active: boolean }) {
  if (!active) return null;
  return (
    <div className="flex justify-start my-2">
      <div className="flex gap-2.5 max-w-[85%] items-center">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-indigo-100 text-indigo-700 text-sm font-bold shadow-2xs border border-indigo-200/50">
          🎓
        </div>
        <div className="flex items-center gap-2 rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3 shadow-xs">
          <div className="flex gap-1.5">
            <span className="h-2 w-2 rounded-full bg-indigo-600 animate-bounce [animation-delay:-0.3s]" />
            <span className="h-2 w-2 rounded-full bg-indigo-600 animate-bounce [animation-delay:-0.15s]" />
            <span className="h-2 w-2 rounded-full bg-indigo-600 animate-bounce" />
          </div>
          <span className="text-xs font-medium text-slate-500 ml-1">Assistant is looking up records...</span>
        </div>
      </div>
    </div>
  );
}

export default function ChatWindow({
  messages,
  loading,
  error,
  onSend,
  onRetry,
  health,
  onToggleSidebar,
}: {
  sessionId: string;
  messages: ChatMessage[];
  loading: boolean;
  error: string | null;
  onSend: (q: string) => void;
  onRetry: () => void;
  health: HealthResponse | null;
  summary: import("../types").SchemaSummaryResponse | null;
  onToggleSidebar?: () => void;
}) {
  const [input, setInput] = useState("");
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
    <main className="flex h-screen min-w-0 flex-1 flex-col bg-slate-50/70">
      {/* Top Header */}
      <header className="flex items-center justify-between border-b border-slate-200/80 bg-white px-3 sm:px-6 py-2.5 sm:py-3 shadow-2xs">
        <div className="flex items-center gap-2.5 sm:gap-3">
          {onToggleSidebar && (
            <button
              onClick={onToggleSidebar}
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-slate-50 text-slate-700 shadow-2xs hover:bg-slate-100 lg:hidden cursor-pointer active:scale-95 transition"
              title="Open Campus Guide"
              aria-label="Open Campus Guide"
            >
              <span className="text-base font-bold">☰</span>
            </button>
          )}
          <div>
            <h2 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-1.5">
              <span>School AI Assistant</span>
            </h2>
            <p className="text-[11px] text-slate-500 hidden sm:block">Ask anything about students, faculty, timetables, attendance, or school guidelines</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] sm:text-xs font-medium border ${
              health?.status === "ok"
                ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                : "bg-amber-50 text-amber-700 border-amber-200"
            }`}
          >
            <span className={`h-2 w-2 rounded-full ${health?.status === "ok" ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`} />
            <span>{health?.status === "ok" ? "Assistant Online" : "Connecting..."}</span>
          </span>
        </div>
      </header>

      {/* Messages Scroll Area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6 sm:px-8">
        {messages.length === 0 ? (
          <div className="mx-auto my-auto max-w-xl text-center py-12">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-tr from-indigo-600 to-indigo-500 text-3xl text-white shadow-md">
              🎓
            </div>
            <h3 className="mt-4 text-xl font-bold text-slate-900">How can I help you today?</h3>
            <p className="mt-2 text-sm text-slate-600 leading-relaxed">
              I can answer questions regarding students, attendance, faculty, timetables, exam marks, and administrative procedures.
            </p>

            <div className="mt-8 text-left">
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 text-center">
                Popular Quick Questions
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => onSend(s)}
                    className="rounded-xl border border-slate-200 bg-white p-3 text-xs text-slate-700 shadow-2xs transition hover:border-indigo-300 hover:bg-indigo-50/50 hover:text-indigo-700 hover:shadow-xs active:scale-[0.98] text-left cursor-pointer"
                  >
                    <span className="font-medium">💬 {s}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4 max-w-4xl mx-auto">
            {messages.map((m, idx) => (
              <MessageBubble
                key={idx}
                role={m.role}
                content={m.content}
                thinking={m.thinking}
                sourceType={m.anytime?.source_type}
                manualReferences={m.anytime?.manual_references}
                followups={m.anytime?.suggested_followups}
                onFollowup={(q) => onSend(q)}
                failed={m.failed}
              />
            ))}
            <TypingIndicator active={loading} />
          </div>
        )}

        {/* Error notification banner */}
        {error ? (
          <div className="mx-auto mt-4 max-w-lg rounded-xl border border-rose-200 bg-rose-50 p-4 text-center text-xs text-rose-800 shadow-2xs">
            <p className="font-bold">Notice</p>
            <p className="mt-0.5">{error}</p>
            <button
              onClick={onRetry}
              className="mt-2 inline-flex items-center gap-1.5 rounded-lg bg-rose-600 px-3 py-1.5 font-semibold text-white shadow-xs hover:bg-rose-700 transition cursor-pointer"
            >
              <span>Retry Question</span>
            </button>
          </div>
        ) : null}
      </div>

      {/* Input Form Bar */}
      <footer className="border-t border-slate-200/80 bg-white p-4 sm:px-8">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
          className="mx-auto flex max-w-4xl items-center gap-2"
        >
          <div className="relative flex-1">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about students, teachers, attendance, timetables, exams, leaves..."
              disabled={loading}
              className="w-full rounded-2xl border border-slate-300 bg-slate-50/60 px-4 py-3 text-sm text-slate-800 shadow-inner transition focus:border-indigo-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-100 disabled:opacity-50"
            />
          </div>
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="flex h-11 items-center justify-center gap-1.5 rounded-2xl bg-indigo-600 px-5 text-sm font-semibold text-white shadow-xs transition hover:bg-indigo-700 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 cursor-pointer"
          >
            <span>Send</span>
            <span>➤</span>
          </button>
        </form>
        <p className="mt-2 text-center text-[11px] text-slate-400">
          School AI Agent • Responses are grounded strictly in the school database
        </p>
      </footer>
    </main>
  );
}
