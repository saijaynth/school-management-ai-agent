import { useEffect, useRef, useState } from "react";
import type { ChatMessage, HealthResponse } from "../types";
import MessageBubble from "./MessageBubble";
import { GraduationCapIcon, MenuIcon, SendIcon } from "./Icons";

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
      <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 shadow-2xs text-xs text-slate-500">
        <span className="h-1.5 w-1.5 rounded-full bg-slate-400 animate-pulse" />
        <span>Looking up campus records...</span>
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
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-200/80 bg-white px-3 sm:px-6">
        <div className="flex items-center gap-2.5 sm:gap-3">
          {onToggleSidebar && (
            <button
              onClick={onToggleSidebar}
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100 lg:hidden cursor-pointer active:scale-95 transition-colors"
              title="Open Campus Guide"
              aria-label="Open Campus Guide"
            >
              <MenuIcon className="h-4 w-4" />
            </button>
          )}
          <div>
            <h1 className="text-sm font-semibold text-slate-900 tracking-tight flex items-center gap-2">
              <span>School Management Assistant</span>
            </h1>
            <p className="text-[11px] text-slate-500 hidden sm:block">Campus database records and policy guidance</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-600">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                health?.status === "ok" ? "bg-emerald-600" : "bg-amber-500"
              }`}
            />
            <span>{health?.status === "ok" ? "Database connected" : "Connecting"}</span>
          </span>
        </div>
      </header>

      {/* Messages Scroll Area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 sm:px-6 py-4">
        {messages.length === 0 ? (
          <div className="mx-auto my-auto max-w-xl text-center py-8 sm:py-16 px-2">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-800 shadow-2xs">
              <GraduationCapIcon className="h-6 w-6" />
            </div>
            <h2 className="mt-4 text-lg font-semibold text-slate-900 tracking-tight">Campus Records Assistant</h2>
            <p className="mt-1.5 text-xs text-slate-600 leading-relaxed max-w-md mx-auto">
              Ask questions regarding student attendance, exam rankings, teacher timetables, homework, or official school procedures.
            </p>

            <div className="mt-8 text-left">
              <p className="text-[11px] font-medium text-slate-400 mb-2.5 text-center">
                Frequently asked queries
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => onSend(s)}
                    className="rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-700 shadow-2xs transition hover:border-slate-300 hover:bg-slate-50 text-left cursor-pointer active:scale-[0.99]"
                  >
                    <span className="font-normal text-slate-800">{s}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-3 max-w-4xl mx-auto">
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
          <div className="mt-4 rounded-lg border border-rose-200 bg-rose-50/70 p-3.5 text-xs text-rose-800 max-w-xl mx-auto">
            <p className="font-medium">The assistant could not reach the database</p>
            <p className="mt-1 text-[11px] text-rose-700">{error}</p>
            <button
              onClick={onRetry}
              className="mt-2.5 inline-flex items-center rounded-md bg-rose-700 px-3 py-1.5 text-xs font-medium text-white shadow-xs hover:bg-rose-800 transition cursor-pointer"
            >
              Retry query
            </button>
          </div>
        ) : null}
      </div>

      {/* Input Form Footer */}
      <footer className="border-t border-slate-200/80 bg-white p-3 sm:p-4">
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
              placeholder="Ask about students, attendance, timetables, marks, leaves..."
              disabled={loading}
              className="w-full rounded-xl border border-slate-300 bg-slate-50/50 px-3.5 py-2.5 text-xs sm:text-sm text-slate-900 placeholder:text-slate-400 shadow-inner transition focus:border-slate-400 focus:bg-white focus:outline-none focus:ring-1 focus:ring-slate-300 disabled:opacity-50"
            />
          </div>
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="flex h-10 min-h-[44px] min-w-[44px] sm:px-4 items-center justify-center gap-1.5 rounded-xl bg-slate-900 text-xs font-medium text-white shadow-2xs transition hover:bg-slate-800 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 cursor-pointer"
            title="Send query"
            aria-label="Send query"
          >
            <span className="hidden sm:inline">Send</span>
            <SendIcon className="h-3.5 w-3.5" />
          </button>
        </form>
        <p className="mt-1.5 text-center text-[10px] sm:text-[11px] text-slate-400">
          Grounded strictly in the school database and user manual
        </p>
      </footer>
    </main>
  );
}
