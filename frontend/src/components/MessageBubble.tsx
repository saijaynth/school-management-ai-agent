import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ManualReference, SourceType } from "../types";

function SourceBadge({ source }: { source?: SourceType }) {
  if (!source) return null;
  const map: Record<string, { cls: string; label: string; icon: string }> = {
    database: { cls: "bg-emerald-50 text-emerald-700 border-emerald-200", label: "Campus Records", icon: "🎓" },
    manual: { cls: "bg-blue-50 text-blue-700 border-blue-200", label: "School Guidelines", icon: "📘" },
    hybrid: { cls: "bg-purple-50 text-purple-700 border-purple-200", label: "Campus Records & Guidelines", icon: "✨" },
    clarification: { cls: "bg-slate-50 text-slate-600 border-slate-200", label: "Assistant", icon: "💬" },
    error: { cls: "bg-rose-50 text-rose-700 border-rose-200", label: "Notice", icon: "⚠️" },
  };
  const cfg = map[source] || map.clarification;
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-medium shadow-2xs ${cfg.cls}`}>
      <span>{cfg.icon}</span>
      <span>{cfg.label}</span>
    </span>
  );
}

function ManualRefs({ refs }: { refs: ManualReference[] }) {
  if (!refs?.length) return null;
  return (
    <div className="mt-3 space-y-1.5 border-t border-slate-100 pt-2.5">
      <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">Referenced Guidelines:</p>
      {refs.map((r) => (
        <div key={r.section} className="rounded-lg border border-blue-100 bg-blue-50/50 px-3 py-2 text-xs text-blue-900">
          <p className="font-semibold text-blue-950">📖 {r.section}</p>
          <p className="mt-0.5 text-[11px] text-blue-800/90 leading-relaxed">{r.summary}</p>
        </div>
      ))}
    </div>
  );
}

export default function MessageBubble({
  role,
  content,
  thinking,
  sourceType,
  manualReferences,
  followups,
  onFollowup,
  failed,
}: {
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  sourceType?: SourceType;
  manualReferences?: ManualReference[];
  followups?: string[];
  onFollowup?: (q: string) => void;
  failed?: boolean;
}) {
  const [showThinking, setShowThinking] = useState(false);
  const isUser = role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end my-1">
        <div className="max-w-[78%] rounded-2xl rounded-br-sm bg-gradient-to-r from-indigo-600 to-indigo-700 px-4 py-2.5 text-sm leading-relaxed text-white shadow-xs">
          {content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start my-2">
      <div className="flex gap-2.5 max-w-[94%] items-start">
        {/* Assistant Avatar */}
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-indigo-100 text-indigo-700 text-sm font-bold shadow-2xs border border-indigo-200/50">
          🎓
        </div>

        <div
          className={`flex-1 rounded-2xl rounded-tl-sm border px-5 py-4 shadow-xs transition ${
            failed ? "border-rose-200 bg-rose-50/70" : "border-slate-200/90 bg-white"
          }`}
        >
          {/* Header Badge */}
          <div className="mb-2 flex items-center justify-between gap-2">
            <SourceBadge source={sourceType} />
            {thinking ? (
              <button
                onClick={() => setShowThinking(!showThinking)}
                className="text-[11px] text-slate-400 hover:text-indigo-600 transition flex items-center gap-1 cursor-pointer font-medium"
              >
                <span>{showThinking ? "Hide reasoning" : "View reasoning"}</span>
                <span>{showThinking ? "▲" : "▼"}</span>
              </button>
            ) : null}
          </div>

          {/* Reasoning trace (collapsible, subtle) */}
          {thinking && showThinking ? (
            <div className="mb-3 rounded-xl border border-indigo-100 bg-indigo-50/40 p-3 text-xs text-indigo-900 leading-relaxed font-mono">
              <Markdown remarkPlugins={[remarkGfm]}>{thinking}</Markdown>
            </div>
          ) : null}

          {/* Main Answer Content */}
          <div className="prose prose-slate max-w-none text-sm leading-relaxed text-slate-800 break-words">
            <Markdown
              remarkPlugins={[remarkGfm]}
              components={{
                table: ({ ...props }) => (
                  <div className="my-3 overflow-x-auto rounded-xl border border-slate-200 shadow-2xs">
                    <table className="min-w-full divide-y divide-slate-200 text-left text-xs" {...props} />
                  </div>
                ),
                thead: ({ ...props }) => <thead className="bg-slate-50 font-semibold text-slate-700" {...props} />,
                th: ({ ...props }) => <th className="px-3.5 py-2.5 border-b border-slate-200" {...props} />,
                td: ({ ...props }) => <td className="px-3.5 py-2 border-b border-slate-100 text-slate-600" {...props} />,
                tr: ({ ...props }) => <tr className="hover:bg-slate-50/70 transition" {...props} />,
                ul: ({ ...props }) => <ul className="my-2 list-disc list-inside space-y-1 text-slate-700" {...props} />,
                ol: ({ ...props }) => <ol className="my-2 list-decimal list-inside space-y-1 text-slate-700" {...props} />,
                p: ({ ...props }) => <p className="my-1.5" {...props} />,
                strong: ({ ...props }) => <strong className="font-semibold text-slate-900" {...props} />,
              }}
            >
              {content}
            </Markdown>
          </div>

          {/* Manual references if available */}
          {manualReferences && <ManualRefs refs={manualReferences} />}

          {/* Suggested Interactive Follow-up Chips */}
          {followups && followups.length > 0 && onFollowup ? (
            <div className="mt-3.5 pt-2.5 border-t border-slate-100">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide mb-1.5">Suggested Next Questions:</p>
              <div className="flex flex-wrap gap-1.5">
                {followups.map((sugg) => (
                  <button
                    key={sugg}
                    onClick={() => onFollowup(sugg)}
                    className="inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-700 transition hover:border-indigo-300 hover:bg-indigo-50 hover:text-indigo-700 cursor-pointer active:scale-95 shadow-2xs"
                  >
                    <span>💬</span>
                    <span>{sugg}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
