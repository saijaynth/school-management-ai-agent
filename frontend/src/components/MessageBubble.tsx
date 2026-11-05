import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ManualReference, SqlQueryRecord, SourceType } from "../types";
import SqlResultTable from "./SqlResultTable";

function SourceBadge({ source }: { source?: SourceType }) {
  if (!source) return null;
  const map: Record<string, { cls: string; label: string }> = {
    database: { cls: "bg-emerald-100 text-emerald-700 border-emerald-200", label: "Live Database Query" },
    manual: { cls: "bg-amber-100 text-amber-700 border-amber-200", label: "User Manual" },
    hybrid: { cls: "bg-indigo-100 text-indigo-700 border-indigo-200", label: "Database + User Manual" },
    clarification: { cls: "bg-slate-100 text-slate-600 border-slate-200", label: "Clarification" },
    error: { cls: "bg-rose-100 text-rose-700 border-rose-200", label: "Error" },
  };
  const cfg = map[source];
  if (!cfg) return null;
  return (
    <span className={`inline-block rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${cfg.cls}`}>
      {cfg.label}
    </span>
  );
}

function ManualRefs({ refs }: { refs: ManualReference[] }) {
  if (!refs?.length) return null;
  return (
    <div className="mt-2 space-y-2">
      {refs.map((r) => (
        <details key={r.section} className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm">
          <summary className="cursor-pointer font-medium text-amber-800">📖 {r.section}</summary>
          <p className="mt-1 whitespace-pre-wrap text-xs text-amber-900/80">{r.summary}</p>
        </details>
      ))}
    </div>
  );
}

export default function MessageBubble({
  role,
  content,
  sourceType,
  sqlQueries,
  manualReferences,
  followups,
  onFollowup,
  failed,
}: {
  role: "user" | "assistant";
  content: string;
  sourceType?: SourceType;
  sqlQueries?: SqlQueryRecord[];
  manualReferences?: ManualReference[];
  followups?: string[];
  onFollowup?: (q: string) => void;
  failed?: boolean;
}) {
  const [showData, setShowData] = useState(true);
  const isUser = role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-indigo-600 px-4 py-2.5 text-sm leading-relaxed text-white shadow-sm">
          {content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div
        className={`max-w-[92%] rounded-2xl rounded-bl-sm border px-4 py-3 shadow-sm ${
          failed ? "border-rose-200 bg-rose-50" : "border-slate-200 bg-white"
        }`}
      >
        <div className="mb-1.5 flex flex-wrap items-center gap-2">
          <SourceBadge source={sourceType} />
          {sourceType && (sourceType === "database" || sourceType === "hybrid") && sqlQueries?.length ? (
            <button
              onClick={() => setShowData((v) => !v)}
              className="rounded-full border border-slate-200 px-2.5 py-0.5 text-[11px] font-medium text-slate-600 hover:bg-slate-50"
            >
              {showData ? "Hide" : "Show"} Executed SQL &amp; Data ({sqlQueries.length})
            </button>
          ) : null}
        </div>

        <div className={`prose prose-slate prose-sm max-w-none ${failed ? "text-rose-800" : ""}`}>
          <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
        </div>

        {showData &&
          sqlQueries?.map((q, i) => <SqlResultTable key={i} query={q} />)}
        <ManualRefs refs={manualReferences ?? []} />

        {followups?.length ? (
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {followups.map((f) => (
              <button
                key={f}
                onClick={() => onFollowup?.(f)}
                className="rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700 transition hover:bg-indigo-100"
              >
                {f}
              </button>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
