import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ManualReference, SourceType } from "../types";
import { BookOpenIcon, DatabaseIcon, GraduationCapIcon } from "./Icons";

function SourceBadge({ source }: { source?: SourceType }) {
  if (!source) return null;
  const map: Record<string, { label: string; icon: typeof DatabaseIcon }> = {
    database: { label: "Campus database", icon: DatabaseIcon },
    manual: { label: "School manual", icon: BookOpenIcon },
    hybrid: { label: "Database & manual", icon: DatabaseIcon },
    clarification: { label: "Assistant", icon: GraduationCapIcon },
    error: { label: "Notice", icon: GraduationCapIcon },
  };
  const cfg = map[source] || map.clarification;
  const IconComponent = cfg.icon;

  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] font-medium text-slate-600">
      <IconComponent className="h-3 w-3 text-slate-500 shrink-0" />
      <span>{cfg.label}</span>
    </span>
  );
}

function ManualRefs({ refs }: { refs: ManualReference[] }) {
  if (!refs?.length) return null;
  return (
    <div className="mt-3 space-y-1.5 border-t border-slate-100 pt-2.5">
      <p className="text-[11px] font-medium text-slate-400">Referenced guidelines</p>
      {refs.map((r) => (
        <div key={r.section} className="rounded-lg border border-slate-200 bg-slate-50/70 p-2.5 text-xs text-slate-800">
          <p className="font-semibold text-slate-900">{r.section}</p>
          <p className="mt-0.5 text-[11px] text-slate-600 leading-relaxed">{r.summary}</p>
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
      <div className="flex justify-end my-1.5">
        <div className="max-w-[88%] sm:max-w-[76%] rounded-2xl rounded-tr-sm bg-slate-900 px-4 py-2.5 text-sm leading-relaxed text-slate-50 shadow-2xs">
          {content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start my-2">
      <div className="flex w-full sm:max-w-[92%] lg:max-w-[88%] gap-3 items-start">
        {/* Desktop-only side avatar (reclaims 44px on mobile) */}
        <div className="hidden sm:flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-slate-200 bg-slate-100 text-slate-700">
          <GraduationCapIcon className="h-4 w-4" />
        </div>

        {/* Bubble card */}
        <div
          className={`w-full min-w-0 rounded-xl border p-3.5 sm:p-5 shadow-2xs transition-colors ${
            failed ? "border-rose-200 bg-rose-50/50" : "border-slate-200/90 bg-white"
          }`}
        >
          {/* Header row */}
          <div className="mb-2.5 flex items-center justify-between gap-2 border-b border-slate-100 pb-2">
            <div className="flex items-center gap-2">
              <span className="sm:hidden flex h-5 w-5 items-center justify-center rounded bg-slate-100 text-slate-700">
                <GraduationCapIcon className="h-3 w-3" />
              </span>
              <SourceBadge source={sourceType} />
            </div>

            {thinking ? (
              <button
                onClick={() => setShowThinking(!showThinking)}
                className="text-[11px] font-medium text-slate-400 hover:text-slate-700 transition cursor-pointer"
              >
                {showThinking ? "Hide reasoning" : "View reasoning"}
              </button>
            ) : null}
          </div>

          {/* Reasoning trace (collapsible) */}
          {thinking && showThinking ? (
            <div className="mb-3 rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-xs text-slate-700 font-mono leading-relaxed overflow-x-auto">
              <Markdown remarkPlugins={[remarkGfm]}>{thinking}</Markdown>
            </div>
          ) : null}

          {/* Markdown Content */}
          <div className="prose prose-slate max-w-none text-sm leading-relaxed text-slate-800 break-words">
            <Markdown
              remarkPlugins={[remarkGfm]}
              components={{
                table: ({ ...props }) => (
                  <div className="my-3 -mx-1 sm:mx-0 overflow-x-auto rounded-lg border border-slate-200 bg-white">
                    <table className="min-w-full divide-y divide-slate-200 text-left text-xs tabular-nums" {...props} />
                  </div>
                ),
                thead: ({ ...props }) => (
                  <thead className="bg-slate-50/90 text-[11px] font-semibold text-slate-600 uppercase tracking-wide" {...props} />
                ),
                th: ({ ...props }) => (
                  <th className="px-3 py-2 border-b border-slate-200 whitespace-nowrap" {...props} />
                ),
                td: ({ ...props }) => (
                  <td className="px-3 py-2 border-b border-slate-100 text-slate-700 whitespace-nowrap font-normal" {...props} />
                ),
                tr: ({ ...props }) => <tr className="hover:bg-slate-50/60 transition-colors" {...props} />,
                ul: ({ ...props }) => <ul className="my-2 list-disc list-inside space-y-1 text-slate-700" {...props} />,
                ol: ({ ...props }) => <ol className="my-2 list-decimal list-inside space-y-1 text-slate-700" {...props} />,
                p: ({ ...props }) => <p className="my-1.5" {...props} />,
                strong: ({ ...props }) => <strong className="font-semibold text-slate-900" {...props} />,
                h3: ({ ...props }) => <h3 className="text-sm font-semibold text-slate-900 mt-2 mb-1" {...props} />,
                h4: ({ ...props }) => <h4 className="text-xs font-semibold text-slate-800 mt-2 mb-1" {...props} />,
              }}
            >
              {content}
            </Markdown>
          </div>

          {/* Manual references */}
          {manualReferences && <ManualRefs refs={manualReferences} />}

          {/* Follow-up Question Chips */}
          {followups && followups.length > 0 && onFollowup ? (
            <div className="mt-3.5 pt-2.5 border-t border-slate-100">
              <p className="text-[11px] font-medium text-slate-400 mb-1.5">Suggested follow-ups</p>
              <div className="flex flex-wrap gap-1.5">
                {followups.map((sugg) => (
                  <button
                    key={sugg}
                    onClick={() => onFollowup(sugg)}
                    className="inline-flex items-center rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-700 transition hover:border-slate-300 hover:bg-slate-100 hover:text-slate-900 cursor-pointer active:scale-95"
                  >
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
