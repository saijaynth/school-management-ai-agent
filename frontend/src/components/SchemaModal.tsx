import type { SchemaSummaryResponse } from "../types";

export default function SchemaModal({ summary, onClose }: { summary: SchemaSummaryResponse | null; onClose: () => void }) {
  if (!summary) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div
        className="max-h-[85vh] w-full max-w-3xl overflow-y-auto rounded-xl bg-white p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Database Schema Overview</h2>
            <p className="text-sm text-slate-500">
              Schema <span className="font-mono">{summary.schema_name}</span> • 12 modules • 29 tables •{" "}
              {summary.views.length} views
            </p>
          </div>
          <button onClick={onClose} className="rounded-md px-3 py-1 text-slate-400 hover:bg-slate-100">
            ✕
          </button>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          {summary.modules.map((m) => (
            <div key={m.module} className="rounded-lg border border-slate-200 p-3">
              <p className="text-sm font-semibold text-slate-800">{m.module}</p>
              <div className="mt-1.5 flex flex-wrap gap-1">
                {m.tables.map((t) => (
                  <span key={t} className="rounded bg-indigo-50 px-1.5 py-0.5 font-mono text-[10px] text-indigo-700">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        <h3 className="mt-5 mb-2 text-sm font-semibold text-slate-700">Built-in Reporting Views</h3>
        <div className="space-y-2">
          {summary.views.map((v) => (
            <div key={v.name} className="rounded-lg border border-emerald-200 bg-emerald-50/60 p-3">
              <p className="font-mono text-sm font-semibold text-emerald-800">{v.name}</p>
              <p className="text-xs text-emerald-900/80">{v.description}</p>
            </div>
          ))}
        </div>

        <h3 className="mt-5 mb-2 text-sm font-semibold text-slate-700">Sample Data Quick Reference</h3>
        <div className="grid gap-2 sm:grid-cols-2">
          {summary.entities.map((e) => (
            <div key={e.kind} className="rounded-lg bg-slate-50 p-2.5">
              <p className="text-xs font-semibold text-slate-700">{e.label}</p>
              <p className="mt-0.5 text-[11px] text-slate-500">{e.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
