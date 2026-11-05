import { useEffect, useState } from "react";
import type { SchemaSummaryResponse, HealthResponse } from "../types";
import { STARTER_PROMPTS } from "../types";

type Tab = "starters" | "explorer";

export default function Sidebar({
  health,
  summary,
  onPick,
  onReset,
}: {
  health: HealthResponse | null;
  summary: SchemaSummaryResponse | null;
  onPick: (q: string) => void;
  onReset: () => void;
}) {
  const [tab, setTab] = useState<Tab>("starters");
  const [openModule, setOpenModule] = useState<string | null>(null);
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({ Admin: true, Faculty: true, Student: true });
  const [drawerOpen, setDrawerOpen] = useState(false);

  useEffect(() => {
    if (openModule === null && summary?.modules.length) setOpenModule(summary.modules[0].module);
  }, [summary, openModule]);

  const toggleGroup = (g: string) => setOpenGroups((s) => ({ ...s, [g]: !s[g] }));

  const content = (
    <div className="flex h-full w-72 min-w-72 flex-col border-r border-slate-200 bg-white">
      {/* Branding + connection status */}
      <div className="border-b border-slate-100 px-4 py-4">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600 text-lg font-bold text-white">🎓</div>
          <div>
            <h1 className="text-sm font-semibold leading-tight text-slate-900">School Management AI Assistant</h1>
            <p className="text-[11px] text-slate-500">PostgreSQL • Read-only | grounded answers</p>
          </div>
        </div>
        <div
          className={`mt-3 rounded-md px-2.5 py-1.5 text-[11px] font-medium ${
            health?.database.connected ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-700"
          }`}
        >
          {health?.database.connected
            ? `Connected: ${health.database.schema ?? "school"} schema • ${health.counts.tables} Tables • ${health.counts.views} Views`
            : "Database not connected"}
          {health && !health.llm.ready ? <div className="mt-0.5 text-amber-600">LLM key not configured</div> : null}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-100 text-sm">
        <button
          onClick={() => setTab("starters")}
          className={`flex-1 px-3 py-2 font-medium ${tab === "starters" ? "border-b-2 border-indigo-600 text-indigo-700" : "text-slate-500"}`}
        >
          Quick Prompts
        </button>
        <button
          onClick={() => setTab("explorer")}
          className={`flex-1 px-3 py-2 font-medium ${tab === "explorer" ? "border-b-2 border-indigo-600 text-indigo-700" : "text-slate-500"}`}
        >
          Database Explorer
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3">
        {tab === "starters" &&
          STARTER_PROMPTS.map((g) => (
            <div key={g.group} className="mb-3">
              <button
                onClick={() => toggleGroup(g.group)}
                className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left text-xs font-semibold uppercase tracking-wide text-slate-500 hover:bg-slate-50"
              >
                {g.group} Perspective
                <span>{openGroups[g.group] ? "−" : "+"}</span>
              </button>
              {openGroups[g.group] && (
                <div className="mt-1 space-y-1">
                  {g.items.map((q) => (
                    <button
                      key={q}
                      onClick={() => onPick(q)}
                      className="block w-full rounded-md border border-slate-100 bg-slate-50 px-2.5 py-2 text-left text-xs text-slate-700 transition hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}

        {tab === "explorer" && summary && (
          <div>
            <div className="mb-3 rounded-lg bg-slate-50 p-3 text-[11px] leading-relaxed text-slate-600">
              <p className="font-semibold text-slate-700">Quick reference sample logins</p>
              <p className="mt-0.5">admin01 · fac_anita · fac_vijay · fac_sara · stu_arjun · stu_meera · stu_kiran · stu_divya</p>
            </div>
            {summary.entities.map((e) => (
              <div key={e.kind} className="mb-2 rounded-lg border border-slate-100 p-2.5">
                <p className="text-xs font-semibold text-slate-800">{e.label}</p>
                <p className="mt-0.5 text-[11px] leading-snug text-slate-500">{e.detail}</p>
              </div>
            ))}
            <p className="mt-4 text-xs font-semibold uppercase tracking-wide text-slate-500">12 Modules · 29 Tables</p>
            {summary.modules.map((m) => (
              <div key={m.module} className="mt-2 rounded-lg border border-slate-100">
                <button
                  onClick={() => setOpenModule(openModule === m.module ? null : m.module)}
                  className="flex w-full items-center justify-between px-2.5 py-2 text-left text-xs font-medium text-slate-700 hover:bg-slate-50"
                >
                  {m.module} <span className="text-slate-400">{openModule === m.module ? "−" : "+"}</span>
                </button>
                {openModule === m.module && (
                  <div className="flex flex-wrap gap-1 border-t border-slate-100 px-2.5 py-2">
                    {m.tables.map((t) => (
                      <span key={t} className="rounded bg-indigo-50 px-1.5 py-0.5 font-mono text-[10px] text-indigo-700">
                        {t}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
            <p className="mt-4 text-xs font-semibold uppercase tracking-wide text-slate-500">4 Built-in Views</p>
            {summary.views.map((v) => (
              <div key={v.name} className="mb-2 rounded-lg border border-emerald-100 bg-emerald-50/50 p-2.5">
                <p className="font-mono text-[11px] font-semibold text-emerald-800">{v.name}</p>
                <p className="mt-0.5 text-[11px] text-emerald-900/70">{v.description}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="border-t border-slate-100 p-3">
        <button
          onClick={onReset}
          className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-slate-900"
        >
          ⟲ New Chat / Reset Session
        </button>
      </div>
    </div>
  );

  return (
    <>
      <button
        onClick={() => setDrawerOpen(true)}
        className="fixed bottom-4 left-4 z-40 rounded-full bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-lg lg:hidden"
      >
        ☰ Menu
      </button>
      <aside className="hidden lg:block lg:h-screen">{content}</aside>
      {drawerOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/40" onClick={() => setDrawerOpen(false)} />
          <div className="absolute inset-y-0 left-0 w-72">{content}</div>
        </div>
      )}
    </>
  );
}
