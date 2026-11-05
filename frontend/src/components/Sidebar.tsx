import { useState } from "react";
import type { HealthResponse, SchemaSummaryResponse } from "../types";
import { STARTER_PROMPTS } from "../types";

type Tab = "prompts" | "directory";

export default function Sidebar({
  health,
  summary: _summary,
  onPick,
  onReset,
}: {
  health: HealthResponse | null;
  summary?: SchemaSummaryResponse | null;
  onPick: (q: string) => void;
  onReset: () => void;
}) {
  const [tab, setTab] = useState<Tab>("prompts");
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({
    "Attendance & Leaves": true,
    "Classes & Timetable": true,
    "Exams & Marks": true,
    "Campus Guidelines": true,
  });

  const toggleGroup = (g: string) => setOpenGroups((s) => ({ ...s, [g]: !s[g] }));

  return (
    <aside className="flex h-full w-80 min-w-80 flex-col border-r border-slate-200 bg-white shadow-xs">
      {/* Friendly Campus Branding */}
      <div className="border-b border-slate-100 p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-indigo-600 to-indigo-500 text-xl text-white shadow-sm">
            🎓
          </div>
          <div>
            <h1 className="text-sm font-bold text-slate-900 tracking-tight">School AI Assistant</h1>
            <p className="text-xs text-slate-500">Your interactive campus guide</p>
          </div>
        </div>

        {/* Status Pill */}
        <div
          className={`mt-3 flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
            health?.database.connected ? "bg-emerald-50 text-emerald-700 border border-emerald-100" : "bg-rose-50 text-rose-700 border border-rose-100"
          }`}
        >
          <span className={`h-2 w-2 rounded-full ${health?.database.connected ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`} />
          <span>{health?.database.connected ? "School Records Online & Ready" : "System Reconnecting..."}</span>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-100 bg-slate-50/50 text-xs font-semibold">
        <button
          onClick={() => setTab("prompts")}
          className={`flex-1 py-2.5 text-center transition ${
            tab === "prompts" ? "border-b-2 border-indigo-600 bg-white text-indigo-600 shadow-2xs" : "text-slate-500 hover:text-slate-700"
          }`}
        >
          💡 Quick Topics
        </button>
        <button
          onClick={() => setTab("directory")}
          className={`flex-1 py-2.5 text-center transition ${
            tab === "directory" ? "border-b-2 border-indigo-600 bg-white text-indigo-600 shadow-2xs" : "text-slate-500 hover:text-slate-700"
          }`}
        >
          🏫 Campus Directory
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {tab === "prompts" &&
          STARTER_PROMPTS.map((g) => (
            <div key={g.group} className="rounded-xl border border-slate-100 bg-slate-50/40 p-2">
              <button
                onClick={() => toggleGroup(g.group)}
                className="flex w-full items-center justify-between px-2 py-1 text-left text-xs font-semibold text-slate-700 hover:text-indigo-600"
              >
                <span>{g.group}</span>
                <span className="text-slate-400 text-sm">{openGroups[g.group] ? "▾" : "▸"}</span>
              </button>
              {openGroups[g.group] && (
                <div className="mt-1.5 space-y-1">
                  {g.items.map((q) => (
                    <button
                      key={q}
                      onClick={() => onPick(q)}
                      className="block w-full rounded-lg border border-transparent bg-white px-3 py-2 text-left text-xs text-slate-700 shadow-2xs transition hover:border-indigo-200 hover:bg-indigo-50/60 hover:text-indigo-700 hover:shadow-xs active:scale-[0.99]"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}

        {tab === "directory" && (
          <div className="space-y-2.5">
            <div className="rounded-xl border border-indigo-100 bg-indigo-50/50 p-3 text-xs leading-relaxed text-indigo-900">
              <p className="font-semibold text-indigo-950 flex items-center gap-1.5">
                <span>✨</span> Instant Query Support
              </p>
              <p className="mt-1 text-[11px] text-indigo-800">
                You can ask questions about any student, teacher, timetable, or exam in normal plain English!
              </p>
            </div>

            {/* Students Card */}
            <div className="rounded-xl border border-slate-200/80 bg-white p-3 shadow-2xs">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-900">
                <span>🧑‍🎓</span>
                <span>Enrolled Students</span>
                <span className="ml-auto rounded-full bg-slate-100 px-2 py-0.5 text-[10px] text-slate-600 font-bold">4</span>
              </div>
              <ul className="mt-2 space-y-1.5 text-xs text-slate-600">
                <li className="flex justify-between items-center bg-slate-50 rounded-md px-2 py-1">
                  <span className="font-medium text-slate-800">Grade 9-A</span>
                  <span>Arjun Rao, Meera Iyer</span>
                </li>
                <li className="flex justify-between items-center bg-slate-50 rounded-md px-2 py-1">
                  <span className="font-medium text-slate-800">Grade 10-A</span>
                  <span>Kiran Patel, Divya Nair</span>
                </li>
              </ul>
            </div>

            {/* Faculty Card */}
            <div className="rounded-xl border border-slate-200/80 bg-white p-3 shadow-2xs">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-900">
                <span>👨‍🏫</span>
                <span>Faculty &amp; Staff</span>
                <span className="ml-auto rounded-full bg-slate-100 px-2 py-0.5 text-[10px] text-slate-600 font-bold">3</span>
              </div>
              <div className="mt-2 space-y-1.5 text-xs text-slate-600">
                <div className="flex items-center justify-between border-b border-slate-100 pb-1">
                  <span className="font-medium text-slate-800">Anita Sharma</span>
                  <span className="text-[11px] text-slate-500">Mathematics</span>
                </div>
                <div className="flex items-center justify-between border-b border-slate-100 pb-1">
                  <span className="font-medium text-slate-800">Vijay Reddy</span>
                  <span className="text-[11px] text-slate-500">Science (Class Teacher 10-A)</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="font-medium text-slate-800">Sara Thomas</span>
                  <span className="text-[11px] text-slate-500">English</span>
                </div>
              </div>
            </div>

            {/* Academic Highlights */}
            <div className="rounded-xl border border-slate-200/80 bg-white p-3 shadow-2xs">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-900">
                <span>📚</span>
                <span>Active Modules</span>
              </div>
              <p className="mt-1.5 text-xs text-slate-600 leading-normal">
                Timetable (Monday &amp; Tuesday), Grade 10 Mid Term Exam 2026, Homework, Assignments, and Attendance Tracking.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Footer / Reset Action */}
      <div className="border-t border-slate-100 p-3">
        <button
          onClick={onReset}
          className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-100 hover:text-slate-900 active:scale-[0.99]"
        >
          <span>🔄</span>
          <span>New Conversation / Reset</span>
        </button>
      </div>
    </aside>
  );
}
