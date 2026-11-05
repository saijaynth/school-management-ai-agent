import { useState } from "react";
import type { HealthResponse, SchemaSummaryResponse } from "../types";
import { STARTER_PROMPTS } from "../types";
import {
  ChevronDownIcon,
  ChevronRightIcon,
  CloseIcon,
  GraduationCapIcon,
  RefreshIcon,
  UserGroupIcon,
  UserIcon,
} from "./Icons";

type Tab = "prompts" | "directory";

export default function Sidebar({
  health,
  summary: _summary,
  onPick,
  onReset,
  isOpen = false,
  onClose,
}: {
  health: HealthResponse | null;
  summary?: SchemaSummaryResponse | null;
  onPick: (q: string) => void;
  onReset: () => void;
  isOpen?: boolean;
  onClose?: () => void;
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
    <>
      {/* Mobile Backdrop Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/40 backdrop-blur-xs transition-opacity lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-50 flex h-full w-72 max-w-[85vw] flex-col border-r border-slate-200 bg-white shadow-xl transition-transform duration-200 ease-in-out lg:static lg:w-72 lg:min-w-72 lg:shadow-none lg:translate-x-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Header */}
        <div className="border-b border-slate-100 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 bg-slate-50 text-slate-800">
                <GraduationCapIcon className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-sm font-semibold text-slate-900 tracking-tight leading-tight">School Records</h2>
                <p className="text-[11px] text-slate-500">Academic Assistant</p>
              </div>
            </div>

            {onClose && (
              <button
                onClick={onClose}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 hover:bg-slate-100 hover:text-slate-600 lg:hidden cursor-pointer"
                title="Close drawer"
                aria-label="Close drawer"
              >
                <CloseIcon className="h-4 w-4" />
              </button>
            )}
          </div>

          {/* Status Indicator */}
          <div className="mt-3 flex items-center gap-2 rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] text-slate-600">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                health?.database.connected ? "bg-emerald-600" : "bg-amber-500"
              }`}
            />
            <span>{health?.database.connected ? "Database connected" : "Connecting..."}</span>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-slate-100 bg-slate-50/50 text-xs font-medium">
          <button
            onClick={() => setTab("prompts")}
            className={`flex-1 py-2.5 text-center transition ${
              tab === "prompts"
                ? "border-b-2 border-slate-900 bg-white text-slate-900 font-semibold"
                : "text-slate-500 hover:text-slate-700"
            }`}
          >
            Queries
          </button>
          <button
            onClick={() => setTab("directory")}
            className={`flex-1 py-2.5 text-center transition ${
              tab === "directory"
                ? "border-b-2 border-slate-900 bg-white text-slate-900 font-semibold"
                : "text-slate-500 hover:text-slate-700"
            }`}
          >
            Directory
          </button>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2.5">
          {tab === "prompts" &&
            STARTER_PROMPTS.map((g) => (
              <div key={g.group} className="rounded-lg border border-slate-100 bg-slate-50/40 p-2">
                <button
                  onClick={() => toggleGroup(g.group)}
                  className="flex w-full items-center justify-between px-1.5 py-1 text-left text-xs font-medium text-slate-800 hover:text-slate-900"
                >
                  <span>{g.group}</span>
                  <span className="text-slate-400">
                    {openGroups[g.group] ? (
                      <ChevronDownIcon className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronRightIcon className="h-3.5 w-3.5" />
                    )}
                  </span>
                </button>
                {openGroups[g.group] && (
                  <div className="mt-1 space-y-1">
                    {g.items.map((q) => (
                      <button
                        key={q}
                        onClick={() => onPick(q)}
                        className="block w-full rounded-md border border-slate-200/60 bg-white px-2.5 py-1.5 text-left text-xs text-slate-700 shadow-2xs transition hover:border-slate-300 hover:bg-slate-50 hover:text-slate-900 active:scale-[0.99] cursor-pointer"
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
              {/* Enrolled Students */}
              <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-2xs">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-900 mb-2">
                  <div className="flex items-center gap-1.5">
                    <UserGroupIcon className="h-4 w-4 text-slate-500" />
                    <span>Enrolled students</span>
                  </div>
                  <span className="tabular-nums font-mono text-[11px] text-slate-500">4 total</span>
                </div>
                <div className="space-y-1.5 text-xs">
                  <div className="bg-slate-50 rounded p-1.5">
                    <div className="font-medium text-slate-800 text-[11px]">Grade 9-A</div>
                    <div className="text-[11px] text-slate-600">Arjun Rao, Meera Iyer</div>
                  </div>
                  <div className="bg-slate-50 rounded p-1.5">
                    <div className="font-medium text-slate-800 text-[11px]">Grade 10-A</div>
                    <div className="text-[11px] text-slate-600">Divya Nair, Kiran Patel</div>
                  </div>
                </div>
              </div>

              {/* Faculty */}
              <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-2xs">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-900 mb-2">
                  <div className="flex items-center gap-1.5">
                    <UserIcon className="h-4 w-4 text-slate-500" />
                    <span>Faculty</span>
                  </div>
                  <span className="tabular-nums font-mono text-[11px] text-slate-500">3 total</span>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between py-1 border-b border-slate-100">
                    <span className="font-medium text-slate-800 text-[11px]">Anita Sharma</span>
                    <span className="text-[11px] text-slate-500">Math</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-100">
                    <span className="font-medium text-slate-800 text-[11px]">Vijay Reddy</span>
                    <span className="text-[11px] text-slate-500">Science</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="font-medium text-slate-800 text-[11px]">Sara Thomas</span>
                    <span className="text-[11px] text-slate-500">English</span>
                  </div>
                </div>
              </div>

              {/* Modules */}
              <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-2.5 text-xs text-slate-600 leading-relaxed">
                <p className="font-medium text-slate-900 text-[11px] mb-1">Active academic term</p>
                <p className="text-[11px] text-slate-600">
                  Academic Year 2026-2027 • Grade 10 Mid Term Examination • Weekly schedule (Mon–Tue).
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-slate-100 p-3">
          <button
            onClick={onReset}
            className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-700 transition hover:bg-slate-100 hover:text-slate-900 active:scale-[0.99] cursor-pointer"
          >
            <RefreshIcon className="h-3.5 w-3.5 text-slate-500" />
            <span>Reset session</span>
          </button>
        </div>
      </aside>
    </>
  );
}
