import { useState } from "react";
import type { SqlQueryRecord } from "../types";

const GOOD = new Set(["Present", "Approved", "Pass", "Checked", "Graded", "Submitted", "ON LEAVE: false", true as unknown as string]);
const WARN = new Set(["Pending", "Late", "Half Day", ""]);
const BAD = new Set(["Absent", "Rejected", "Fail"]);

function statusClass(value: unknown): string {
  const s = String(value ?? "").trim();
  if (GOOD.has(s)) return "bg-emerald-100 text-emerald-800";
  if (WARN.has(s)) return "bg-amber-100 text-amber-800";
  if (BAD.has(s)) return "bg-rose-100 text-rose-800";
  return "bg-slate-100 text-slate-700";
}

export default function SqlResultTable({ query }: { query: SqlQueryRecord }) {
  const [showSql, setShowSql] = useState(false);
  const [copied, setCopied] = useState(false);

  const copySql = async () => {
    try {
      await navigator.clipboard.writeText(query.sql);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div className="mt-3 overflow-hidden rounded-lg border border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b border-slate-100 bg-slate-50 px-3 py-1.5 text-xs text-slate-500">
        <span>
          {query.row_count} row{query.row_count === 1 ? "" : "s"} · {query.execution_time_ms} ms
        </span>
        <button
          onClick={() => setShowSql((v) => !v)}
          className="rounded px-2 py-0.5 font-medium text-indigo-600 hover:bg-indigo-50"
        >
          {showSql ? "Hide SQL" : "View Executed SQL"}
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              {query.columns.map((c) => (
                <th key={c} className="whitespace-nowrap px-3 py-2 font-semibold">
                  {c.replace(/_/g, " ")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {query.rows.map((row, i) => (
              <tr key={i} className={i % 2 ? "bg-slate-50/50" : ""}>
                {query.columns.map((c) => {
                  const v = row[c];
                  return (
                    <td key={c} className="whitespace-nowrap px-3 py-1.5">
                      {typeof v === "boolean" ? (
                        <span className={`inline-block rounded px-1.5 py-0.5 text-xs font-medium ${statusClass(v)}`}>
                          {v ? "Yes" : "No"}
                        </span>
                      ) : /[A-Za-z]/.test(String(v ?? "")) &&
                        (GOOD.has(String(v ?? "").trim()) ||
                          BAD.has(String(v ?? "").trim()) ||
                          WARN.has(String(v ?? "").trim())) ? (
                        <span className={`inline-block rounded px-1.5 py-0.5 text-xs font-medium ${statusClass(v)}`}>
                          {String(v)}
                        </span>
                      ) : (
                        String(v ?? "")
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
            {query.rows.length === 0 && (
              <tr>
                <td colSpan={Math.max(query.columns.length, 1)} className="px-3 py-4 text-center text-sm text-slate-400">
                  No rows returned (0 rows) - the database has no records for this criteria.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {showSql && (
        <div className="border-t border-slate-100 bg-slate-900 px-3 py-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Executed SQL (read-only)</span>
            <button onClick={copySql} className="rounded px-2 py-0.5 text-xs text-indigo-300 hover:bg-slate-800">
              {copied ? "Copied!" : "Copy"}
            </button>
          </div>
          <pre className="mt-1 overflow-x-auto whitespace-pre text-xs leading-relaxed text-emerald-200">
            <code>{query.sql}</code>
          </pre>
        </div>
      )}
    </div>
  );
}
