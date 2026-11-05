import type { ChatResponse, HealthResponse, SchemaSummaryResponse } from "../types";

const BASE =
  import.meta.env.VITE_API_BASE ||
  (typeof window !== "undefined" && window.location.port === "5173"
    ? "http://127.0.0.1:8000"
    : "");

async function request<T>(path: string, init?: RequestInit, timeoutMs = 300_000): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
      signal: ctrl.signal,
    });
    if (!resp.ok) {
      let detail = `${resp.status} ${resp.statusText}`;
      try {
        const body = await resp.json();
        if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    return (await resp.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  health: () => request<HealthResponse>("/api/health", {}, 20_000),
  schemaSummary: () => request<SchemaSummaryResponse>("/api/schema-summary", {}, 20_000),
  chat: (sessionId: string, message: string) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, message }),
    }),
  clearSession: (sessionId: string) =>
    request<{ cleared: boolean }>(`/api/chat/${encodeURIComponent(sessionId)}`, { method: "DELETE" }, 20_000),
};
