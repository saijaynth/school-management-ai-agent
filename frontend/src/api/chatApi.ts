import type { ChatResponse, HealthResponse, SchemaSummaryResponse } from "../types";

// Determine the most reliable base URL for API requests
function getBaseCandidates(): string[] {
  if (import.meta.env.VITE_API_BASE) {
    return [import.meta.env.VITE_API_BASE];
  }
  if (typeof window !== "undefined") {
    // If the frontend is served directly by the backend on port 8000
    if (window.location.port === "8000" || !window.location.port) {
      return ["", "http://127.0.0.1:8000"];
    }
    // If running in development (e.g. port 5173, 5174, 3000)
    return ["http://127.0.0.1:8000", "http://localhost:8000", ""];
  }
  return ["http://127.0.0.1:8000"];
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 300_000): Promise<T> {
  const candidates = getBaseCandidates();
  let lastError: Error | null = null;

  for (let i = 0; i < candidates.length; i++) {
    const base = candidates[i];
    const url = `${base}${path}`;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);

    try {
      const resp = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...init,
        signal: ctrl.signal,
      });

      clearTimeout(timer);

      if (!resp.ok) {
        let detail = `${resp.status} ${resp.statusText}`;
        try {
          const body = await resp.json();
          if (body?.detail) {
            detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
          }
        } catch {
          /* ignore */
        }
        throw new Error(detail);
      }

      return (await resp.json()) as T;
    } catch (err: any) {
      clearTimeout(timer);
      lastError = err;
      // If it's an HTTP error (4xx/5xx with status detail), don't retry other hosts
      if (err?.message && !err.message.includes("Failed to fetch") && !err.message.includes("NetworkError") && !err.message.includes("aborted")) {
        throw err;
      }
      // If network failed, continue loop to try next candidate
    }
  }

  throw lastError || new Error("Failed to reach backend API. Make sure the backend is running on port 8000.");
}

export const api = {
  health: () => request<HealthResponse>("/api/health", {}, 10_000),
  schemaSummary: () => request<SchemaSummaryResponse>("/api/schema-summary", {}, 15_000),
  chat: (sessionId: string, message: string) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, message }),
    }),
  clearSession: (sessionId: string) =>
    request<{ cleared: boolean }>(`/api/chat/${encodeURIComponent(sessionId)}`, { method: "DELETE" }, 15_000),
};
