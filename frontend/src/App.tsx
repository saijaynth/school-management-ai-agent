import { useCallback, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import { api } from "./api/chatApi";
import type { ChatMessage, HealthResponse, SchemaSummaryResponse } from "./types";

const SESSION_KEY = "school-ai-session-id";

export default function App() {
  const [sessionId, setSessionId] = useState<string>(() => sessionStorage.getItem(SESSION_KEY) ?? "");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [summary, setSummary] = useState<SchemaSummaryResponse | null>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    const checkStatus = () => {
      api
        .health()
        .then((h) => {
          setHealth(h);
          if (!summary && h?.status === "ok") {
            api.schemaSummary().then(setSummary).catch(() => {});
          }
        })
        .catch(() => setHealth(null));
    };

    checkStatus();
    timer = setInterval(checkStatus, 3000);
    return () => clearInterval(timer);
  }, [summary]);

  const send = useCallback(
    async (text: string) => {
      setError(null);
      setMessages((m) => [...m, { role: "user", content: text }]);
      setLoading(true);
      try {
        let sid = sessionId;
        if (!sid) {
          sid = "web-" + Math.random().toString(36).slice(2, 10);
          setSessionId(sid);
          sessionStorage.setItem(SESSION_KEY, sid);
        }
        const resp = await api.chat(sid, text);
        setSessionId(resp.session_id);
        sessionStorage.setItem(SESSION_KEY, resp.session_id);
        setMessages((m) => [
          ...m,
          {
            role: "assistant",
            content: resp.answer,
            thinking: resp.thinking,
            anytime: {
              source_type: resp.source_type,
              sql_queries: resp.sql_queries,
              manual_references: resp.manual_references,
              suggested_followups: resp.suggested_followups,
            },
          },
        ]);
      } catch (exc) {
        setError((exc as Error).message || "Network error");
        setMessages((m) => [
          ...m,
          { role: "assistant", content: "The request failed. See the error banner below.", failed: true },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [sessionId],
  );

  const reset = useCallback(async () => {
    const sid = sessionId;
    setMessages([]);
    setSessionId("");
    sessionStorage.removeItem(SESSION_KEY);
    if (sid) {
      try {
        await api.clearSession(sid);
      } catch {
        /* backend may be down; UI clearing still applies */
      }
    }
  }, [sessionId]);

  const retry = useCallback(() => {
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (lastUser) {
      setMessages((m) => {
        const copy = [...m];
        while (copy.length && copy[copy.length - 1].failed) copy.pop();
        // also drop the trailing user message so send() re-adds it
        if (copy.length && copy[copy.length - 1].role === "user") copy.pop();
        return copy;
      });
      void send(lastUser.content);
    }
  }, [messages, send]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50 text-slate-900 antialiased">
      <Sidebar health={health} summary={summary} onPick={send} onReset={reset} />
      <ChatWindow
        sessionId={sessionId}
        messages={messages}
        loading={loading}
        error={error}
        onSend={send}
        onRetry={retry}
        health={health}
        summary={summary}
      />
    </div>
  );
}
