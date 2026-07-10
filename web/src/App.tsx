import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import * as client from "./api/client";
import { AutonomyToggle } from "./components/AutonomyToggle";
import { ChatView } from "./components/ChatView";
import { SessionSidebar } from "./components/SessionSidebar";
import { initialSession, Pending, sessionReducer } from "./state/useSession";
import "./styles.css";

function snapshotPending(items: { interrupt: string; [k: string]: unknown }[]): Pending[] {
  return items.map((item) => ({
    interrupt: item.interrupt,
    tool: String(item.tool ?? ""),
    detail: String(item.detail ?? ""),
  }));
}

export default function App() {
  const [state, dispatch] = useReducer(sessionReducer, initialSession);
  const [threadId, setThreadId] = useState<string | null>(
    () => localStorage.getItem("sentinel.thread"),
  );
  const [refreshKey, setRefreshKey] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const streamAbort = useRef<AbortController | null>(null);

  const abortStream = useCallback(() => {
    streamAbort.current?.abort();
    streamAbort.current = null;
  }, []);

  const selectThread = useCallback(
    async (tid: string) => {
      abortStream();
      try {
        const snap = await client.getSnapshot(tid);
        dispatch({ type: "hydrate", messages: snap.messages });
        dispatch({ type: "setPending", pending: snapshotPending(snap.pending_confirmations) });
        setThreadId(tid);
        localStorage.setItem("sentinel.thread", tid);
        setLoadError(null);
      } catch (err) {
        dispatch({ type: "reset" });
        setThreadId(null);
        localStorage.removeItem("sentinel.thread");
        setLoadError(String(err));
      }
    },
    [abortStream],
  );

  useEffect(() => {
    if (threadId) void selectThread(threadId);
  }, []);

  useEffect(() => abortStream, [abortStream]);

  function newSession() {
    abortStream();
    dispatch({ type: "reset" });
    setThreadId(null);
    localStorage.removeItem("sentinel.thread");
    setLoadError(null);
  }

  return (
    <main className="app-shell">
      <SessionSidebar
        activeThreadId={threadId}
        refreshKey={refreshKey}
        onSelect={selectThread}
        onNew={newSession}
      />
      <div className="main-column">
        <header className="app-header">
          <div>
            <strong>{threadId ? "Active session" : "New session"}</strong>
            {threadId ? <span>{threadId.slice(0, 8)}</span> : null}
          </div>
          <AutonomyToggle threadId={threadId} />
        </header>
        {loadError ? <p className="load-error">{loadError}</p> : null}
        <ChatView
          state={state}
          dispatch={dispatch}
          threadId={threadId}
          setThreadId={setThreadId}
          streamAbort={streamAbort}
          onSessionStarted={() => setRefreshKey((key) => key + 1)}
          onTurnFinished={() => setRefreshKey((key) => key + 1)}
        />
      </div>
    </main>
  );
}
