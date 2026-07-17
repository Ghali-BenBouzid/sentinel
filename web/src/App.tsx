import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import * as client from "./api/client";
import { ChatView } from "./components/ChatView";
import { LeaderboardPanel } from "./components/LeaderboardPanel";
import { SessionSidebar } from "./components/SessionSidebar";
import { Icon } from "./components/Icon";
import { initialSession, sessionReducer } from "./state/useSession";
import type { Pending } from "./state/useSession";
import "./styles.css";

function snapshotPending(items: { interrupt: string; [k: string]: unknown }[]): Pending[] {
  return items.map((item) => ({
    interrupt: item.interrupt,
    tool: String(item.tool ?? ""),
    detail: String(item.detail ?? ""),
    decision: null,
  }));
}

export default function App() {
  const [state, dispatch] = useReducer(sessionReducer, initialSession);
  const [threadId, setThreadId] = useState<string | null>(
    () => localStorage.getItem("sentinel.thread"),
  );
  const [refreshKey, setRefreshKey] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const streamAbort = useRef<AbortController | null>(null);
  const initialThreadId = useRef(threadId);
  const restoredInitialThread = useRef(false);

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
    if (restoredInitialThread.current) return;
    restoredInitialThread.current = true;
    if (initialThreadId.current) {
      void selectThread(initialThreadId.current);
    }
  }, [selectThread]);

  useEffect(() => abortStream, [abortStream]);

  function newSession() {
    abortStream();
    dispatch({ type: "reset" });
    setThreadId(null);
    localStorage.removeItem("sentinel.thread");
    setLoadError(null);
  }

  return (
    <main className={`app-shell ${leftCollapsed ? "left-collapsed" : ""} ${rightCollapsed ? "right-collapsed" : ""}`}>
      <SessionSidebar
        activeThreadId={threadId}
        refreshKey={refreshKey}
        collapsed={leftCollapsed}
        onToggle={() => setLeftCollapsed((value) => !value)}
        onSelect={selectThread}
        onNew={newSession}
      />
      <div className="main-column">
        <header className="app-header">
          <div className="header-title">
            <strong>Agent Orchestrator</strong>
            <span className="live-status"><i />{threadId ? "Connected" : "Ready for a new session"}</span>
          </div>
          <div className="header-actions">
            <button className="icon-button" type="button" onClick={() => setRightCollapsed((value) => !value)} aria-label="Toggle operations panel"><Icon name="panel" size={15} /></button>
          </div>
        </header>
        {loadError ? <p className="load-error">{loadError}</p> : null}
        <div className="workspace">
          <ChatView
            state={state}
            dispatch={dispatch}
            threadId={threadId}
            setThreadId={setThreadId}
            streamAbort={streamAbort}
            onSessionStarted={() => setRefreshKey((key) => key + 1)}
            onTurnFinished={() => setRefreshKey((key) => key + 1)}
          />
          {!rightCollapsed && <LeaderboardPanel threadId={threadId} state={state} />}
        </div>
      </div>
      {rightCollapsed && <button className="edge-handle edge-handle-right" type="button" onClick={() => setRightCollapsed(false)} aria-label="Expand operations panel" />}
    </main>
  );
}
