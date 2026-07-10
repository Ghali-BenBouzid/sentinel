import { useEffect, useState } from "react";
import * as client from "../api/client";
import type { SessionSummary } from "../api/types";

interface SessionSidebarProps {
  activeThreadId: string | null;
  refreshKey: number;
  onSelect: (threadId: string) => void;
  onNew: () => void;
}

export function SessionSidebar({
  activeThreadId,
  refreshKey,
  onSelect,
  onNew,
}: SessionSidebarProps) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let ignore = false;
    client
      .listSessions()
      .then((items) => {
        if (!ignore) {
          setSessions(items);
          setError(null);
        }
      })
      .catch((err) => {
        if (!ignore) setError(String(err));
      });
    return () => {
      ignore = true;
    };
  }, [refreshKey]);

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>Sentinel</h1>
        <button type="button" onClick={onNew}>
          New session
        </button>
      </div>
      {error ? <p className="sidebar-error">{error}</p> : null}
      {sessions.length === 0 ? (
        <p className="empty-sidebar">Start your first session</p>
      ) : (
        <nav className="session-list" aria-label="Sessions">
          {sessions.map((session) => (
            <button
              type="button"
              key={session.thread_id}
              className={session.thread_id === activeThreadId ? "active" : ""}
              onClick={() => onSelect(session.thread_id)}
            >
              <span>{session.last_message || "New conversation"}</span>
              <small>{session.autonomy || "unknown"}</small>
            </button>
          ))}
        </nav>
      )}
    </aside>
  );
}
