import { useEffect, useState } from "react";
import * as client from "../api/client";
import type { SessionSummary } from "../api/types";
import { Icon } from "./Icon";

interface SessionSidebarProps {
  activeThreadId: string | null;
  refreshKey: number;
  collapsed: boolean;
  onToggle: () => void;
  onSelect: (threadId: string) => void;
  onNew: () => void;
}

const nav = [
  ["agent", "Agent", "agent"],
  ["datasets", "Datasets", "database"],
  ["experiments", "Experiments", "experiment"],
  ["models", "Models", "model"],
  ["predictions", "Predictions", "chart"],
  ["monitoring", "Monitoring", "monitor"],
  ["alerts", "Alerts", "alert"],
] as const;

export function SessionSidebar({
  activeThreadId,
  refreshKey,
  collapsed,
  onToggle,
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
    <aside className={`sidebar ${collapsed ? "is-collapsed" : ""}`}>
      <div className="brand-row">
        <button className="brand" type="button" onClick={onNew} aria-label="New session">
          <span className="brand-mark"><img src="/brand/sentinel-mark-reversed.svg" alt="" width={16} height={16} /></span>
          {!collapsed && <span className="brand-name">SENTINEL</span>}
        </button>
        {!collapsed && (
          <button className="icon-button rail-toggle" type="button" onClick={onToggle} aria-label="Collapse navigation">
            <Icon name="chevron" size={14} />
          </button>
        )}
      </div>

      <div className="sidebar-new-task">
        <button className="new-task-button" type="button" onClick={onNew} title="New task">
          <Icon name="plus" size={14} />
          {!collapsed && <span>New task</span>}
        </button>
      </div>

      <div className="sidebar-scroll">
        <nav className="primary-nav" aria-label="Primary navigation">
          {nav.map(([key, label, icon]) => (
            <button
              key={key}
              className={key === "agent" ? "active" : ""}
              type="button"
              title={key === "agent" ? label : `${label} page is planned`}
              disabled={key !== "agent"}
            >
              <Icon name={icon} />
              {!collapsed && <span>{label}</span>}
            </button>
          ))}
        </nav>

        {!collapsed && (
          <div className="history-block">
            <p className="eyebrow">Chat history</p>
            <p className="history-group">Recent</p>
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
                    title={session.title || session.last_message || "New conversation"}
                  >
                    {session.title || session.last_message || "New conversation"}
                  </button>
                ))}
              </nav>
            )}
          </div>
        )}
      </div>

      <div className="sidebar-footer">
        {(["integrations", "settings"] as const).map((name) => (
          <button type="button" key={name} disabled title={`${name[0].toUpperCase() + name.slice(1)} page is planned`}>
            <Icon name={name} />
            {!collapsed && <span>{name[0].toUpperCase() + name.slice(1)}</span>}
          </button>
        ))}
      </div>
      {collapsed && (
        <button className="edge-handle" type="button" onClick={onToggle} aria-label="Expand navigation" />
      )}
    </aside>
  );
}
