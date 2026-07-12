import { useCallback, useEffect, useState } from "react";
import * as client from "../api/client";
import type { LeaderboardRow } from "../api/types";
import type { SessionState } from "../state/useSession";
import { Icon } from "./Icon";

interface LeaderboardPanelProps {
  threadId: string | null;
  state: SessionState;
}

export function LeaderboardPanel({ threadId, state }: LeaderboardPanelProps) {
  const [active, setActive] = useState<string | null>(null);
  const [rows, setRows] = useState<LeaderboardRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    if (!threadId) return;
    try {
      const result = await client.getLeaderboard(threadId);
      setActive(result.active);
      setRows(result.leaderboard);
      setError(null);
      setUpdatedAt(new Date());
    } catch (err) {
      setError(String(err));
    }
  }, [threadId]);

  useEffect(() => {
    setActive(null);
    setRows([]);
    setError(null);
    setExpanded(false);
    setUpdatedAt(null);
  }, [threadId]);

  useEffect(() => {
    if (threadId && !state.streaming) void refresh();
  }, [threadId, refresh, state.streaming]);

  useEffect(() => {
    if (!threadId) return;
    const timer = window.setInterval(() => void refresh(), 15_000);
    return () => window.clearInterval(timer);
  }, [threadId, refresh]);

  const errorMessages = state.transcript.filter((item) => item.kind === "error").slice(-3);
  const activityStatus = state.streaming ? "Running" : state.pending.length ? "Awaiting input" : "Idle";

  return (
    <aside className="operations-panel">
      <Panel title="Active dataset" icon="database">
        <div className="empty-panel-value">
          <strong>No dataset metadata</strong>
          <span>The current API does not expose active dataset details.</span>
        </div>
      </Panel>

      <Panel title="Agent activity" icon="status" badge={activityStatus} badgeTone={state.streaming ? "accent" : state.pending.length ? "warning" : "neutral"}>
        <p className="activity-label">{state.activity.label}</p>
        <div className="progress-row">
          <div className="progress-track"><span style={{ width: `${state.activity.progress}%` }} /></div>
          <code>{state.activity.progress}%</code>
        </div>
        <p className="eyebrow card-eyebrow">Recent actions</p>
        {state.activity.recent.length ? (
          <div className="recent-actions">
            {state.activity.recent.map((item, index) => (
              <div key={`${item.label}-${index}`}>
                <span className={item.status === "done" ? "action-done" : "action-running"} />
                <span>{item.label}</span>
              </div>
            ))}
          </div>
        ) : <p className="panel-muted">No actions in this session</p>}
      </Panel>

      <Panel
        title="Model leaderboard"
        icon="chart"
        action={
          rows.length ? (
            <button
              className="panel-link"
              type="button"
              onClick={() => setExpanded((value) => !value)}
            >
              {expanded ? "Show summary" : "View full leaderboard"}
            </button>
          ) : undefined
        }
      >
        {error ? <p className="control-error">{error}</p> : null}
        {rows.length && !expanded ? (
          <div className="compact-leaderboard">
            {rows.slice(0, 5).map((row, index) => {
              const entries = Object.entries(row);
              const name = String(row.Model ?? row.model ?? entries[0]?.[1] ?? "Model");
              const metric = entries.find(([key]) => /rmse|mae|f1|score/i.test(key));
              return (
                <div key={`${name}-${index}`}>
                  <code>{index + 1}</code>
                  <strong>{index === 0 && <span className="star">★</span>}{name}</strong>
                  <code>{metric ? `${metric[0]} ${metric[1]}` : ""}</code>
                </div>
              );
            })}
            {active && <p className="active-model">Active model <code>{active}</code></p>}
          </div>
        ) : null}
        {rows.length && expanded ? (
          <div className="full-leaderboard-wrap">
            <table className="full-leaderboard">
              <thead>
                <tr>
                  <th>#</th>
                  {Object.keys(rows[0]).map((column) => <th key={column}>{column}</th>)}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr key={index}>
                    <td>{index + 1}</td>
                    {Object.keys(rows[0]).map((column) => (
                      <td key={column}>{String(row[column] ?? "")}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
        {!rows.length ? <p className="panel-muted">No models trained yet</p> : null}
        {updatedAt && (
          <div className="leaderboard-freshness">
            <span>Updates automatically every 15 seconds</span>
            <button type="button" onClick={() => void refresh()}>Refresh now</button>
          </div>
        )}
      </Panel>

      <Panel title="Alerts" icon="alert" badge={String(errorMessages.length)} badgeTone={errorMessages.length ? "critical" : "neutral"}>
        {errorMessages.length ? (
          <div className="alert-list">
            {errorMessages.map((item) => <div key={item.id}><Icon name="alert" size={13} /><span><strong>Agent failure</strong><small>{item.text}</small></span></div>)}
          </div>
        ) : <p className="panel-muted">No alerts in this session</p>}
      </Panel>
    </aside>
  );
}

function Panel({
  title,
  icon,
  badge,
  badgeTone = "neutral",
  action,
  children,
}: {
  title: string;
  icon: "database" | "status" | "chart" | "alert";
  badge?: string;
  badgeTone?: "neutral" | "accent" | "warning" | "critical";
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="operation-card">
      <header><strong><Icon name={icon} size={14} />{title}</strong>{badge && <span className={`status-pill ${badgeTone}`}>{badge}</span>}{action}</header>
      {children}
    </section>
  );
}
