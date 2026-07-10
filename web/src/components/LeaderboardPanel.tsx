import { useEffect, useState } from "react";
import * as client from "../api/client";
import type { LeaderboardRow } from "../api/types";

interface LeaderboardPanelProps {
  threadId: string | null;
}

export function LeaderboardPanel({ threadId }: LeaderboardPanelProps) {
  const [active, setActive] = useState<string | null>(null);
  const [rows, setRows] = useState<LeaderboardRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    if (!threadId || loading) return;
    setLoading(true);
    try {
      const result = await client.getLeaderboard(threadId);
      setActive(result.active);
      setRows(result.leaderboard);
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setActive(null);
    setRows([]);
    setError(null);
    if (threadId) void refresh();
  }, [threadId]);

  const columns = rows.length > 0 ? Object.keys(rows[0]) : [];

  return (
    <aside className="leaderboard">
      <div className="panel-header">
        <div>
          <h2>Leaderboard</h2>
          <p>{active ? `Active model: ${active}` : "No active model"}</p>
        </div>
        <button type="button" onClick={refresh} disabled={!threadId || loading}>
          Refresh
        </button>
      </div>
      {error ? <p className="control-error">{error}</p> : null}
      {rows.length === 0 ? (
        <p className="empty-leaderboard">No models trained yet</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {columns.map((column) => (
                  <th key={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={index}>
                  {columns.map((column) => (
                    <td key={column}>{row[column]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </aside>
  );
}
