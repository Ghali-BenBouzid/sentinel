import type { LeaderboardRow, SessionSummary, Snapshot } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function json<T>(resp: Response): Promise<T> {
  if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
  return resp.json() as Promise<T>;
}

// Validate before a caller pipes the body through parseSSEStream.
async function streamFetch(
  url: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<Response> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  const ct = response.headers.get("content-type") ?? "";
  if (!ct.includes("text/event-stream")) {
    throw new Error(`expected event-stream, got ${ct}`);
  }
  return response;
}

export async function startSession(
  message: string,
  autonomy?: string,
  signal?: AbortSignal,
) {
  const response = await streamFetch(
    `${BASE}/sessions`,
    { message, ...(autonomy ? { autonomy } : {}) },
    signal,
  );
  return { threadId: response.headers.get("x-thread-id")!, response };
}

export function sendMessage(threadId: string, message: string, signal?: AbortSignal) {
  return streamFetch(`${BASE}/sessions/${threadId}/message`, { message }, signal);
}

export function resume(
  threadId: string,
  answers: Record<string, string>,
  signal?: AbortSignal,
) {
  return streamFetch(`${BASE}/sessions/${threadId}/resume`, { answers }, signal);
}

export function getSnapshot(threadId: string) {
  return fetch(`${BASE}/sessions/${threadId}`).then(json<Snapshot>);
}

export function listSessions() {
  return fetch(`${BASE}/sessions`)
    .then(json<{ sessions: SessionSummary[] }>)
    .then((d) => d.sessions);
}

export function setAutonomy(threadId: string, autonomy: string) {
  return fetch(`${BASE}/sessions/${threadId}/autonomy`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ autonomy }),
  }).then(json<{ autonomy: string }>);
}

export function getLeaderboard(threadId: string) {
  return fetch(`${BASE}/sessions/${threadId}/leaderboard`).then(
    json<{ active: string | null; leaderboard: LeaderboardRow[] }>,
  );
}
