// Verified against sentinel/api/app.py `_sse` + `_stream`.
export type SentinelEvent =
  | { event: "message"; data: { text: string } }
  | { event: "tool_call"; data: { name: string; args: Record<string, unknown> } }
  | { event: "tool_result"; data: { name: string; text: string } }
  // confirm carries the interrupt value spread + its id; the action name is `tool`.
  | { event: "confirm"; data: { type: "confirm"; tool: string; detail: string; interrupt: string } }
  | { event: "error"; data: { message: string } }
  | { event: "done"; data: Record<string, never> }
  // Custom graph events forwarded by their `type`: notify, auto_approved,
  // stage, model_training, model_trained, and any future one. Never crash on these.
  | { event: string; data: Record<string, unknown> };

export const TERMINAL_EVENTS = ["done", "confirm", "error"] as const;

export interface SessionSummary {
  thread_id: string;
  autonomy: string | null;
  last_message: string | null;
}

export interface TranscriptMessage {
  role: "user" | "agent" | "tool_result";
  content: string;
  name?: string;
  tool_calls?: { name: string; args: Record<string, unknown> }[];
}

export interface Snapshot {
  autonomy: string | null;
  pending_confirmations: { interrupt: string; [k: string]: unknown }[];
  last_message: string | null;
  messages: TranscriptMessage[];
}

export interface LeaderboardRow {
  [column: string]: string | number;
}
