import type { SentinelEvent, TranscriptMessage } from "../api/types";

export interface Bubble {
  id: string;
  kind: "user" | "agent" | "tool_call" | "tool_result" | "error";
  text: string;
  name?: string;
  time?: string;
}

export interface ActivityItem {
  label: string;
  status: "done" | "running";
}

export interface AgentActivity {
  label: string;
  progress: number;
  recent: ActivityItem[];
}

export interface Pending {
  interrupt: string;
  tool: string;
  detail: string;
  decision: "yes" | "no" | null;
}

export interface SessionState {
  transcript: Bubble[];
  pending: Pending[];
  streaming: boolean;
  activity: AgentActivity;
}

export type SessionAction =
  | { type: "start" }
  | { type: "reset" }
  | { type: "user"; text: string }
  | { type: "event"; event: SentinelEvent }
  | { type: "hydrate"; messages: TranscriptMessage[] }
  | { type: "setPending"; pending: Pending[] }
  | { type: "choose"; interrupt: string; decision: "yes" | "no" }
  | { type: "resolveAll" };

export const initialSession: SessionState = {
  transcript: [],
  pending: [],
  streaming: false,
  activity: { label: "Ready for a task", progress: 0, recent: [] },
};

let seq = 0;
const nextId = () => `b${seq++}`;
const now = () =>
  new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date());

export function sessionReducer(
  state: SessionState,
  action: SessionAction,
): SessionState {
  switch (action.type) {
    case "start":
      return { ...state, streaming: true };
    case "reset":
      return initialSession;
    case "user":
      return {
        ...state,
        transcript: [
          ...state.transcript,
          { id: nextId(), kind: "user", text: action.text, time: now() },
        ],
      };
    case "hydrate":
      return {
        ...state,
        transcript: action.messages.map((m) => ({
          id: nextId(),
          kind: m.role === "agent" ? "agent" : m.role,
          text: m.content,
          name: m.name,
          time: "",
        })),
      };
    case "setPending":
      return { ...state, pending: action.pending };
    case "choose":
      return {
        ...state,
        pending: state.pending.map((p) =>
          p.interrupt === action.interrupt ? { ...p, decision: action.decision } : p,
        ),
      };
    case "resolveAll":
      return { ...state, pending: [] };
    case "event":
      return applyEvent(state, action.event);
  }
}

function applyEvent(state: SessionState, ev: SentinelEvent): SessionState {
  switch (ev.event) {
    case "message":
      return push(state, {
        kind: "agent",
        text: (ev.data as { text: string }).text,
        time: now(),
      });
    case "tool_call": {
      const d = ev.data as { name: string; args: unknown };
      return {
        ...push(state, {
          kind: "tool_call",
          text: JSON.stringify(d.args),
          name: d.name,
          time: now(),
        }),
        activity: activity(state, d.name, "running"),
      };
    }
    case "tool_result": {
      const d = ev.data as { name: string; text: string };
      return {
        ...push(state, { kind: "tool_result", text: d.text, name: d.name, time: now() }),
        activity: activity(state, d.name, "done"),
      };
    }
    case "error":
      return {
        ...push(state, {
          kind: "error",
          text: (ev.data as { message: string }).message,
          time: now(),
        }),
        streaming: false,
        activity: { ...state.activity, label: "Operation failed" },
      };
    case "confirm": {
      // Terminal: emitted in place of `done`, so also clear streaming.
      const d = ev.data as { tool: string; detail: string; interrupt: string };
      return {
        ...state,
        streaming: false,
        pending: [
          ...state.pending,
          { interrupt: d.interrupt, tool: d.tool, detail: d.detail, decision: null },
        ],
      };
    }
    case "done":
      return {
        ...state,
        streaming: false,
        activity: { ...state.activity, label: "Task complete", progress: 100 },
      };
    case "stage": {
      const data = ev.data as { text?: string };
      return {
        ...state,
        activity: { ...state.activity, label: data.text || "Working" },
      };
    }
    case "model_training": {
      const data = ev.data as { name?: string; index?: number; total?: number };
      const total = Math.max(data.total || 1, 1);
      return {
        ...state,
        activity: {
          label: `Training ${data.name || "model"}`,
          progress: Math.round((((data.index || 1) - 1) / total) * 100),
          recent: state.activity.recent,
        },
      };
    }
    case "model_trained": {
      const data = ev.data as { name?: string; index?: number; total?: number };
      const total = Math.max(data.total || 1, 1);
      return {
        ...state,
        activity: {
          label: `Trained ${data.name || "model"}`,
          progress: Math.round(((data.index || 1) / total) * 100),
          recent: [
            { label: `Trained ${data.name || "model"}`, status: "done" as const },
            ...state.activity.recent,
          ].slice(0, 4),
        },
      };
    }
    default:
      // Custom/progress events (notify, auto_approved, stage, model_training,
      // model_trained, ...) - ignored here; a progress line consumes them in the view.
      return state;
  }
}

function push(state: SessionState, b: Omit<Bubble, "id">): SessionState {
  return { ...state, transcript: [...state.transcript, { id: nextId(), ...b }] };
}

function activity(
  state: SessionState,
  label: string,
  status: "done" | "running",
): AgentActivity {
  return {
    label: status === "done" ? `Completed ${label}` : `Running ${label}`,
    progress: status === "done" ? 100 : state.activity.progress,
    recent: [
      { label, status },
      ...state.activity.recent.filter((item) => item.label !== label),
    ].slice(0, 4),
  };
}
