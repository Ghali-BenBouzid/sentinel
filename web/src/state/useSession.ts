import type { SentinelEvent, TranscriptMessage } from "../api/types";

export interface Bubble {
  id: string;
  kind: "user" | "agent" | "tool_call" | "tool_result" | "error";
  text: string;
  name?: string;
}

export interface Pending {
  interrupt: string;
  tool: string;
  detail: string;
}

export interface SessionState {
  transcript: Bubble[];
  pending: Pending[];
  streaming: boolean;
}

export type SessionAction =
  | { type: "start" }
  | { type: "reset" }
  | { type: "user"; text: string }
  | { type: "event"; event: SentinelEvent }
  | { type: "hydrate"; messages: TranscriptMessage[] }
  | { type: "setPending"; pending: Pending[] }
  | { type: "resolve"; interrupt: string };

export const initialSession: SessionState = {
  transcript: [],
  pending: [],
  streaming: false,
};

let seq = 0;
const nextId = () => `b${seq++}`;

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
          { id: nextId(), kind: "user", text: action.text },
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
        })),
      };
    case "setPending":
      return { ...state, pending: action.pending };
    case "resolve":
      return {
        ...state,
        pending: state.pending.filter((p) => p.interrupt !== action.interrupt),
      };
    case "event":
      return applyEvent(state, action.event);
  }
}

function applyEvent(state: SessionState, ev: SentinelEvent): SessionState {
  switch (ev.event) {
    case "message":
      return push(state, { kind: "agent", text: (ev.data as { text: string }).text });
    case "tool_call": {
      const d = ev.data as { name: string; args: unknown };
      return push(state, { kind: "tool_call", text: JSON.stringify(d.args), name: d.name });
    }
    case "tool_result": {
      const d = ev.data as { name: string; text: string };
      return push(state, { kind: "tool_result", text: d.text, name: d.name });
    }
    case "error":
      return {
        ...push(state, { kind: "error", text: (ev.data as { message: string }).message }),
        streaming: false,
      };
    case "confirm": {
      // Terminal: emitted in place of `done`, so also clear streaming.
      const d = ev.data as { tool: string; detail: string; interrupt: string };
      return {
        ...state,
        streaming: false,
        pending: [
          ...state.pending,
          { interrupt: d.interrupt, tool: d.tool, detail: d.detail },
        ],
      };
    }
    case "done":
      return { ...state, streaming: false };
    default:
      // Custom/progress events (notify, auto_approved, stage, model_training,
      // model_trained, ...) - ignored here; a progress line consumes them in the view.
      return state;
  }
}

function push(state: SessionState, b: Omit<Bubble, "id">): SessionState {
  return { ...state, transcript: [...state.transcript, { id: nextId(), ...b }] };
}
