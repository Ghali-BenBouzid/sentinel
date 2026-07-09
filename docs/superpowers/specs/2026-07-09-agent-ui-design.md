# Sentinel Agent UI - Design

Date: 2026-07-09
Status: Approved (design), pending implementation plan
Author: Ghali (with Claude Code)

## Purpose

A web UI that drives the V2 agentic data scientist over its existing FastAPI/SSE backend (`sentinel/api/app.py`).
It replaces the retired Streamlit dashboard with a real single-page app that:

- Chats with the agent (streamed responses, visible tool calls and results).
- Confirms or denies guarded actions inline in the conversation.
- Views the model-comparison leaderboard.
- Toggles session autonomy (guarded / autonomous) mid-session.
- Switches between multiple sessions.

This document is the handoff spec.
Claude Code scaffolds the project and the backend additions; a Codex session then implements the remaining components against this spec.

## Architecture

Two layers change:

1. A small set of additions to the existing FastAPI app - no changes to the agent/graph internals.
2. A new `web/` React + Vite + TypeScript SPA that talks to that API.

The SPA runs on the Vite dev server (`:5173`) and calls the API (`:8000`) directly; CORS is opened for the dev origin.
There is no build-time coupling between the Python package and `web/`.

### Wire-format note

All streaming endpoints are `POST` returning `text/event-stream`, so the browser's native `EventSource` (GET-only) cannot be used.
The client hand-rolls SSE parsing over `fetch(...).body.getReader()`.
This is intrinsic to the existing API shape and applies to every approach.

## Chosen approach

Plain React with a hand-rolled SSE hook and component-local state (`useReducer`), no state-management or data-caching library.
Rejected alternatives:

- TanStack Query - adds a caching dependency for reads that are fetched once per view and otherwise pushed live over SSE.
  No cross-component cache or polling need exists.
- Zustand/Redux global store - the tree is two levels (sidebar + chat view); Context/props suffice.

The app pays for exactly the one hard problem it has (streaming SSE parsing) and nothing more.

## Backend additions

All in `sentinel/api/app.py` unless noted.
No agent/graph internals change.

### 1. `GET /sessions` - list sessions

Enumerates checkpointed threads via the `SqliteSaver` checkpointer (`checkpointer.list(None)`), dedupes `thread_id`s, and returns for each `{thread_id, autonomy, last_message}`.
Read-only.
Reuses the snapshot's shape logic.

### 2. Extend `GET /sessions/{id}` - full transcript

The snapshot endpoint currently returns only `last_message`.
Extend it to also return `messages`: an ordered array of `{role, content}` derived from `state.values["messages"]`, mapping message types:

- `HumanMessage` -> `role: "user"`
- `AIMessage` -> `role: "agent"` (include tool calls if present, same shape as the SSE `tool_call` event)
- `ToolMessage` -> `role: "tool_result"` (name + content)

This lets the UI hydrate full history when switching to an existing session.
The existing `autonomy`, `pending_confirmations`, and `last_message` fields stay.

### 3. `POST /sessions/{id}/autonomy` - mid-session toggle

Body: `{"autonomy": "guarded" | "autonomous"}`.
Validates the value is one of the two allowed strings (trust-boundary input check); rejects anything else with 400.
Updates checkpointed state via `agent.update_state(thread, {"autonomy": <value>})`.
Returns `{"autonomy": <new value>}`.
This is the "UI-button hook" the V2 Spec 2 note called for.

### 4. `GET /sessions/{id}/leaderboard` - leaderboard read

Reads the active model's leaderboard from the registry directly: `registry.get(active)["metrics"]["leaderboard"]`.
Returns `{"active": <model_id>, "leaderboard": [...rows]}`.
When no model is trained yet, returns `{"active": null, "leaderboard": []}` - the empty state, not an error.

The registry is not currently constructed in `app.py`; the endpoint builds/accesses it the same way the tools layer does (`models_dir="artifacts/models"`), or the app captures a registry handle at factory time.
Implementation decides the cleanest wiring; the contract above is fixed.

### 5. CORS

Add `fastapi.middleware.cors.CORSMiddleware`, allowing the Vite dev origin (`http://localhost:5173`), all methods, and the `x-thread-id` response header exposed.

## Frontend structure

New top-level `web/` directory with its own `package.json` (kept out of the Python package).

```
web/
  package.json, vite.config.ts, tsconfig.json, index.html
  .env.example                 # VITE_API_BASE=http://localhost:8000
  src/
    main.tsx, App.tsx
    api/
      client.ts                # fetch wrappers: listSessions, startSession,
                               #   sendMessage, resume, setAutonomy,
                               #   getLeaderboard, getSnapshot
      sse.ts                   # parseSSEStream(response): AsyncGenerator<SentinelEvent>
      types.ts                 # SentinelEvent union + DTOs
    state/
      useSession.ts            # useReducer: transcript + pending confirmations + streaming flag
    components/
      SessionSidebar.tsx       # session list + "New session", switch active thread
      ChatView.tsx             # transcript + composer, owns the SSE loop
      Message.tsx              # one bubble; variants: user / agent / tool_call / tool_result / error
      ConfirmCard.tsx          # inline Yes/No card rendered from a `confirm` event
      AutonomyToggle.tsx       # reads snapshot, calls setAutonomy
      LeaderboardPanel.tsx     # fetch + table, manual refresh button
    styles.css                 # single stylesheet, no UI kit
```

### Boundaries

- `sse.ts` is the only place that knows the wire format - a generator yielding typed events.
  Everything downstream is typed.
- `client.ts` is the only place that knows endpoint URLs.
- `useSession.ts` is a pure reducer - events in, transcript state out.
  Testable without a browser.
- Components are dumb renderers driven by that state.

No router library - the app is a two-pane layout (sidebar + main).
The active thread id is React state, persisted to `localStorage` so a reload restores it.

### Event types (from the backend SSE stream)

`message`, `tool_call`, `tool_result`, `confirm`, `error`, `done`.
The `confirm` event carries the interrupt value plus its `interrupt` id.

## Data flow

### A chat turn (the core loop, in `ChatView`)

1. User submits text -> reducer appends a `user` message, sets `streaming: true`.
2. Client POSTs to `/sessions/{id}/message` (or `/sessions` for the first turn, capturing `x-thread-id` from the response header).
3. `parseSSEStream(response)` yields events; the loop dispatches each into the reducer:
   - `message` -> append/extend an `agent` bubble
   - `tool_call` -> append a `tool_call` bubble (name + args, collapsed)
   - `tool_result` -> append a `tool_result` bubble
   - `confirm` -> push into `pendingConfirmations` (id + action + detail)
   - `error` -> append an `error` bubble, stop streaming
   - `done` -> clear `streaming`
4. Stream ends -> if `pendingConfirmations` is non-empty, `ChatView` renders `ConfirmCard`s inline at the bottom of the transcript.

### Resolving a confirmation

- User clicks Yes/No on a `ConfirmCard` -> POST `/sessions/{id}/resume` with `{answers: {[interruptId]: "yes" | "no"}}`.
  Always the map form (never the single-pending convenience path) so multiple pending confirmations just work.
- That response is itself an SSE stream -> the same loop consumes it, continuing the transcript.
- The resolved confirmation is removed from `pendingConfirmations` on submit.
- Confirmation buttons are disabled while a resume request is in flight (no double-submit).

### Leaderboard and autonomy (out of band)

These are not part of the SSE turn.

- `AutonomyToggle` reads the initial value from `GET /sessions/{id}`, writes via `POST .../autonomy`, updates local state on success.
- `LeaderboardPanel` fetches `GET .../leaderboard` on mount and on a manual refresh button.
  It does not auto-refresh on every `tool_result`.
  `ponytail:` refresh button is the minimum that works; upgrade to auto-refresh-on-train-`tool_result` if it feels stale in practice.

### Session switching

Selecting a thread in the sidebar loads its transcript via the extended `GET /sessions/{id}` (which now returns the full `messages` array).
The UI hydrates the full transcript; new turns append live.
The composer is disabled while `streaming: true`.

## Error handling

- SSE `error` event -> red error bubble in the transcript, streaming stops, composer re-enabled.
  (The backend already emits this on any agent exception.)
- Network failure on the fetch itself (backend down) -> caught in the loop, surfaced via the same error-bubble path.
- Non-2xx from a plain fetch (404 unknown session, 400 bad autonomy value) -> thrown, shown as an inline error near the relevant control, not a transcript bubble.
- Confirmation buttons disabled during an in-flight resume.
- Composer disabled during streaming.

## Testing

Kept proportional - this is a scaffold-and-handoff, not a test-heavy deliverable.

- `sse.ts` - unit test the parser: feed a chunked byte stream (including a `data:` line split across chunk boundaries) and assert the yielded typed events.
  This is the one piece with real parsing logic, so it gets the one real frontend test.
- `useSession.ts` reducer - a couple of assertions: a `message` then `done` sequence produces the right transcript; a `confirm` event populates pending.
  Pure function, no DOM.
- Backend - extend the test suite (`tests/test_agents.py` or a small `tests/test_api.py`) with FastAPI `TestClient` checks for the new/changed endpoints: list sessions, autonomy toggle validation (reject a bad value), leaderboard empty-state, snapshot returns messages.
  Offline, using the existing fake-LLM fixtures.
- No component/E2E framework in scope for the handoff.
  Codex can add Playwright later if wanted.
  `ponytail:` deferred.

## Out of scope

- Auth / multi-user.
- Production build/deploy of the SPA (dev-server workflow only for now).
- Auto-refreshing the leaderboard on training completion.
- Reconstructing tool-call argument diffs or run-progress visualizations beyond the raw `tool_call` / `tool_result` bubbles.

## Handoff split

- Claude Code: scaffolds `web/` (Vite project, `sse.ts`, `client.ts`, `types.ts`, `useSession.ts`, app shell) and implements the five backend additions with their tests.
- Codex: builds out the remaining components (`SessionSidebar`, `ChatView` polish, `ConfirmCard`, `AutonomyToggle`, `LeaderboardPanel`, styling) against the fixed contracts above.
