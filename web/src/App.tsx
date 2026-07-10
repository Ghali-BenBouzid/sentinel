import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import * as client from "./api/client";
import { SessionSidebar } from "./components/SessionSidebar";
import { initialSession, Pending, sessionReducer } from "./state/useSession";
import { consumeStream } from "./state/useStream";
import "./styles.css";

function snapshotPending(items: { interrupt: string; [k: string]: unknown }[]): Pending[] {
  return items.map((item) => ({
    interrupt: item.interrupt,
    tool: String(item.tool ?? ""),
    detail: String(item.detail ?? ""),
  }));
}

export default function App() {
  const [state, dispatch] = useReducer(sessionReducer, initialSession);
  const [threadId, setThreadId] = useState<string | null>(
    () => localStorage.getItem("sentinel.thread"),
  );
  const [input, setInput] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const streamAbort = useRef<AbortController | null>(null);

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
    if (threadId) void selectThread(threadId);
  }, []);

  useEffect(() => abortStream, [abortStream]);

  async function send() {
    const text = input.trim();
    if (!text || state.streaming) return;
    setInput("");
    dispatch({ type: "user", text });
    const controller = new AbortController();
    streamAbort.current = controller;
    let tid = threadId;
    let response: Response;
    if (!tid) {
      const started = await client.startSession(text, undefined, controller.signal);
      tid = started.threadId;
      response = started.response;
      setThreadId(tid);
      localStorage.setItem("sentinel.thread", tid);
      setRefreshKey((key) => key + 1);
    } else {
      response = await client.sendMessage(tid, text, controller.signal);
    }
    await consumeStream(response, dispatch);
    streamAbort.current = null;
    setRefreshKey((key) => key + 1);
  }

  async function answer(interrupt: string, value: "yes" | "no") {
    if (!threadId) return;
    dispatch({ type: "resolve", interrupt });
    const controller = new AbortController();
    streamAbort.current = controller;
    const response = await client.resume(threadId, { [interrupt]: value }, controller.signal);
    await consumeStream(response, dispatch);
    streamAbort.current = null;
    setRefreshKey((key) => key + 1);
  }

  function newSession() {
    abortStream();
    dispatch({ type: "reset" });
    setInput("");
    setThreadId(null);
    localStorage.removeItem("sentinel.thread");
    setLoadError(null);
  }

  return (
    <main className="app-shell">
      <SessionSidebar
        activeThreadId={threadId}
        refreshKey={refreshKey}
        onSelect={selectThread}
        onNew={newSession}
      />
      <section className="app">
        {loadError ? <p className="load-error">{loadError}</p> : null}
        <section className="transcript">
          {state.transcript.map((b) => (
            <div key={b.id} className={`bubble bubble-${b.kind}`}>
              {b.name ? <strong>{b.name}: </strong> : null}
              {b.text}
            </div>
          ))}
          {state.pending.map((p) => (
            <div key={p.interrupt} className="confirm">
              <span>
                {p.tool} {p.detail}?
              </span>
              <button onClick={() => answer(p.interrupt, "yes")}>Yes</button>
              <button onClick={() => answer(p.interrupt, "no")}>No</button>
            </div>
          ))}
        </section>
        <div className="composer">
          <input
            value={input}
            disabled={state.streaming}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="Message the agent..."
          />
          <button onClick={send} disabled={state.streaming}>
            Send
          </button>
        </div>
      </section>
    </main>
  );
}
