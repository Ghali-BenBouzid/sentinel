import { useReducer, useState } from "react";
import * as client from "./api/client";
import { initialSession, sessionReducer } from "./state/useSession";
import { consumeStream } from "./state/useStream";
import "./styles.css";

export default function App() {
  const [state, dispatch] = useReducer(sessionReducer, initialSession);
  const [threadId, setThreadId] = useState<string | null>(
    () => localStorage.getItem("sentinel.thread"),
  );
  const [input, setInput] = useState("");

  async function send() {
    const text = input.trim();
    if (!text || state.streaming) return;
    setInput("");
    dispatch({ type: "user", text });
    let tid = threadId;
    let response: Response;
    if (!tid) {
      const started = await client.startSession(text);
      tid = started.threadId;
      response = started.response;
      setThreadId(tid);
      localStorage.setItem("sentinel.thread", tid);
    } else {
      response = await client.sendMessage(tid, text);
    }
    await consumeStream(response, dispatch);
  }

  async function answer(interrupt: string, value: "yes" | "no") {
    if (!threadId) return;
    dispatch({ type: "resolve", interrupt });
    const response = await client.resume(threadId, { [interrupt]: value });
    await consumeStream(response, dispatch);
  }

  return (
    <main className="app">
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
    </main>
  );
}
