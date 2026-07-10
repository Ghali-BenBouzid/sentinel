import { useState } from "react";
import * as client from "../api/client";
import type { SessionAction, SessionState } from "../state/useSession";
import { consumeStream } from "../state/useStream";
import { Message } from "./Message";

interface ChatViewProps {
  state: SessionState;
  dispatch: React.Dispatch<SessionAction>;
  threadId: string | null;
  setThreadId: (threadId: string | null) => void;
  streamAbort: React.MutableRefObject<AbortController | null>;
  onSessionStarted: () => void;
  onTurnFinished: () => void;
}

export function ChatView({
  state,
  dispatch,
  threadId,
  setThreadId,
  streamAbort,
  onSessionStarted,
  onTurnFinished,
}: ChatViewProps) {
  const [input, setInput] = useState("");

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
      onSessionStarted();
    } else {
      response = await client.sendMessage(tid, text, controller.signal);
    }
    await consumeStream(response, dispatch);
    streamAbort.current = null;
    onTurnFinished();
  }

  async function answer(interrupt: string, value: "yes" | "no") {
    if (!threadId) return;
    dispatch({ type: "resolve", interrupt });
    const controller = new AbortController();
    streamAbort.current = controller;
    const response = await client.resume(threadId, { [interrupt]: value }, controller.signal);
    await consumeStream(response, dispatch);
    streamAbort.current = null;
    onTurnFinished();
  }

  return (
    <section className="app">
      <section className="transcript">
        {state.transcript.map((bubble) => (
          <Message key={bubble.id} bubble={bubble} />
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
  );
}
