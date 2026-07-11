import { useState } from "react";
import * as client from "../api/client";
import type { SessionAction, SessionState } from "../state/useSession";
import { consumeStream } from "../state/useStream";
import { ConfirmCard } from "./ConfirmCard";
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
  const [resumeBusy, setResumeBusy] = useState(false);

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

  function choose(interrupt: string, value: "yes" | "no") {
    if (resumeBusy) return;
    dispatch({ type: "choose", interrupt, decision: value });
  }

  async function submitDecisions() {
    if (!threadId || resumeBusy) return;
    if (state.pending.length === 0 || !state.pending.every((p) => p.decision !== null)) return;
    setResumeBusy(true);
    const answers: Record<string, string> = {};
    for (const p of state.pending) answers[p.interrupt] = p.decision as string;
    const controller = new AbortController();
    streamAbort.current = controller;
    dispatch({ type: "resolveAll" });
    try {
      const response = await client.resume(threadId, answers, controller.signal);
      await consumeStream(response, dispatch);
      streamAbort.current = null;
      onTurnFinished();
    } finally {
      setResumeBusy(false);
    }
  }

  return (
    <section className="app">
      <section className="transcript">
        {state.transcript.map((bubble) => (
          <Message key={bubble.id} bubble={bubble} />
        ))}
        {state.pending.map((p) => (
          <ConfirmCard
            key={p.interrupt}
            pending={p}
            disabled={resumeBusy}
            onChoose={(value) => choose(p.interrupt, value)}
          />
        ))}
        {state.pending.length > 0 && (
          <button
            type="button"
            onClick={submitDecisions}
            disabled={resumeBusy || !state.pending.every((p) => p.decision !== null)}
          >
            Submit responses
          </button>
        )}
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
