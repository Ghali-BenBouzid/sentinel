import { useEffect, useRef, useState } from "react";
import * as client from "../api/client";
import type { SessionAction, SessionState } from "../state/useSession";
import { consumeStream } from "../state/useStream";
import { AutonomyToggle } from "./AutonomyToggle";
import { ConfirmCard } from "./ConfirmCard";
import { Icon } from "./Icon";
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
  const transcriptRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight });
  }, [state.transcript.length, state.pending.length, state.streaming]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [input]);

  async function send() {
    const text = input.trim();
    if (!text || state.streaming) return;
    setInput("");
    dispatch({ type: "user", text });
    dispatch({ type: "start" });
    const controller = new AbortController();
    streamAbort.current = controller;
    try {
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
    } catch (err) {
      if ((err as Error)?.name !== "AbortError") {
        dispatch({
          type: "event",
          event: { event: "error", data: { message: String(err) } },
        });
      }
    } finally {
      streamAbort.current = null;
      onTurnFinished();
    }
  }

  function choose(interrupt: string, value: "yes" | "no") {
    if (!resumeBusy) dispatch({ type: "choose", interrupt, decision: value });
  }

  async function submitDecisions() {
    if (!threadId || resumeBusy) return;
    if (state.pending.length === 0 || !state.pending.every((p) => p.decision)) return;
    setResumeBusy(true);
    const answers = Object.fromEntries(
      state.pending.map((p) => [p.interrupt, p.decision as string]),
    );
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

  const canSend = input.trim().length > 0 && !state.streaming;

  return (
    <section className="chat-workspace">
      <AutonomyToggle threadId={threadId} />
      <div className="transcript" ref={transcriptRef}>
        <div className="transcript-inner">
          {state.transcript.length === 0 && state.pending.length === 0 ? (
            <div className="empty-chat">
              <span><Icon name="agent" size={22} /></span>
              <h2>New session started</h2>
              <p>Ask the agent to inspect, train, compare, report, or monitor. Nothing has run yet.</p>
            </div>
          ) : null}
          {state.transcript.map((bubble) => (
            <Message key={bubble.id} bubble={bubble} />
          ))}
          {state.pending.length > 0 && (
            <section className="confirmation-card">
              <div className="message-header"><span className="agent-mark"><Icon name="agent" size={12} /></span><strong>Agent</strong></div>
              <div className="confirmation-body">
                <p>I need your approval before I continue.</p>
                {state.pending.map((pending) => (
                  <ConfirmCard
                    key={pending.interrupt}
                    pending={pending}
                    disabled={resumeBusy}
                    onChoose={(value) => choose(pending.interrupt, value)}
                  />
                ))}
                <button
                  className="primary-button submit-decisions"
                  type="button"
                  onClick={submitDecisions}
                  disabled={resumeBusy || !state.pending.every((p) => p.decision)}
                >
                  Submit responses
                </button>
              </div>
            </section>
          )}
          {state.streaming && (
            <div className="working-indicator" role="status">
              <span /><span /><span /> working
            </div>
          )}
        </div>
      </div>

      <div className="composer-wrap">
        <div className="composer">
          <textarea
            ref={textareaRef}
            value={input}
            disabled={state.streaming}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send();
              }
            }}
            rows={1}
            placeholder="Ask the agent to analyze, train, compare…"
          />
          <div className="composer-actions">
            <div>
              <button type="button" disabled title="Dataset attachment API is not available yet"><Icon name="database" size={13} /> Attach dataset</button>
              <button type="button" disabled title="Source integration is planned"><Icon name="integrations" size={13} /> Connect source</button>
            </div>
            <button className="send-button" aria-label="Send message" type="button" onClick={send} disabled={!canSend}>
              <Icon name="send" size={15} />
            </button>
          </div>
        </div>
        <p className="disclaimer">The agent can make mistakes. Always validate important results before acting on them.</p>
      </div>
    </section>
  );
}
