import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Bubble } from "../state/useSession";
import { Icon } from "./Icon";

interface MessageProps {
  bubble: Bubble;
}

export function Message({ bubble }: MessageProps) {
  if (bubble.kind === "user") {
    return (
      <article className="message message-user">
        <div className="user-message">
          <div>{bubble.text}</div>
          {bubble.time && <time>{bubble.time}</time>}
        </div>
      </article>
    );
  }

  if (bubble.kind === "tool_call" || bubble.kind === "tool_result") {
    return (
      <article className="message message-agent">
        <MessageHeader time={bubble.time} />
        <details className={`tool-event ${bubble.kind === "tool_result" ? "is-result" : ""}`} open={bubble.kind === "tool_result"}>
          <summary>
            <span className="tool-status"><Icon name={bubble.kind === "tool_result" ? "status" : "spark"} size={13} /></span>
            <span>{bubble.name || "Tool"}</span>
            <small>{bubble.kind === "tool_result" ? "completed" : "called"}</small>
          </summary>
          <pre>{bubble.text}</pre>
        </details>
      </article>
    );
  }

  if (bubble.kind === "error") {
    return (
      <article className="message message-agent">
        <MessageHeader time={bubble.time} />
        <div className="error-card"><Icon name="alert" />{bubble.text}</div>
      </article>
    );
  }

  return (
    <article className="message message-agent">
      <MessageHeader time={bubble.time} />
      <div className="agent-content markdown-body">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ children, ...props }) => (
              <a {...props} target="_blank" rel="noreferrer">{children}</a>
            ),
          }}
        >
          {bubble.text}
        </ReactMarkdown>
      </div>
    </article>
  );
}

function MessageHeader({ time }: { time?: string }) {
  return (
    <header className="message-header">
      <span className="agent-mark"><Icon name="agent" size={12} /></span>
      <strong>Agent</strong>
      {time && <time>{time}</time>}
    </header>
  );
}
