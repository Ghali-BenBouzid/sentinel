import type { Bubble } from "../state/useSession";

interface MessageProps {
  bubble: Bubble;
}

export function Message({ bubble }: MessageProps) {
  if (bubble.kind === "tool_call") {
    return (
      <details className={`bubble bubble-${bubble.kind}`}>
        <summary>{bubble.name || "tool_call"}</summary>
        <pre>{bubble.text}</pre>
      </details>
    );
  }

  return (
    <div className={`bubble bubble-${bubble.kind}`}>
      {bubble.name ? <strong>{bubble.name}: </strong> : null}
      {bubble.text}
    </div>
  );
}
