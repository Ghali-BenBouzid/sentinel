import type { SentinelEvent } from "./types";

export async function* parseSSEStream(
  response: Response,
): AsyncGenerator<SentinelEvent> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const ev = frameToEvent(frame);
        if (ev) yield ev;
      }
    }
    // Flush a trailing frame that arrived without its final blank line.
    const ev = frameToEvent(buffer);
    if (ev) yield ev;
  } finally {
    reader.releaseLock();
  }
}

function frameToEvent(frame: string): SentinelEvent | null {
  const line = frame.split("\n").find((l) => l.startsWith("data:"));
  if (!line) return null;
  return JSON.parse(line.slice(5).trim()) as SentinelEvent;
}
