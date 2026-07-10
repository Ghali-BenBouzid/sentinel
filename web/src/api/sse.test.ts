import { describe, expect, it } from "vitest";
import { parseSSEStream } from "./sse";
import type { SentinelEvent } from "./types";

function responseFromChunks(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c));
      controller.close();
    },
  });
  return new Response(stream);
}

async function collect(resp: Response): Promise<SentinelEvent[]> {
  const out: SentinelEvent[] = [];
  for await (const ev of parseSSEStream(resp)) out.push(ev);
  return out;
}

describe("parseSSEStream", () => {
  it("parses whole events", async () => {
    const resp = responseFromChunks([
      'data: {"event":"message","data":{"text":"hi"}}\n\n',
      'data: {"event":"done","data":{}}\n\n',
    ]);
    const events = await collect(resp);
    expect(events).toEqual([
      { event: "message", data: { text: "hi" } },
      { event: "done", data: {} },
    ]);
  });

  it("reassembles an event split across chunk boundaries", async () => {
    const resp = responseFromChunks([
      'data: {"event":"mess',
      'age","data":{"text":"hi"}}\n\n',
    ]);
    const events = await collect(resp);
    expect(events).toEqual([{ event: "message", data: { text: "hi" } }]);
  });
});
