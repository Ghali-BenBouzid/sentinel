import { describe, expect, it } from "vitest";
import { initialSession, sessionReducer } from "./useSession";

describe("sessionReducer", () => {
  it("streams an agent message then finishes on done", () => {
    let s = sessionReducer(initialSession, { type: "start" });
    expect(s.streaming).toBe(true);
    s = sessionReducer(s, {
      type: "event",
      event: { event: "message", data: { text: "hello" } },
    });
    s = sessionReducer(s, {
      type: "event",
      event: { event: "done", data: {} },
    });
    expect(s.streaming).toBe(false);
    expect(s.transcript.at(-1)).toMatchObject({ kind: "agent", text: "hello" });
  });

  it("collects a confirm event into pending and clears streaming", () => {
    let s = sessionReducer(initialSession, { type: "start" });
    s = sessionReducer(s, {
      type: "event",
      event: {
        event: "confirm",
        data: { type: "confirm", tool: "promote", detail: "et-v2", interrupt: "i1" },
      },
    });
    expect(s.pending).toEqual([{ interrupt: "i1", tool: "promote", detail: "et-v2" }]);
    expect(s.streaming).toBe(false);
  });

  it("ignores unknown custom events without crashing", () => {
    const s = sessionReducer(initialSession, {
      type: "event",
      event: { event: "model_training", data: { index: 1, total: 11 } },
    });
    expect(s.transcript).toEqual([]);
  });

  it("removes a pending confirmation on resolve", () => {
    let s = sessionReducer(initialSession, {
      type: "event",
      event: {
        event: "confirm",
        data: { type: "confirm", tool: "delete", detail: "et-v1", interrupt: "i1" },
      },
    });
    s = sessionReducer(s, { type: "resolve", interrupt: "i1" });
    expect(s.pending).toEqual([]);
  });
});
