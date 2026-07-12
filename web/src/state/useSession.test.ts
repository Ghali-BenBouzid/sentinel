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
    expect(s.pending).toEqual([
      { interrupt: "i1", tool: "promote", detail: "et-v2", decision: null },
    ]);
    expect(s.streaming).toBe(false);
  });

  it("ignores unknown custom events without crashing", () => {
    const s = sessionReducer(initialSession, {
      type: "event",
      event: { event: "model_training", data: { index: 1, total: 11 } },
    });
    expect(s.transcript).toEqual([]);
  });

  it("clears pending confirmations on resolveAll", () => {
    let s = sessionReducer(initialSession, {
      type: "event",
      event: {
        event: "confirm",
        data: { type: "confirm", tool: "delete", detail: "et-v1", interrupt: "i1" },
      },
    });
    s = sessionReducer(s, { type: "resolveAll" });
    expect(s.pending).toEqual([]);
  });

  it("resets transcript, pending confirmations, and streaming", () => {
    let s = sessionReducer(initialSession, { type: "user", text: "train" });
    s = sessionReducer(s, {
      type: "event",
      event: {
        event: "confirm",
        data: { type: "confirm", tool: "train", detail: "rul", interrupt: "i1" },
      },
    });
    s = sessionReducer(s, { type: "start" });
    expect(sessionReducer(s, { type: "reset" })).toEqual(initialSession);
  });

  it("seeds pending confirmations from a hydrated snapshot", () => {
    const s = sessionReducer(initialSession, {
      type: "setPending",
      pending: [{ interrupt: "i1", tool: "promote", detail: "et-v2", decision: null }],
    });
    expect(s.pending).toEqual([
      { interrupt: "i1", tool: "promote", detail: "et-v2", decision: null },
    ]);
  });
});

describe("choose/resolveAll", () => {
  it("records a local decision without removing the pending card", () => {
    const withPending = sessionReducer(initialSession, {
      type: "event",
      event: { event: "confirm", data: { interrupt: "abc:0", tool: "train", detail: "{}" } },
    });
    const chosen = sessionReducer(withPending, {
      type: "choose",
      interrupt: "abc:0",
      decision: "yes",
    });
    expect(chosen.pending).toHaveLength(1);
    expect(chosen.pending[0].decision).toBe("yes");
  });

  it("resolveAll clears every pending card regardless of decision state", () => {
    const withPending = sessionReducer(initialSession, {
      type: "event",
      event: { event: "confirm", data: { interrupt: "abc:0", tool: "train", detail: "{}" } },
    });
    const cleared = sessionReducer(withPending, { type: "resolveAll" });
    expect(cleared.pending).toHaveLength(0);
  });
});
