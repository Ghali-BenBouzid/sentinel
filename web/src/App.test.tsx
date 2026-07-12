// @vitest-environment jsdom
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as client from "./api/client";
import App from "./App";

vi.mock("./api/client", () => ({
  getLeaderboard: vi.fn(),
  getSnapshot: vi.fn(),
  listSessions: vi.fn(),
  resume: vi.fn(),
  sendMessage: vi.fn(),
  setAutonomy: vi.fn(),
  startSession: vi.fn(),
}));

function delayedSse(signal?: AbortSignal): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const timer = window.setTimeout(() => {
        controller.enqueue(
          encoder.encode(
            'data: {"event":"message","data":{"text":"Server reply"}}\n\n' +
              'data: {"event":"done","data":{}}\n\n',
          ),
        );
        controller.close();
      }, 20);
      signal?.addEventListener("abort", () => {
        window.clearTimeout(timer);
        controller.error(new DOMException("Aborted", "AbortError"));
      });
    },
  });
  return new Response(body, {
    headers: { "content-type": "text/event-stream", "x-thread-id": "new-thread" },
  });
}

describe("App new-session streaming", () => {
  beforeEach(() => {
    HTMLElement.prototype.scrollTo = vi.fn();
    localStorage.clear();
    vi.clearAllMocks();
    vi.mocked(client.listSessions).mockResolvedValue([]);
    vi.mocked(client.getLeaderboard).mockResolvedValue({ active: null, leaderboard: [] });
    vi.mocked(client.getSnapshot).mockResolvedValue({
      title: null,
      autonomy: "guarded",
      pending_confirmations: [],
      last_message: null,
      messages: [],
    });
    vi.mocked(client.startSession).mockImplementation(async (_message, _autonomy, signal) => ({
      threadId: "new-thread",
      response: delayedSse(signal),
    }));
  });

  it("does not abort the first SSE stream when the new thread id arrives", async () => {
    render(<App />);

    fireEvent.change(screen.getByPlaceholderText(/ask the agent/i), {
      target: { value: "Can you see the leaderboard?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    expect(await screen.findByText("Server reply")).toBeTruthy();
    expect(screen.queryByText("working")).toBeNull();
  });
});
