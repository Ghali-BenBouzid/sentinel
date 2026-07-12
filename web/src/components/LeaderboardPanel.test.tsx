// @vitest-environment jsdom
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as client from "../api/client";
import { initialSession } from "../state/useSession";
import { LeaderboardPanel } from "./LeaderboardPanel";

vi.mock("../api/client", () => ({ getLeaderboard: vi.fn() }));

const rows = Array.from({ length: 7 }, (_, index) => ({
  Model: `model-${index + 1}`,
  MAE: 10 + index,
  RMSE: 20 + index,
  R2: 0.9 - index / 100,
}));

describe("LeaderboardPanel", () => {
  afterEach(() => vi.clearAllMocks());

  it("shows a compact leaderboard and expands to every row and metric", async () => {
    vi.mocked(client.getLeaderboard).mockResolvedValue({
      active: "model-1",
      leaderboard: rows,
    });
    render(<LeaderboardPanel threadId="thread-1" state={initialSession} />);

    await screen.findAllByText("model-1");
    expect(screen.queryByText("model-7")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /view full leaderboard/i }));

    expect(screen.getByText("model-7")).toBeTruthy();
    expect(screen.getByRole("columnheader", { name: "MAE" })).toBeTruthy();
    expect(screen.getByRole("columnheader", { name: "RMSE" })).toBeTruthy();
    expect(screen.getByRole("columnheader", { name: "R2" })).toBeTruthy();
  });

  it("polls automatically while a session is active", async () => {
    vi.useFakeTimers();
    vi.mocked(client.getLeaderboard).mockResolvedValue({
      active: "model-1",
      leaderboard: rows,
    });
    render(<LeaderboardPanel threadId="thread-1" state={initialSession} />);

    await act(async () => {});
    expect(client.getLeaderboard).toHaveBeenCalledTimes(1);
    await act(async () => vi.advanceTimersByTimeAsync(15_000));
    expect(client.getLeaderboard).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });
});
