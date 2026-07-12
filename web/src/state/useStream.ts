import type { Dispatch } from "react";
import { parseSSEStream } from "../api/sse";
import type { SessionAction } from "./useSession";

import { TERMINAL_EVENTS } from "../api/types";

export async function consumeStream(
  response: Response,
  dispatch: Dispatch<SessionAction>,
): Promise<void> {
  dispatch({ type: "start" });
  let sawTerminal = false;
  try {
    for await (const event of parseSSEStream(response)) {
      if ((TERMINAL_EVENTS as readonly string[]).includes(event.event)) {
        sawTerminal = true;
      }
      dispatch({ type: "event", event });
    }
    if (!sawTerminal) {
      dispatch({
        type: "event",
        event: { event: "error", data: { message: "stream ended unexpectedly" } },
      });
    }
  } catch (err) {
    // AbortError from a deliberate session switch is not a real error.
    if ((err as Error)?.name === "AbortError") return;
    dispatch({
      type: "event",
      event: { event: "error", data: { message: String(err) } },
    });
  }
}
