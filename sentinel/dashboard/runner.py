"""GraphRunner: run the M2 graph in a background thread, bridged to a UI by an
append-only event transcript.

The M2 interviewer talks to a human through a *blocking* ``ask(question) -> str``
callable, and the graph runs start-to-finish in one ``graph.invoke`` call. A
rerun-on-interaction UI (Streamlit) cannot call a blocking function on its own
thread. So we run ``graph.invoke`` on a daemon thread and bridge the graph's
injected ``ask``/``notify`` seam to the UI:

- ``ask(q)``  -> append an Event(kind="prompt", payload=q) to the transcript and
                 record it as the pending prompt, then block until ``answer(text)``.
- ``notify(m)`` -> append an Event(kind="notify", payload=m).
- the training function is wrapped to bracket the (long, live) PyCaret run with
  ``training_started`` / ``training_finished`` events and a start timestamp.

Everything the run produces goes onto a single append-only ``history`` list, and
the user's own answers are recorded there too. The UI renders as a pure function
of that history, so it can rebuild the whole conversation on a fresh page load
(a browser refresh) - nothing lives only in the browser session.

Two rules keep the concurrency honest:
- ``answer()`` clears the pending prompt *synchronously* (not the worker thread),
  so the UI never re-renders the just-answered question and never lands the next
  reply on the wrong field.
- the worker thread never touches ``st.*`` (background threads have no Streamlit
  context); it only appends to ``history`` and reads the answer queue.

This module imports no ``streamlit`` and is the seam a future real web app reuses;
only ``app.py`` is throwaway.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

from ..agents.state import AgentState


@dataclass
class Event:
    """One thing that happened during the run, in transcript order."""

    kind: str  # "prompt" | "answer" | "notify" | "training_started" | "training_finished" | "error"
    payload: object = None


class GraphRunner:
    """Drive a compiled graph on a background thread, bridged to a UI by a transcript."""

    def __init__(self, *, graph, provider_smart, provider_cheap, train_fn, ticket_dir: str) -> None:
        self._graph = graph
        self._provider_smart = provider_smart
        self._provider_cheap = provider_cheap
        self._train_fn = train_fn
        self._ticket_dir = ticket_dir

        self._history: list[Event] = []  # append-only transcript (worker + answers)
        self._polled = 0  # index consumed by poll()
        self._in: "queue.Queue[str]" = queue.Queue()  # UI -> graph (interview answers)
        self._seen_kinds: set[str] = set()
        self._pending_prompt: str | None = None
        self._training_started_at: float | None = None
        self._final: AgentState | None = None
        self._error: Exception | None = None
        self._thread: threading.Thread | None = None

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker thread once (a second call is a no-op)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        configurable = {
            "ask": self._ask,
            "notify": self._notify,
            "provider_smart": self._provider_smart,
            "provider_cheap": self._provider_cheap,
            "train_fn": self._wrapped_train_fn,
            "ticket_dir": self._ticket_dir,
        }
        try:
            self._final = self._graph.invoke({"event": "start"}, config={"configurable": configurable})
        except Exception as exc:  # noqa: BLE001 - surface any unexpected failure to the UI
            self._error = exc
            self._emit("error", f"{type(exc).__name__}: {exc}")

    # --- seam callables injected into the graph (run on the worker thread) ---

    def _ask(self, question: str) -> str:
        """Blocking ask: record the prompt, then wait for the UI's answer."""
        self._pending_prompt = question
        self._emit("prompt", question)
        answer = self._in.get()  # blocks the worker until answer() is called
        self._pending_prompt = None
        return answer

    def _notify(self, message: str) -> None:
        self._emit("notify", message)

    def _wrapped_train_fn(self, config):
        """Bracket the real training run with lifecycle events + a start time."""
        self._training_started_at = time.time()
        self._emit("training_started")
        run = self._train_fn(config)  # may raise; the trainer node catches it (run_failed)
        self._emit("training_finished")
        return run

    def _emit(self, kind: str, payload: object = None) -> None:
        self._seen_kinds.add(kind)
        self._history.append(Event(kind, payload))

    # --- UI-facing API (run on the main / Streamlit thread) ---------------

    def history(self) -> list[Event]:
        """The full append-only transcript (prompts, answers, notifications, markers)."""
        return list(self._history)

    def poll(self) -> list[Event]:
        """Return transcript events emitted since the last poll (single consumer)."""
        new = self._history[self._polled :]
        self._polled = len(self._history)
        return new

    def pending_prompt(self) -> str | None:
        return self._pending_prompt

    def answer(self, text: str) -> None:
        """Deliver the user's reply. Clears the pending prompt synchronously and
        records the answer in the transcript, then unblocks the worker."""
        if self._pending_prompt is None:
            return
        self._pending_prompt = None
        self._history.append(Event("answer", text))
        self._in.put(text)

    def saw(self, kind: str) -> bool:
        return kind in self._seen_kinds

    def training_elapsed(self) -> float | None:
        """Seconds since training started, or None if it has not started."""
        if self._training_started_at is None:
            return None
        return time.time() - self._training_started_at

    @property
    def done(self) -> bool:
        return self._final is not None or self._error is not None

    @property
    def error(self) -> Exception | None:
        return self._error

    def final_state(self) -> AgentState | None:
        return self._final
