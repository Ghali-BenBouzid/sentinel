"""GraphRunner: run the M2 graph in a background thread, bridged to a UI by queues.

The M2 interviewer talks to a human through a *blocking* ``ask(question) -> str``
callable, and the graph runs start-to-finish in one ``graph.invoke`` call. A
rerun-on-interaction UI (Streamlit) cannot call a blocking function on its own
thread. So we run ``graph.invoke`` on a daemon thread and bridge the graph's
injected ``ask``/``notify`` seam to the UI through two thread-safe queues:

- ``ask(q)``  -> push an Event(kind="prompt", payload=q) onto the OUT queue,
                 then block on the IN queue until the UI calls ``answer(text)``.
- ``notify(m)`` -> push an Event(kind="notify", payload=m) onto the OUT queue.

The training function is wrapped to bracket the (long, live) PyCaret run with
``training_started`` / ``training_finished`` events so the UI can show progress.

This module is deliberately framework-agnostic: it imports no ``streamlit`` and
the worker thread never touches ``st.*``. It is the seam a future real web app
reuses; only ``app.py`` is throwaway.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from ..agents.state import AgentState


@dataclass
class Event:
    """One thing the running graph emitted for the UI to react to."""

    kind: str  # "prompt" | "notify" | "training_started" | "training_finished" | "error"
    payload: object = None


class GraphRunner:
    """Drive a compiled graph on a background thread, bridged to a UI by queues."""

    def __init__(self, *, graph, provider_smart, provider_cheap, train_fn, ticket_dir: str) -> None:
        self._graph = graph
        self._provider_smart = provider_smart
        self._provider_cheap = provider_cheap
        self._train_fn = train_fn
        self._ticket_dir = ticket_dir

        self._out: "queue.Queue[Event]" = queue.Queue()  # graph -> UI
        self._in: "queue.Queue[str]" = queue.Queue()  # UI -> graph (interview answers)
        self._seen_kinds: set[str] = set()
        self._pending_prompt: str | None = None
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
        """Blocking ask: publish the prompt, then wait for the UI's answer."""
        self._pending_prompt = question
        self._emit("prompt", question)
        answer = self._in.get()  # blocks the worker until answer() is called
        self._pending_prompt = None
        return answer

    def _notify(self, message: str) -> None:
        self._emit("notify", message)

    def _wrapped_train_fn(self, config):
        """Bracket the real training run with lifecycle events for the UI."""
        self._emit("training_started")
        run = self._train_fn(config)  # may raise; the trainer node catches it (run_failed)
        self._emit("training_finished")
        return run

    def _emit(self, kind: str, payload: object = None) -> None:
        self._seen_kinds.add(kind)
        self._out.put(Event(kind, payload))

    # --- UI-facing API (run on the main / Streamlit thread) ---------------

    def poll(self) -> list[Event]:
        """Drain and return every event emitted since the last poll."""
        events: list[Event] = []
        while True:
            try:
                events.append(self._out.get_nowait())
            except queue.Empty:
                break
        return events

    def pending_prompt(self) -> str | None:
        return self._pending_prompt

    def answer(self, text: str) -> None:
        """Hand the user's reply to the blocked ask (no-op if nothing pending)."""
        if self._pending_prompt is not None:
            self._in.put(text)

    def saw(self, kind: str) -> bool:
        return kind in self._seen_kinds

    @property
    def done(self) -> bool:
        return self._final is not None or self._error is not None

    @property
    def error(self) -> Exception | None:
        return self._error

    def final_state(self) -> AgentState | None:
        return self._final
