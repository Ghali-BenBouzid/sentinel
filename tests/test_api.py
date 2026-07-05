"""End-to-end test for the FastAPI/SSE surface over the resumable agent graph."""

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver


class OneShotProvider:
    """Fake provider returning a scripted JSON string per complete() call.

    Used for the interviewer's gate classification only - the all-defaults fast
    path resolves in exactly one classifier call (see
    tests/test_interviewer_state.py::test_gate_accept_fills_all_defaults_in_one_turn).
    """

    def __init__(self, replies):
        self._it = iter(replies)
        self.calls = 0

    def complete(self, messages, **kw):
        self.calls += 1
        return next(self._it)


class FixedProvider:
    """Fake provider for the report writer - always returns the same text."""

    def complete(self, messages, **kw):
        return "Report: model looks fine."


def _fake_run():
    """A TrainingRun stand-in with a real, serializable to_state() (mirrors the
    fixture in tests/test_train_state.py) so trainer -> report -> monitor run
    offline without PyCaret or a real model file on disk.
    """
    from sentinel.agents.training import TrainingRun
    from sentinel.core.automl import TrainResult

    lb = pd.DataFrame([{"Model": "Extra Trees", "RMSE": 17.1}])
    result = TrainResult(
        leaderboard=lb,
        best_model=object(),
        metrics={"rmse": 17.1, "mae": 12.0, "r2": 0.83},
        model_path=Path("artifacts/model"),
        metrics_path=Path("artifacts/metrics.json"),
    )
    test_eval = pd.DataFrame([{"unit": 1, "cycle": 200, "RUL": 40.0, "s2": 1.5}])
    return TrainingRun(result=result, test_eval=test_eval, predict=lambda f: [1.0])


def _sse_events(resp):
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            yield json.loads(line[6:])


def test_session_start_reaches_first_prompt_and_resume_finishes(tmp_path, monkeypatch):
    from sentinel.agents import monitor
    from sentinel.api.app import create_app

    # The fake train_state's model_path isn't a real model on disk - monkeypatch
    # the loader so the monitor leg runs offline (same trick test_train_state.py uses).
    monkeypatch.setattr(
        monitor, "load_predict", lambda model_path: lambda f: [50.0] * len(f)
    )

    gate_provider = OneShotProvider(['{"all_defaults": true}'])
    report_provider = FixedProvider()

    def factory():
        return {
            "provider_smart": gate_provider,
            "provider_cheap": report_provider,
            "train_fn": lambda c: _fake_run(),
            "ticket_dir": str(tmp_path),
        }

    app = create_app(configurable_factory=factory, checkpointer=MemorySaver())
    client = TestClient(app)

    r = client.post("/sessions")
    assert r.status_code == 200
    tid = r.headers["x-thread-id"]
    events = list(_sse_events(r))
    assert events[-1]["event"] == "prompt"  # gate question awaits an answer

    r2 = client.post(f"/sessions/{tid}/resume", json={"answer": "yes defaults"})
    ev2 = list(_sse_events(r2))
    assert any(e["event"] == "notify" for e in ev2)
    assert ev2[-1]["event"] == "done"

    snap = client.get(f"/sessions/{tid}").json()
    assert snap["phase"] == "done"
    assert isinstance(snap["config"], dict)  # config is native (rehydrated by readers)


def test_unknown_thread_id_returns_404_on_get_and_resume():
    """A thread with no checkpoint must 404, not silently start a fresh interview
    (resume) or return a nulls snapshot (get)."""
    from sentinel.api.app import create_app

    def factory():
        return {
            "provider_smart": OneShotProvider([]),
            "provider_cheap": FixedProvider(),
            "train_fn": lambda c: _fake_run(),
            "ticket_dir": "artifacts/tickets",
        }

    app = create_app(configurable_factory=factory, checkpointer=MemorySaver())
    client = TestClient(app)

    assert client.get("/sessions/does-not-exist").status_code == 404
    assert client.post("/sessions/does-not-exist/resume", json={"answer": "x"}).status_code == 404


def test_node_failure_mid_stream_ends_with_error_event(tmp_path, monkeypatch):
    """If a node raises after the stream has started (here the monitor's model load),
    the SSE stream must close with a terminal `error` event, not truncate."""
    from sentinel.agents import monitor
    from sentinel.api.app import create_app

    def boom(model_path):
        raise FileNotFoundError("no model on disk")

    monkeypatch.setattr(monitor, "load_predict", boom)

    def factory():
        return {
            "provider_smart": OneShotProvider(['{"all_defaults": true}']),
            "provider_cheap": FixedProvider(),
            "train_fn": lambda c: _fake_run(),
            "ticket_dir": str(tmp_path),
        }

    app = create_app(configurable_factory=factory, checkpointer=MemorySaver())
    client = TestClient(app)

    tid = client.post("/sessions").headers["x-thread-id"]
    r = client.post(f"/sessions/{tid}/resume", json={"answer": "yes defaults"})
    assert r.status_code == 200  # the stream started fine
    events = list(_sse_events(r))
    assert events[-1]["event"] == "error"
    assert "FileNotFoundError" in events[-1]["data"]["message"]
