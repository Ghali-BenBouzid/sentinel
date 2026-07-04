"""Offline tests for the dashboard's GraphRunner (the threaded queue bridge).

No live LLM, no PyCaret, no Streamlit: the graph is real but its dependencies are
the same fakes tests/test_agents.py uses. We drive the interview by polling for
prompts and calling answer(), exactly as the Streamlit view does, and assert the
bridge carries a full run to a completed final_state.
"""

from __future__ import annotations

import json
import time

import pandas as pd

from sentinel.agents.graph import build_graph
from sentinel.agents.state import InterviewConfig
from sentinel.agents.training import TrainingRun
from sentinel.core.automl import TrainResult
from sentinel.dashboard.runner import Event, GraphRunner


class QueueProvider:
    """A Provider that returns queued replies in order (last one repeats)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, messages, **kwargs):
        i = min(self.calls, len(self.replies) - 1)
        self.calls += 1
        return self.replies[i]


class FakeProvider:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, messages, **kwargs):
        return self.reply


def _turn(classification, reply, value=None, deduced=None):
    return json.dumps(
        {"classification": classification, "reply": reply, "value": value, "deduced": deduced or []}
    )


def _gate(all_defaults):
    return json.dumps({"all_defaults": all_defaults})


def _fake_run():
    lb = pd.DataFrame({"Model": ["Extra Trees"], "MAE": [11.9], "RMSE": [17.1], "R2": [0.82]})
    result = TrainResult(
        leaderboard=lb,
        best_model=object(),
        metrics={"rmse": 17.1, "mae": 11.9, "r2": 0.82},
        model_path="artifacts/rul_model.pkl",
        metrics_path="artifacts/metrics.json",
    )
    test_eval = pd.DataFrame({"unit": [1, 2], "RUL": [10, 90]})
    return TrainingRun(result=result, test_eval=test_eval, predict=lambda f: list(f["RUL"]))


def _wait(pred, timeout=5.0):
    """Spin until pred() is true or timeout (keeps threaded tests from hanging)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_runner_bridges_full_interview_to_completed_state(tmp_path):
    # Gate declined, then one CLEAR turn per field (4 fields) -> 5 asks total.
    smart = QueueProvider(
        [
            _gate(False),
            _turn("CLEAR", "ok", "turbofan RUL"),
            _turn("CLEAR", "ok", 30),
            _turn("CLEAR", "ok", "each run"),
            _turn("CLEAR", "ok", "RMSE < 20"),
        ]
    )
    runner = GraphRunner(
        graph=build_graph(),
        provider_smart=smart,
        provider_cheap=FakeProvider("Report: the model is good."),
        train_fn=lambda cfg: _fake_run(),
        ticket_dir=str(tmp_path),
    )
    runner.start()

    answers = iter(["no, let's tune it", "turbofan RUL", "30", "each run", "RMSE < 20"])
    # Answer every prompt the interviewer emits until the graph completes.
    while True:
        assert _wait(lambda: runner.pending_prompt() is not None or runner.done), "runner stalled"
        if runner.done:
            break
        runner.answer(next(answers, ""))

    assert _wait(lambda: runner.done)
    assert runner.error is None
    final = runner.final_state()
    assert isinstance(final["config"], InterviewConfig)
    assert final["config"].failure_threshold == 30
    assert final["report"] == "Report: the model is good."
    assert final["event"] == "monitor_done"
    assert [a["unit"] for a in final["alerts"]] == [1]
    assert (tmp_path / "ticket_unit_1.json").exists()
    # The training lifecycle markers were emitted for the UI to react to.
    assert runner.saw("training_started") and runner.saw("training_finished")


def test_runner_surfaces_training_failure_without_hanging(tmp_path):
    def boom(cfg):
        raise RuntimeError("PyCaret exploded")

    smart = QueueProvider([_gate(True)])  # take the all-defaults fast path, skip Q&A
    runner = GraphRunner(
        graph=build_graph(),
        provider_smart=smart,
        provider_cheap=FakeProvider("unused"),
        train_fn=boom,
        ticket_dir=str(tmp_path),
    )
    runner.start()

    # Gate is the only prompt; accept defaults, then the graph trains (and fails).
    assert _wait(lambda: runner.pending_prompt() is not None)
    runner.answer("yes, just use defaults")

    assert _wait(lambda: runner.done), "runner hung on training failure"
    final = runner.final_state()
    # The graph itself reports the failure and stops before the monitor.
    assert final["event"] == "failed_reported"
    assert "PyCaret exploded" in final["error"]
    assert "alerts" not in final
    assert runner.error is None  # a graph-handled failure is NOT an unexpected thread error


def test_poll_returns_notify_events(tmp_path):
    # All-defaults path announces each applied default via notify -> Event(kind="notify").
    smart = QueueProvider([_gate(True)])
    runner = GraphRunner(
        graph=build_graph(),
        provider_smart=smart,
        provider_cheap=FakeProvider("Report."),
        train_fn=lambda cfg: _fake_run(),
        ticket_dir=str(tmp_path),
    )
    runner.start()
    assert _wait(lambda: runner.pending_prompt() is not None)
    runner.answer("yes, use defaults")
    assert _wait(lambda: runner.done)

    kinds = set()
    # Drain everything the run emitted (poll is incremental).
    for _ in range(50):
        for ev in runner.poll():
            assert isinstance(ev, Event)
            kinds.add(ev.kind)
        if runner.done and not runner.pending_prompt():
            break
        time.sleep(0.01)
    assert "notify" in kinds
