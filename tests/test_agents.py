"""Offline tests for the surviving agent-layer leaf helpers."""
from __future__ import annotations

import pandas as pd

from sentinel.agents.monitor import decide, run_monitor
from sentinel.agents.report_writer import write_report
from sentinel.agents.state import InterviewConfig


class FakeProvider:
    """A chat model that returns canned content and records the last prompt."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_messages: list[dict] | None = None

    def invoke(self, messages: list[dict], **kwargs):
        self.last_messages = messages

        class _Response:
            content = self.reply

        return _Response()


def _fake_train_result():
    from sentinel.core.automl import TrainResult

    leaderboard = pd.DataFrame(
        {
            "Model": ["Extra Trees", "Ridge"],
            "MAE": [11.9, 15.0],
            "RMSE": [17.1, 20.0],
            "R2": [0.82, 0.7],
        }
    )
    return TrainResult(
        leaderboard=leaderboard,
        best_model=object(),
        metrics={"rmse": 17.1, "mae": 11.9, "r2": 0.82},
        model_path="artifacts/rul_model.pkl",
        metrics_path="artifacts/metrics.json",
    )


def test_write_report_feeds_metrics_to_provider():
    provider = FakeProvider("Report body.")
    assert write_report(_fake_train_result(), provider) == "Report body."
    user = provider.last_messages[-1]["content"]
    assert "17.10" in user and "11.90" in user and "0.820" in user


def test_write_report_prompt_is_grounding_constrained():
    provider = FakeProvider("ok")
    write_report(_fake_train_result(), provider)
    system = provider.last_messages[0]["content"]
    user = provider.last_messages[-1]["content"]
    assert "Root Mean Squared Error" in system + user
    assert "square root" in system.lower()
    assert "METRICS" in user


def test_write_report_prompt_forbids_metric_as_prediction():
    provider = FakeProvider("ok")
    write_report(_fake_train_result(), provider)
    system = provider.last_messages[0]["content"]
    glossary = provider.last_messages[-1]["content"]
    assert "prediction of remaining life" in system.lower()
    assert "fail" in system.lower()
    assert "not a prediction" in glossary.lower()


def test_write_report_frames_metrics_as_held_out_test():
    provider = FakeProvider("ok")
    write_report(_fake_train_result(), provider)
    system = provider.last_messages[0]["content"].lower()
    user = provider.last_messages[-1]["content"].lower()
    assert "held-out test" in user
    assert "cross-validation" in user or "cross validation" in user
    assert "held-out test" in system
    assert "never saw" in system or "never seen" in system
    assert "above or below" in system or "under the target" in system


def test_success_verdict_decided_in_code():
    from sentinel.agents.report_writer import _success_verdict

    metrics = {"rmse": 17.09, "mae": 11.95, "r2": 0.818}
    assert _success_verdict("held-out RMSE under 20 cycles", metrics) is True
    assert _success_verdict("RMSE under 15", metrics) is False
    assert _success_verdict("R2 above 0.9", metrics) is False
    assert _success_verdict("just make it accurate", metrics) is None


def test_write_report_passes_precomputed_verdict_not_a_comparison():
    provider = FakeProvider("ok")
    config = InterviewConfig(
        framing="turbofan RUL",
        failure_threshold=30,
        reporting_cadence="each run",
        success_metric="held-out RMSE under 20 cycles",
    )
    write_report(_fake_train_result(), provider, config)
    user = provider.last_messages[-1]["content"]
    assert "SUCCESS CHECK" in user
    assert "MEETS the user" in user
    assert "do NOT re-compare" in user


def test_report_only_cites_grounded_numbers():
    import re

    grounded_reply = (
        "The Extra Trees model won. On average it is off by 11.90 cycles "
        "(MAE), with an RMSE of 17.10 cycles and an R2 of 0.820."
    )
    provider = FakeProvider(grounded_reply)
    write_report(_fake_train_result(), provider)
    allowed = {"11.90", "17.10", "0.820"}
    numbers = set(re.findall(r"\d+\.\d+", grounded_reply))
    assert numbers <= allowed


def test_decide_thresholds():
    assert decide(20, threshold=30) == "alert"
    assert decide(30, threshold=30) == "alert"
    assert decide(45, threshold=30) == "report"
    assert decide(80, threshold=30) == "ok"


def test_run_monitor_files_tickets_only_on_alerts(tmp_path):
    rows = pd.DataFrame({"unit": [1, 2, 3], "RUL": [10, 45, 90]})
    events = run_monitor(
        rows,
        lambda frame: list(frame["RUL"]),
        threshold=30,
        ticket_dir=tmp_path,
    )
    assert {event["unit"]: event["decision"] for event in events} == {
        1: "alert",
        2: "report",
    }
    assert [path.name for path in tmp_path.glob("ticket_unit_*.json")] == [
        "ticket_unit_1.json"
    ]
