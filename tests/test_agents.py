"""Fast, offline tests for the agent layer's deterministic parts.

No live LLM, no PyCaret, no network: the `Provider.complete` calls are faked,
training is a stub, and prediction is a plain function. These cover the pieces
that are ours to get right - provider selection, graph routing, the interviewer
extraction, the monitor threshold logic, the mock ticket action, and that the
whole graph wires interview -> train -> report -> monitor and terminates.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sentinel.agents.graph import build_graph, route
from sentinel.agents.interviewer import collect_config
from sentinel.agents.monitor import decide, run_monitor
from sentinel.agents.report_writer import write_report
from sentinel.agents.state import InterviewConfig
from sentinel.llm.provider import (
    AnthropicProvider,
    GroqProvider,
    get_provider,
)


class FakeProvider:
    """A `Provider` that returns a canned string and records the last prompt."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_messages: list[dict] | None = None

    def complete(self, messages: list[dict], **kwargs) -> str:
        self.last_messages = messages
        return self.reply


# --- provider selection --------------------------------------------------


def test_get_provider_selects_by_env(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "anthropic")
    # Avoid constructing a real SDK client / needing a key.
    monkeypatch.setattr(AnthropicProvider, "__init__", lambda self, model, api_key=None: None)
    assert isinstance(get_provider("smart"), AnthropicProvider)

    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "groq")
    monkeypatch.setattr(GroqProvider, "__init__", lambda self, model, api_key=None: None)
    assert isinstance(get_provider("cheap"), GroqProvider)


def test_get_provider_maps_tier_to_model(monkeypatch):
    captured = {}
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(
        AnthropicProvider, "__init__", lambda self, model, api_key=None: captured.update(model=model)
    )
    get_provider("smart")
    assert captured["model"] == "claude-sonnet-5"
    get_provider("cheap")
    assert captured["model"] == "claude-haiku-4-5"


def test_get_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError):
        get_provider("smart")
    with pytest.raises(ValueError):
        get_provider("nonsense", name="groq")


# --- orchestrator routing ------------------------------------------------


def test_route_follows_event_lifecycle():
    assert route({"event": "start"}) == "interviewer"
    assert route({}) == "interviewer"  # no event yet
    assert route({"event": "interview_done"}) == "trainer"
    assert route({"event": "run_finished"}) == "report_writer"
    assert route({"event": "run_failed"}) == "report_writer"
    assert route({"event": "report_ready"}) == "monitor"
    # monitor_done / failed_reported both terminate.
    from langgraph.graph import END

    assert route({"event": "monitor_done"}) == END
    assert route({"event": "failed_reported"}) == END


def test_route_rejects_unknown_event():
    with pytest.raises(ValueError):
        route({"event": "surprise"})


# --- interviewer extraction ----------------------------------------------


def test_collect_config_parses_llm_json():
    provider = FakeProvider(
        'Here you go: {"framing": "turbofan RUL", "failure_threshold": 25, '
        '"reporting_cadence": "daily", "success_metric": "RMSE < 20", '
        '"rul_cap": 100, "window": 7}'
    )
    cfg = collect_config({}, provider)
    assert cfg.failure_threshold == 25
    assert cfg.rul_cap == 100
    assert cfg.window == 7
    assert cfg.framing == "turbofan RUL"


def test_collect_config_falls_back_on_bad_json():
    cfg = collect_config(
        {"framing": "raw answer", "failure_threshold": "ignored"},
        FakeProvider("not json at all"),
    )
    # Defaults kick in; free-text framing falls back to the raw answer.
    assert cfg.failure_threshold == 30
    assert cfg.rul_cap == 125
    assert cfg.window == 5
    assert cfg.framing == "raw answer"


# --- report writer --------------------------------------------------------


def _fake_train_result():
    from sentinel.core.automl import TrainResult

    lb = pd.DataFrame({"Model": ["Extra Trees", "Ridge"], "MAE": [11.9, 15.0], "RMSE": [17.1, 20.0], "R2": [0.82, 0.7]})
    return TrainResult(
        leaderboard=lb,
        best_model=object(),
        metrics={"rmse": 17.1, "mae": 11.9, "r2": 0.82},
        model_path="artifacts/rul_model.pkl",
        metrics_path="artifacts/metrics.json",
    )


def test_write_report_feeds_metrics_to_provider():
    provider = FakeProvider("The Extra Trees model predicts RUL within ~17 cycles.")
    out = write_report(_fake_train_result(), provider)
    assert out == "The Extra Trees model predicts RUL within ~17 cycles."
    # The prompt actually carried the metrics the report should be grounded in.
    prompt = provider.last_messages[0]["content"]
    assert "17.10" in prompt and "0.820" in prompt


# --- monitor threshold logic + mock action -------------------------------


def test_decide_thresholds():
    assert decide(20, threshold=30) == "alert"  # at/under threshold
    assert decide(30, threshold=30) == "alert"
    assert decide(45, threshold=30) == "report"  # within warn band (<= 2x)
    assert decide(80, threshold=30) == "ok"  # healthy


def test_run_monitor_files_tickets_only_on_alerts(tmp_path):
    rows = pd.DataFrame({"unit": [1, 2, 3], "RUL": [10, 45, 90]})

    def predict(frame):
        # Predict RUL == the true RUL column, so decisions are deterministic.
        return list(frame["RUL"])

    events = run_monitor(rows, predict, threshold=30, ticket_dir=tmp_path)
    decisions = {e["unit"]: e["decision"] for e in events}
    assert decisions == {1: "alert", 2: "report"}  # unit 3 (ok) not recorded
    # Exactly one ticket file, for the alerting unit.
    tickets = list(tmp_path.glob("ticket_unit_*.json"))
    assert [p.name for p in tickets] == ["ticket_unit_1.json"]


# --- full graph wiring (offline, faked deps) -----------------------------


def test_graph_runs_interview_to_monitor(tmp_path):
    from sentinel.agents.training import TrainingRun

    result = _fake_train_result()
    test_eval = pd.DataFrame({"unit": [1, 2], "RUL": [10, 90]})
    run = TrainingRun(result=result, test_eval=test_eval, predict=lambda f: list(f["RUL"]))

    interview_json = (
        '{"framing": "turbofan RUL", "failure_threshold": 30, '
        '"reporting_cadence": "each run", "success_metric": "RMSE < 20"}'
    )

    configurable = {
        "ask": lambda q: "scripted",
        "provider_smart": FakeProvider(interview_json),
        "provider_cheap": FakeProvider("Report: the model is good."),
        "train_fn": lambda cfg: run,
        "ticket_dir": str(tmp_path),
    }

    final = build_graph().invoke({"event": "start"}, config={"configurable": configurable})

    assert isinstance(final["config"], InterviewConfig)
    assert final["config"].failure_threshold == 30
    assert final["report"] == "Report: the model is good."
    assert final["event"] == "monitor_done"
    # Unit 1 (RUL 10) alerts and files a ticket; unit 2 (RUL 90) is ok.
    assert [a["unit"] for a in final["alerts"]] == [1]
    assert (tmp_path / "ticket_unit_1.json").exists()


def test_graph_reports_training_failure():
    def boom(cfg):
        raise RuntimeError("PyCaret exploded")

    configurable = {
        "ask": lambda q: "x",
        "provider_smart": FakeProvider('{"failure_threshold": 30}'),
        "provider_cheap": FakeProvider("unused"),
        "train_fn": boom,
        "ticket_dir": "artifacts/tickets",
    }
    final = build_graph().invoke({"event": "start"}, config={"configurable": configurable})

    # Failure is reported and the graph stops before monitoring (no model).
    assert final["event"] == "failed_reported"
    assert "PyCaret exploded" in final["error"]
    assert "did not complete" in final["report"]
    assert "alerts" not in final
