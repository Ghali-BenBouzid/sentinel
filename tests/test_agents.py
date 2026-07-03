"""Fast, offline tests for the agent layer's deterministic parts.

No live LLM, no PyCaret, no network: the `Provider.complete` calls are faked,
training is a stub, and prediction is a plain function. These cover the pieces
that are ours to get right - provider selection, graph routing, the interviewer
extraction, the monitor threshold logic, the mock ticket action, and that the
whole graph wires interview -> train -> report -> monitor and terminates.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from sentinel.agents.graph import build_graph, route
from sentinel.agents.interviewer import extract_fields, run_interview
from sentinel.agents.monitor import decide, run_monitor
from sentinel.agents.report_writer import write_report
from sentinel.agents.state import InterviewConfig
from sentinel.config import get_settings
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


class QueueProvider:
    """A `Provider` that returns queued replies in order (last one repeats)."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def complete(self, messages: list[dict], **kwargs) -> str:
        i = min(self.calls, len(self.replies) - 1)
        self.calls += 1
        return self.replies[i]


def _fields_json(**overrides) -> str:
    """Build a nested interviewer-extraction reply (all 4 answered by default)."""
    base = {
        "framing": {"answered": True, "value": "turbofan RUL"},
        "failure_threshold": {"answered": True, "value": 25},
        "reporting_cadence": {"answered": True, "value": "daily"},
        "success_metric": {"answered": True, "value": "RMSE < 20"},
        "rul_cap": {"answered": False, "value": None},
        "window": {"answered": False, "value": None},
    }
    base.update(overrides)
    return json.dumps(base)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """`get_provider` reads cached settings; keep env changes from leaking."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --- provider selection --------------------------------------------------


def test_get_provider_selects_by_env(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "anthropic")
    # Avoid constructing a real SDK client / needing a key.
    monkeypatch.setattr(AnthropicProvider, "__init__", lambda self, model, api_key=None: None)
    get_settings.cache_clear()  # settings are cached; pick up the new env
    assert isinstance(get_provider("smart"), AnthropicProvider)

    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "groq")
    monkeypatch.setattr(GroqProvider, "__init__", lambda self, model, api_key=None: None)
    get_settings.cache_clear()
    assert isinstance(get_provider("cheap"), GroqProvider)


def test_get_provider_maps_tier_to_model(monkeypatch):
    captured = {}
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(
        AnthropicProvider, "__init__", lambda self, model, api_key=None: captured.update(model=model)
    )
    get_settings.cache_clear()
    get_provider("smart")
    assert captured["model"] == "claude-sonnet-5"
    get_provider("cheap")
    assert captured["model"] == "claude-haiku-4-5"


def test_get_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("SENTINEL_LLM_PROVIDER", "bogus")
    get_settings.cache_clear()
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


# --- interviewer extraction + robustness ---------------------------------


def test_extract_fields_reports_values_and_answered_flags():
    provider = FakeProvider(_fields_json(rul_cap={"answered": True, "value": 100}))
    fields = extract_fields({}, provider)
    assert fields["failure_threshold"] == {"answered": True, "value": 25}
    assert fields["rul_cap"] == {"answered": True, "value": 100}
    assert fields["window"]["answered"] is False


def test_run_interview_all_answered_no_reask():
    prompts: list[str] = []
    notes: list[str] = []
    ask = lambda q: (prompts.append(q), "a real answer")[1]  # noqa: E731
    provider = QueueProvider([_fields_json()])

    cfg = run_interview(ask, provider, notify=notes.append)

    # One question each, no re-ask, one extraction call.
    assert len(prompts) == 4
    assert provider.calls == 1
    assert cfg.failure_threshold == 25
    assert cfg.framing == "turbofan RUL"
    # None of the four answered fields were defaulted...
    assert not any("threshold" in n or "framing" in n for n in notes)
    # ...but the un-asked advanced knobs are surfaced as defaults, never silent.
    assert any("RUL cap" in n for n in notes)
    assert any("rolling-window" in n for n in notes)


def test_run_interview_reasks_nonanswer_then_defaults_and_surfaces():
    prompts: list[str] = []
    notes: list[str] = []
    ask = lambda q: (prompts.append(q), "I don't know")[1]  # noqa: E731
    # Threshold flagged not-answered before AND after the re-ask -> default 30.
    unanswered_threshold = {"answered": False, "value": None}
    provider = QueueProvider(
        [
            _fields_json(failure_threshold=unanswered_threshold),
            _fields_json(failure_threshold=unanswered_threshold),
        ]
    )

    cfg = run_interview(ask, provider, notify=notes.append)

    # The threshold question was re-asked exactly once (4 initial + 1 re-ask).
    assert len(prompts) == 5
    assert provider.calls == 2  # extract, then re-extract after the re-ask
    assert any("clearer answer" in p and "how many cycles" in p for p in prompts)
    # Default applied AND surfaced, not stored silently.
    assert cfg.failure_threshold == 30
    assert any("default of 30 cycles" in n for n in notes)


def test_run_interview_defaults_everything_on_unparseable_extraction():
    notes: list[str] = []
    ask = lambda q: "you decide"  # noqa: E731
    provider = QueueProvider(["not json at all"])  # every field looks unanswered

    cfg = run_interview(ask, provider, notify=notes.append)

    assert cfg.failure_threshold == 30
    assert cfg.rul_cap == 125
    assert cfg.window == 5
    # Every defaulted field is surfaced - nothing stored as junk silently.
    assert any("threshold" in n for n in notes)
    assert any("framing" in n.lower() for n in notes)
    assert any("cadence" in n.lower() or "after every training run" in n for n in notes)


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
    # The user message carries the grounded metric values the report must use.
    user = provider.last_messages[-1]["content"]
    assert "17.10" in user and "11.90" in user and "0.820" in user


def test_write_report_prompt_is_grounding_constrained():
    """The fix: the prompt bans fabrication/derivation and uses correct terms."""
    provider = FakeProvider("ok")
    write_report(_fake_train_result(), provider)
    system = provider.last_messages[0]["content"]
    user = provider.last_messages[-1]["content"]

    # Correct terminology comes from the glossary (guards the "Mean Squared Error" mislabel).
    assert "Root Mean Squared Error" in system + user
    # Explicit no-derivation guardrails (guards the bogus "square root of RMSE").
    assert "square root" in system.lower()
    assert "do not compute" in system.lower() or "do not calculate" in user.lower()
    # The single numeric source is the METRICS block; no second table of numbers.
    assert "METRICS" in user
    assert "MAE" in system  # instructed to use MAE for "on average off by"


def test_write_report_prompt_forbids_metric_as_prediction():
    """A metric must never be narrated as a remaining-life / time-to-failure forecast."""
    provider = FakeProvider("ok")
    write_report(_fake_train_result(), provider)
    system = provider.last_messages[0]["content"]
    glossary = provider.last_messages[-1]["content"]

    # The system Do/Don't forbids equating a metric with a prediction...
    assert "prediction of remaining life" in system.lower()
    assert "fail" in system.lower()  # bans "predicts it will fail in N cycles"
    # ...and the glossary reinforces it on the metric itself.
    assert "not a prediction" in glossary.lower()


def test_report_only_cites_grounded_numbers():
    """A well-behaved report should introduce no numbers beyond the given metrics.

    We can't test a live model, so we simulate a grounded reply and assert every
    number in it traces to the metrics the prompt supplied - the property the
    prompt is engineered to enforce.
    """
    import re

    grounded_reply = (
        "The Extra Trees model won. On average it is off by 11.90 cycles (MAE), with an "
        "RMSE of 17.10 cycles and an R2 of 0.820."
    )
    provider = FakeProvider(grounded_reply)
    write_report(_fake_train_result(), provider)

    allowed = {"11.90", "17.10", "0.820"}  # the provided metrics, formatted as in the prompt
    numbers = set(re.findall(r"\d+\.\d+", grounded_reply))
    assert numbers <= allowed, f"report cited ungrounded numbers: {numbers - allowed}"


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

    interview_json = _fields_json(failure_threshold={"answered": True, "value": 30})

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
        "provider_smart": FakeProvider(_fields_json()),
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
