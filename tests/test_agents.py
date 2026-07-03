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
from sentinel.agents.interviewer import MAX_NONANSWERS, _resolve_field, run_interview
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


class RecordingAsk:
    """A fake `ask` that returns queued user replies and records the bot prompts."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        i = min(len(self.prompts) - 1, len(self.replies) - 1)
        return self.replies[i]


def _turn(classification: str, reply: str, value=None) -> str:
    """Build one per-turn interviewer decision as the LLM would return it."""
    return json.dumps({"classification": classification, "reply": reply, "value": value})


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


# --- interviewer: turn-by-turn conversation ------------------------------

_Q = "Below how many cycles of remaining life should we alert?"  # the threshold question


def test_unclear_reply_gets_immediate_clarify_and_default_offer():
    # The bot's very next message (same turn) must push back AND offer the default,
    # not batch the reaction to the end.
    ask = RecordingAsk(["you tell me", "50 cycles"])
    clarify = "I need a number here - roughly how many cycles? Or I can use the default of 30 cycles - want that?"
    provider = QueueProvider([_turn("UNCLEAR", clarify), _turn("CLEAR", "Got it - alerting below 50.", 50)])

    value, ack = _resolve_field("failure_threshold", _Q, ask, provider, preamble="")

    # The second bot message (right after the unclear reply) is the clarify+default offer.
    assert ask.prompts[1] == clarify
    assert "default of 30" in ask.prompts[1]
    assert value == 50  # the later clear answer resolves the field


def test_user_question_is_answered_from_glossary_then_reasked_not_consumed():
    # Four questions in a row must each be answered and re-asked WITHOUT being
    # counted as non-answers; only the final clear reply resolves the field.
    explain = "RMSE is the model's typical error in cycles (lower is better). A common target is under 20. So - what would success look like?"
    ask = RecordingAsk(["what does RMSE mean?"] * 4 + ["RMSE under 20 cycles"])
    provider = QueueProvider(
        [_turn("QUESTION", explain)] * 4 + [_turn("CLEAR", "Great - success is RMSE under 20.", "RMSE under 20 cycles")]
    )

    value, ack = _resolve_field("success_metric", "What result would make this a success?", ask, provider, preamble="")

    # The bot answered the question and re-asked in the same turn...
    assert ask.prompts[1] == explain
    # ...and did NOT consume the question as the answer, nor hit the non-answer bound
    # (4 questions > MAX_NONANSWERS would have defaulted if they counted).
    assert value == "RMSE under 20 cycles"


def test_wants_default_uses_the_default_with_a_one_line_ack():
    ask = RecordingAsk(["you decide"])
    ack_msg = "No problem - I'll alert below 30 cycles, a common default."
    provider = QueueProvider([_turn("WANTS_DEFAULT", ack_msg, None)])

    value, ack = _resolve_field("failure_threshold", _Q, ask, provider, preamble="")

    assert value == 30  # DEFAULTS["failure_threshold"]
    assert ack == ack_msg
    assert len(ask.prompts) == 1  # resolved in one turn, no extra pushback


def test_field_loop_is_bounded_and_falls_back_to_default():
    # Endless non-answers must terminate at the default, never loop forever.
    ask = RecordingAsk(["nope"])  # every reply is a non-answer
    provider = QueueProvider([_turn("UNCLEAR", "Still need a number - or the default of 30?")])

    value, ack = _resolve_field("failure_threshold", _Q, ask, provider, preamble="")

    assert value == 30
    # Bounded: opening ask + (MAX_NONANSWERS - 1) clarify asks, then fall back.
    assert len(ask.prompts) == MAX_NONANSWERS
    assert "default of 30" in ack


def test_run_interview_walks_all_fields_and_carries_acks_forward():
    ask = RecordingAsk(["turbofan RUL", "30 cycles", "after every run", "RMSE under 20"])
    provider = QueueProvider(
        [
            _turn("CLEAR", "Got it, turbofan RUL.", "turbofan RUL"),
            _turn("CLEAR", "Okay, alerting below 30.", 30),
            _turn("CLEAR", "Sure, a report each run.", "after every run"),
            _turn("CLEAR", "Great, RMSE under 20 it is.", "RMSE under 20"),
        ]
    )
    notes: list[str] = []

    cfg = run_interview(ask, provider, notify=notes.append)

    # One bot turn per field (no second batch pass).
    assert len(ask.prompts) == 4
    # Field 1's acknowledgement leads in to field 2's question (continuous chat).
    assert ask.prompts[1].startswith("Got it, turbofan RUL.")
    assert "Below how many cycles" in ask.prompts[1]
    # Values collected + advanced knobs defaulted; last ack closes the chat.
    assert cfg.framing == "turbofan RUL"
    assert cfg.failure_threshold == 30
    assert cfg.rul_cap == 125 and cfg.window == 5
    assert notes == ["Great, RMSE under 20 it is."]


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

    # The interviewer now converses per field; feed one CLEAR decision per field.
    interview_turns = QueueProvider(
        [
            _turn("CLEAR", "ok", "turbofan RUL"),
            _turn("CLEAR", "ok", 30),
            _turn("CLEAR", "ok", "each run"),
            _turn("CLEAR", "ok", "RMSE < 20"),
        ]
    )
    configurable = {
        "ask": lambda q: "scripted",
        "provider_smart": interview_turns,
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

    interview_turns = QueueProvider([_turn("CLEAR", "ok", 30)])  # resolves every field
    configurable = {
        "ask": lambda q: "x",
        "provider_smart": interview_turns,
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
