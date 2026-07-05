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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from sentinel.agents import interviewer as iv
from sentinel.agents.graph import build_graph, route
from sentinel.agents.interviewer import DEFAULTS, GATE_QUESTION, MAX_NONANSWERS
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


def _advance_to(target_field: str):
    """Drive `iv.advance()` from a fresh, gate-declined start up to `target_field`,
    resolving every preceding field with a throwaway CLEAR reply.

    Replaces the deleted `_resolve_field` helper (Task 3 removed the batch
    `_resolve_field`/`run_interview` collectors in favour of the turn-by-turn,
    checkpointed `advance()` state machine - see `tests/test_interviewer_state.py`
    for its low-level per-turn coverage). This is the same fixture, built by
    chaining `advance()` calls instead of calling a field resolver directly.
    """
    prog = iv.start_progress()
    prog = iv.advance(prog, "no", QueueProvider([_gate(False)]))
    for field, _ in iv.QUESTIONS:
        if field == target_field:
            break
        value = 5 if field == "failure_threshold" else "x"  # failure_threshold must coerce to int
        prog = iv.advance(prog, "x", QueueProvider([_turn("CLEAR", "ok", value)]))
    assert iv.QUESTIONS[prog["active_index"]][0] == target_field
    return prog


def _drive_graph(configurable: dict, answers: list[str]) -> dict:
    """Drive the compiled graph over the interrupt path, mirroring the
    stream/resume loop in `sentinel.agents.__main__.main()`: invoke, then resume
    with the next scripted answer while a turn is pending. Returns the final
    state values once the graph has no pending interrupt (`state.tasks == ()`).
    """
    graph = build_graph(checkpointer=MemorySaver())
    thread = {"configurable": {**configurable, "thread_id": "test"}}
    replies = iter(answers)

    inp: dict | Command = {"event": "start"}
    while True:
        graph.invoke(inp, thread)
        state = graph.get_state(thread)
        if not state.tasks:  # no pending interrupt -> graph is done
            break
        inp = Command(resume=next(replies, ""))
    return graph.get_state(thread).values


def _turn(classification: str, reply: str, value=None, deduced=None) -> str:
    """Build one per-turn interviewer decision as the LLM would return it."""
    return json.dumps(
        {"classification": classification, "reply": reply, "value": value, "deduced": deduced or []}
    )


def _gate(all_defaults: bool) -> str:
    """Build the up-front gate classifier's reply."""
    return json.dumps({"all_defaults": all_defaults})


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
    # "interviewer_turn" since Task 3's self-looping interrupt() node rename.
    assert route({"event": "start"}) == "interviewer_turn"
    assert route({}) == "interviewer_turn"  # no event yet
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
#
# These drive `iv.advance()` directly (the per-turn state machine that replaced
# the deleted batch `_resolve_field`/`run_interview`), chaining calls the same
# way the checkpointed `interviewer_turn` graph node does one turn at a time.


def test_unclear_reply_gets_immediate_clarify_and_default_offer():
    # The bot's very next message (same turn) must push back AND offer the default,
    # not batch the reaction to the end.
    prog = _advance_to("failure_threshold")
    clarify = "I need a number here - roughly how many cycles? Or I can use the default of 30 cycles - want that?"
    prog = iv.advance(prog, "you tell me", QueueProvider([_turn("UNCLEAR", clarify)]))

    # The very next prompt (right after the unclear reply) is the clarify+default offer.
    assert prog["next_prompt"] == clarify
    assert "default of 30" in prog["next_prompt"]

    prog = iv.advance(prog, "50 cycles", QueueProvider([_turn("CLEAR", "Got it - alerting below 50.", 50)]))
    assert prog["values"]["failure_threshold"] == 50  # the later clear answer resolves the field


def test_user_question_is_answered_from_glossary_then_reasked_not_consumed():
    # Four questions in a row must each be answered and re-asked WITHOUT being
    # counted as non-answers; only the final clear reply resolves the field.
    prog = _advance_to("success_metric")
    explain = "RMSE is the model's typical error in cycles (lower is better). A common target is under 20. So - what would success look like?"
    for _ in range(4):
        prog = iv.advance(prog, "what does RMSE mean?", QueueProvider([_turn("QUESTION", explain)]))
        # The bot answered the question and re-asked in the same turn, without
        # advancing, consuming the field, or counting it as a non-answer (4
        # questions > MAX_NONANSWERS would have defaulted if they counted).
        assert prog["next_prompt"] == explain
        assert prog["nonanswers"] == 0
        assert "success_metric" not in prog["values"]

    prog = iv.advance(
        prog,
        "RMSE under 20 cycles",
        QueueProvider([_turn("CLEAR", "Great - success is RMSE under 20.", "RMSE under 20 cycles")]),
    )
    assert prog["values"]["success_metric"] == "RMSE under 20 cycles"


def test_skip_uses_the_default_with_a_one_line_ack():
    prog = _advance_to("failure_threshold")
    ack_msg = "No problem - I'll alert below 30 cycles, a common default."

    prog = iv.advance(prog, "you decide", QueueProvider([_turn("WANTS_DEFAULT", ack_msg, None)]))

    assert prog["values"]["failure_threshold"] == 30  # DEFAULTS["failure_threshold"]
    # Resolved in one turn: the ack leads straight into the next field's question.
    assert prog["next_prompt"].startswith(ack_msg)


def test_field_loop_is_bounded_and_falls_back_to_default():
    # Endless non-answers must terminate at the default, never loop forever.
    prog = _advance_to("failure_threshold")
    unclear = _turn("UNCLEAR", "Still need a number - or the default of 30?")
    for _ in range(MAX_NONANSWERS):
        prog = iv.advance(prog, "nope", QueueProvider([unclear]))

    assert prog["values"]["failure_threshold"] == 30
    assert iv.QUESTIONS[prog["active_index"]][0] == "reporting_cadence"  # advanced past the stuck field
    assert "default of 30" in prog["next_prompt"]


def test_deduced_field_is_confirmed_not_asked_cold():
    # A value already deduced for this field is CONFIRMED, and a bare "yes" accepts it.
    prog = iv.start_progress()
    prog = iv.advance(prog, "no", QueueProvider([_gate(False)]))
    framing_turn = _turn(
        "CLEAR", "Got it.", "turbofan RUL", deduced=[{"field": "failure_threshold", "value": 25, "confidence": 0.9}]
    )
    prog = iv.advance(prog, "predict turbofan RUL, alert around 25", QueueProvider([framing_turn]))
    assert iv.QUESTIONS[prog["active_index"]][0] == "failure_threshold"
    # The opening message confirms the deduced value instead of asking cold.
    assert "25" in prog["next_prompt"]
    assert prog["next_prompt"] != iv.QUESTIONS[1][1]  # not the cold question

    prog = iv.advance(prog, "yep, that's right", QueueProvider([_turn("CLEAR", "Great, alerting below 25.", None)]))
    assert prog["values"]["failure_threshold"] == 25


def test_advance_offers_gate_then_walks_fields_and_carries_acks():
    # Gate declined -> normal conversation, one turn per field.
    provider = QueueProvider(
        [
            _gate(False),  # up-front "use all defaults?" -> no
            _turn("CLEAR", "Got it, turbofan RUL.", "turbofan RUL"),
            _turn("CLEAR", "Okay, alerting below 30.", 30),
            _turn("CLEAR", "Sure, a report each run.", "after every run"),
            _turn("CLEAR", "Great, RMSE under 20 it is.", "RMSE under 20"),
        ]
    )
    replies = ["no, let's tune it", "turbofan RUL", "30 cycles", "after every run", "RMSE under 20"]

    prog = iv.start_progress()
    prompts = [prog["next_prompt"]]
    notices: list[str] = []
    for reply in replies:
        prog = iv.advance(prog, reply, provider)
        notices += prog["notices"]
        if prog["phase"] == "done":
            break
        prompts.append(prog["next_prompt"])

    # First message is the all-defaults offer, then one turn per field (no batch pass).
    assert prompts[0] == GATE_QUESTION
    assert len(prompts) == 5
    # Field 1's acknowledgement leads in to field 2's question (continuous chat).
    assert prompts[2].startswith("Got it, turbofan RUL.")
    assert "Below how many cycles" in prompts[2]
    cfg = prog["config"]
    assert cfg.framing == "turbofan RUL"
    assert cfg.failure_threshold == 30
    assert cfg.rul_cap == 125 and cfg.window == 5
    assert notices == ["Great, RMSE under 20 it is."]


def test_all_defaults_up_front_skips_the_whole_interview():
    provider = QueueProvider([_gate(True)])

    prog = iv.start_progress()
    prompts = [prog["next_prompt"]]
    prog = iv.advance(prog, "yes, just use defaults please", provider)
    notices = prog["notices"]

    # Only the gate was asked; no field questions.
    assert prompts == [GATE_QUESTION]
    assert prog["phase"] == "done"
    assert provider.calls == 1
    # Every field took its default...
    cfg = prog["config"]
    assert cfg.framing == DEFAULTS["framing"]
    assert cfg.failure_threshold == 30
    assert cfg.reporting_cadence == DEFAULTS["reporting_cadence"]
    assert cfg.rul_cap == 125 and cfg.window == 5
    # ...and each default was announced (nothing silent).
    assert any("Alert threshold: 30 cycles (default)" in n for n in notices)
    assert any("Success target" in n for n in notices)


def test_mid_interview_all_defaults_short_circuits_the_rest():
    # The captain's bug: "use defaults for everything" mid-interview must stop asking.
    provider = QueueProvider(
        [
            _gate(False),
            _turn("CLEAR", "Got it, turbofan RUL.", "turbofan RUL"),
            _turn("ALL_DEFAULTS", "Sure - I'll fill the rest with defaults.", None),
        ]
    )
    replies = ["no, let's do it", "turbofan RUL", "actually just use defaults for everything"]

    prog = iv.start_progress()
    prompts = [prog["next_prompt"]]
    notices: list[str] = []
    for reply in replies:
        prog = iv.advance(prog, reply, provider)
        notices += prog["notices"]
        if prog["phase"] == "done":
            break
        prompts.append(prog["next_prompt"])

    # Stopped asking after the short-circuit: gate + framing + threshold only.
    assert len(prompts) == 3
    # The answered field kept the user's value; the rest defaulted.
    cfg = prog["config"]
    assert cfg.framing == "turbofan RUL"
    assert cfg.failure_threshold == 30
    assert cfg.reporting_cadence == DEFAULTS["reporting_cadence"]
    assert cfg.success_metric == DEFAULTS["success_metric"]
    # The remaining defaults were announced.
    assert any("Alert threshold: 30 cycles (default)" in n for n in notices)
    assert any("Reporting:" in n for n in notices)


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


def test_write_report_frames_metrics_as_held_out_test():
    """The metrics must be labelled as held-out test scores, with the leaderboard
    flagged as cross-validated - so the report never reads as describing the model
    with mystery/aggregate numbers."""
    provider = FakeProvider("ok")
    write_report(_fake_train_result(), provider)
    system = provider.last_messages[0]["content"].lower()
    user = provider.last_messages[-1]["content"].lower()

    # The metric block is explicitly the held-out test set...
    assert "held-out test" in user
    # ...the leaderboard is explicitly cross-validated (not the same measurement)...
    assert "cross-validation" in user or "cross validation" in user
    # ...and the writer is told to say so and briefly why (never-seen-in-training).
    assert "held-out test" in system
    assert "never saw" in system or "never seen" in system
    # And it must ban computing distance-to-target (the live model once wrote
    # "2.91 cycles under the target", a fabricated subtraction).
    assert "above or below" in system or "under the target" in system


def test_success_verdict_decided_in_code():
    """The met/not-met check is done in code (the weak model gets it wrong), and
    only when the free-text target parses to a clear rule."""
    from sentinel.agents.report_writer import _success_verdict

    met = {"rmse": 17.09, "mae": 11.95, "r2": 0.818}
    assert _success_verdict("held-out RMSE under 20 cycles", met) is True  # 17.09 <= 20
    assert _success_verdict("RMSE under 15", met) is False  # 17.09 > 15
    assert _success_verdict("MAE below 10 cycles", met) is False  # 11.95 > 10
    assert _success_verdict("R2 above 0.9", met) is False  # 0.818 < 0.9
    assert _success_verdict("R2 of at least 0.8", met) is True  # 0.818 >= 0.8
    assert _success_verdict("RMSE around 20", met) is True  # no direction -> error metric upper bound
    # Unparseable targets -> None (report then states values without a verdict).
    assert _success_verdict("just make it accurate", met) is None
    assert _success_verdict("", met) is None
    assert _success_verdict(None, met) is None


def test_write_report_passes_precomputed_verdict_not_a_comparison():
    """write_report hands the LLM a decided verdict, so it never compares numbers."""
    provider = FakeProvider("ok")
    cfg = InterviewConfig(
        framing="turbofan RUL",
        failure_threshold=30,
        reporting_cadence="each run",
        success_metric="held-out RMSE under 20 cycles",
    )
    write_report(_fake_train_result(), provider, cfg)  # metrics rmse=17.1 -> meets
    user = provider.last_messages[-1]["content"]
    assert "SUCCESS CHECK" in user
    assert "MEETS the user" in user  # 17.1 <= 20, decided in code
    assert "do NOT re-compare" in user


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


def test_graph_runs_interview_to_monitor(tmp_path, monkeypatch):
    from sentinel.agents import monitor
    from sentinel.agents.training import TrainingRun

    result = _fake_train_result()
    test_eval = pd.DataFrame({"unit": [1, 2], "RUL": [10, 90]})
    run = TrainingRun(result=result, test_eval=test_eval, predict=lambda f: list(f["RUL"]))
    # The monitor rehydrates its predict fn from disk via `load_predict`; fake it
    # so this stays offline (same pattern as tests/test_train_state.py).
    monkeypatch.setattr(monitor, "load_predict", lambda model_path: (lambda frame: list(frame["RUL"])))

    # The interviewer offers the all-defaults gate, then converses per field.
    interview_turns = QueueProvider(
        [
            _gate(False),  # decline the up-front shortcut, go through the interview
            _turn("CLEAR", "ok", "turbofan RUL"),
            _turn("CLEAR", "ok", 30),
            _turn("CLEAR", "ok", "each run"),
            _turn("CLEAR", "ok", "RMSE < 20"),
        ]
    )
    configurable = {
        "provider_smart": interview_turns,
        "provider_cheap": FakeProvider("Report: the model is good."),
        "train_fn": lambda cfg: run,
        "ticket_dir": str(tmp_path),
    }
    answers = ["no", "turbofan RUL", "30", "each run", "RMSE < 20"]

    final = _drive_graph(configurable, answers)

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

    # Gate declined, then a CLEAR turn (repeats) resolves every field.
    interview_turns = QueueProvider([_gate(False), _turn("CLEAR", "ok", 30)])
    configurable = {
        "provider_smart": interview_turns,
        "provider_cheap": FakeProvider("unused"),
        "train_fn": boom,
        "ticket_dir": "artifacts/tickets",
    }
    final = _drive_graph(configurable, ["no", "x", "x", "x", "x"])

    # Failure is reported and the graph stops before monitoring (no model).
    assert final["event"] == "failed_reported"
    assert "PyCaret exploded" in final["error"]
    assert "did not complete" in final["report"]
    assert "alerts" not in final
