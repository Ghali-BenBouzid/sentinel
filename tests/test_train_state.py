"""Training state must survive the checkpointer.

LangGraph 1.2.7 serializes the ENTIRE state after every node with no pickle
fallback: a closure, a DataFrame, or a sklearn estimator all raise "Type is not
msgpack serializable". So only `TrainingRun.to_state()` (native-Python dicts and
lists) may cross the graph-state boundary; the heavy artifacts (model, test_eval
frame) are rehydrated where they're consumed. These tests pin both halves: the
serialization round-trip, and that state actually crosses the trainer -> report
-> monitor checkpoints intact.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command


def _fake_run():
    from sentinel.agents.training import TrainingRun
    from sentinel.core.automl import TrainResult

    lb = pd.DataFrame([{"Model": "Extra Trees", "RMSE": 17.1}, {"Model": "LightGBM", "RMSE": 18.4}])
    result = TrainResult(
        leaderboard=lb,
        best_model=object(),
        metrics={"rmse": 17.1, "mae": 12.0, "r2": 0.83},
        model_path=Path("artifacts/model"),
        metrics_path=Path("artifacts/metrics.json"),
    )
    test_eval = pd.DataFrame([{"unit": 1, "cycle": 200, "RUL": 40.0, "s2": 1.5}])
    return TrainingRun(result=result, test_eval=test_eval, predict=lambda f: [1.0])


def test_to_state_is_msgpack_serializable():
    state = _fake_run().to_state()
    blob = JsonPlusSerializer().dumps_typed(state)  # raises if any value is unserializable
    back = JsonPlusSerializer().loads_typed(blob)
    assert back["metrics"]["rmse"] == 17.1
    assert back["best_model_name"] == "Extra Trees"  # from leaderboard row 0 / type name
    assert isinstance(back["test_eval"], list) and back["test_eval"][0]["unit"] == 1
    # numpy scalars would break msgpack: assert everything is native
    for rec in back["test_eval"] + back["leaderboard"]:
        for v in rec.values():
            assert type(v).__module__ == "builtins"


def test_graph_carries_train_state_across_checkpoints(tmp_path, monkeypatch):
    """The real crash: the checkpointer serializes state after the trainer node.

    With a MemorySaver (which uses the same JsonPlusSerializer) and a serializable
    fake `train_fn`, the graph must reach `monitor_done` with no serialization
    error and produce alerts - proving train_state crosses trainer -> report ->
    monitor intact. `load_predict` is monkeypatched so the monitor runs offline.
    """
    from sentinel.agents import monitor
    from sentinel.agents.graph import build_graph
    from sentinel.agents.state import InterviewConfig

    class FakeProvider:
        def complete(self, messages, **kwargs) -> str:
            return "Report: model looks fine."

    monkeypatch.setattr(monitor, "load_predict", lambda model_path: (lambda f: [50.0] * len(f)))

    cfg = InterviewConfig(
        framing="turbofan RUL",
        failure_threshold=50,  # predicted 50 <= 50 -> alert
        reporting_cadence="each run",
        success_metric="RMSE under 20 cycles",
    )
    configurable = {
        "provider_cheap": FakeProvider(),
        "train_fn": lambda c: _fake_run(),
        "ticket_dir": str(tmp_path),
    }

    graph = build_graph(checkpointer=MemorySaver())
    final = graph.invoke(
        {"event": "interview_done", "config": dataclasses.asdict(cfg)},
        config={"configurable": {**configurable, "thread_id": "t1"}},
    )

    assert final["event"] == "monitor_done"
    assert final["report"] == "Report: model looks fine."
    assert [a["unit"] for a in final["alerts"]] == [1]
    assert (tmp_path / "ticket_unit_1.json").exists()


def test_to_state_failure_routes_to_run_failed_not_a_crash(tmp_path):
    """`to_state()` can raise too (e.g. a serialization problem on real data).

    Behind the FastAPI SSE stream, an unhandled exception here would truncate the
    response instead of yielding a clean error event. The trainer node must treat a
    `to_state()` failure exactly like a `train_fn` failure: no crash, `run_failed`,
    and a descriptive `error` string for the report_writer's existing error path.
    """
    from sentinel.agents.graph import build_graph
    from sentinel.agents.state import InterviewConfig

    class FakeProvider:
        def complete(self, messages, **kwargs) -> str:
            return "Report: model looks fine."

    class ExplodingRun:
        def to_state(self):
            raise ValueError("boom")

    cfg = InterviewConfig(
        framing="turbofan RUL",
        failure_threshold=50,
        reporting_cadence="each run",
        success_metric="RMSE under 20 cycles",
    )
    configurable = {
        "provider_cheap": FakeProvider(),
        "train_fn": lambda c: ExplodingRun(),
        "ticket_dir": str(tmp_path),
    }

    graph = build_graph(checkpointer=MemorySaver())
    final = graph.invoke(
        {"event": "interview_done", "config": dataclasses.asdict(cfg)},
        config={"configurable": {**configurable, "thread_id": "t2"}},
    )

    assert final["event"] == "failed_reported"
    assert "ValueError: boom" in final["error"]
    assert "boom" in final["report"]


def test_full_graph_under_strict_msgpack_no_dataclass_crosses(tmp_path, monkeypatch):
    """The `state["config"]` twin of the train_state rule, under STRICT serialization.

    A `JsonPlusSerializer(allowed_msgpack_modules=None)` is exactly what
    `LANGGRAPH_STRICT_MSGPACK=true` (or a future langgraph) installs: any dataclass
    written into state no longer rehydrates - it comes back as a bare kwargs dict -
    so any node reading `state["config"]` as a dataclass (`config.framing`,
    `config.failure_threshold`) crashes. Driving the WHOLE graph (all-defaults
    interview -> trainer -> report -> monitor) proves config crosses every
    checkpoint as native data. On the pre-fix code this raises in report_writer /
    monitor; after the fix it reaches `monitor_done` and `config` stays a dict.
    """
    from sentinel.agents import monitor
    from sentinel.agents.graph import build_graph

    class GateProvider:
        """provider_smart: the interviewer's gate classifier accepts all-defaults."""

        def complete(self, messages, **kwargs) -> str:
            return '{"all_defaults": true}'

    class ReportProvider:
        def complete(self, messages, **kwargs) -> str:
            return "Report: model looks fine."

    monkeypatch.setattr(monitor, "load_predict", lambda model_path: (lambda f: [10.0] * len(f)))

    configurable = {
        "provider_smart": GateProvider(),
        "provider_cheap": ReportProvider(),
        "train_fn": lambda c: _fake_run(),
        "ticket_dir": str(tmp_path),
    }
    thread = {"configurable": {**configurable, "thread_id": "strict1"}}

    strict = JsonPlusSerializer(allowed_msgpack_modules=None)  # == LANGGRAPH_STRICT_MSGPACK=true
    graph = build_graph(checkpointer=MemorySaver(serde=strict))

    graph.invoke({"event": "start"}, thread)  # runs to the gate interrupt
    final = graph.invoke(Command(resume="yes, just use sensible defaults"), thread)

    assert final["event"] == "monitor_done"
    assert type(final["config"]) is dict  # native crossed the boundary, not a dataclass
    # Nothing checkpointed is a dataclass INSTANCE (the native-only invariant).
    for value in graph.get_state(thread).values.values():
        assert not (dataclasses.is_dataclass(value) and not isinstance(value, type))
