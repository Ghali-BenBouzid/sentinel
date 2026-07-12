"""End-to-end agent trajectory with a scripted model and fake trainer."""
from __future__ import annotations

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from tests.fakes import FakeChatModel, RaisingThenFakeChatModel
from tests.test_harness import _groq_tool_use_failed


def _fake_training_run(tmp_path, rmse=17.1):
    from sentinel.agents.training import TrainingRun
    from sentinel.core.automl import TrainResult

    model_path = tmp_path / "src.pkl"
    model_path.write_bytes(b"m")
    result = TrainResult(
        leaderboard=pd.DataFrame([{"Model": "Extra Trees", "RMSE": rmse}]),
        best_model=object(),
        metrics={"rmse": rmse, "mae": 12.0, "r2": 0.83},
        model_path=model_path,
        metrics_path=tmp_path / "m.json",
    )
    test_eval = pd.DataFrame(
        [{"unit": 1, "cycle": 200, "RUL": 40.0, "s2": 1.5}]
    )
    return TrainingRun(
        result=result,
        test_eval=test_eval,
        predict=lambda frame: [1.0],
    )


def _tc(name, args, id):
    return {"name": name, "args": args, "id": id}


def _build(tmp_path, scripted):
    from sentinel.agents.agent import build_agent

    return build_agent(
        chat_model=FakeChatModel(messages=iter(scripted)),
        train_fn=lambda cfg: _fake_training_run(tmp_path),
        retrain_fn=lambda mid, hp, rc, window: _fake_training_run(
            tmp_path, rmse=16.0
        ),
        tools_chat_model=FakeChatModel(
            messages=iter([AIMessage("Report body.")])
        ),
        ticket_dir=str(tmp_path / "tickets"),
        models_dir=str(tmp_path / "models"),
        checkpointer=MemorySaver(),
        fallback_chat_model=FakeChatModel(messages=iter([AIMessage("fallback unused")])),
    )


def test_train_retrain_compare_promote_trajectory(tmp_path):
    from sentinel.agents.registry import Registry

    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                _tc("train", {"rul_cap": 125, "window": 5}, "c1")
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                _tc(
                    "retrain",
                    {
                        "model_id": "et",
                        "hyperparameters": {"n_estimators": 500},
                    },
                    "c2",
                )
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                _tc(
                    "compare",
                    {"model_id_a": "et-v1", "model_id_b": "et-v2"},
                    "c3",
                )
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                _tc("promote", {"model_id": "et-v2"}, "c4")
            ],
        ),
        AIMessage(content="Done: et-v2 is now active."),
    ]
    agent = _build(tmp_path, scripted)
    thread = {"configurable": {"thread_id": "t1"}}
    final = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "train, retrain et with 500 trees, compare, promote"
                )
            ],
            "autonomy": "autonomous",
        },
        thread,
    )
    assert "et-v2" in final["messages"][-1].content
    registry = Registry(tmp_path / "models")
    assert registry.active() == "et-v2"
    assert set(registry.list()) == {"et-v1", "et-v2"}


def test_autonomy_persists_across_turns(tmp_path):
    scripted = [
        AIMessage(content="", tool_calls=[_tc("train", {}, "c1")]),
        AIMessage(content="Trained."),
        AIMessage(
            content="",
            tool_calls=[_tc("promote", {"model_id": "et-v1"}, "c2")],
        ),
        AIMessage(content="Promoted."),
    ]
    agent = _build(tmp_path, scripted)
    thread = {"configurable": {"thread_id": "t2"}}
    agent.invoke(
        {
            "messages": [HumanMessage("train")],
            "autonomy": "autonomous",
        },
        thread,
    )
    final = agent.invoke(
        {"messages": [HumanMessage("promote et-v1")]}, thread
    )
    assert "Promoted" in final["messages"][-1].content


def test_agent_can_rename_session_in_checkpointed_state(tmp_path):
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                _tc(
                    "rename_session",
                    {"title": "Turbofan RUL baseline"},
                    "rename-1",
                )
            ],
        ),
        AIMessage(content="I understand the task."),
    ]
    agent = _build(tmp_path, scripted)
    thread = {"configurable": {"thread_id": "named-thread"}}

    final = agent.invoke(
        {
            "messages": [HumanMessage("Build an RUL baseline for FD001")],
            "autonomy": "guarded",
        },
        thread,
    )

    assert final["title"] == "Turbofan RUL baseline"
    assert agent.get_state(thread).values["title"] == "Turbofan RUL baseline"


def test_guarded_confirmation_interrupt_and_mapped_resume(tmp_path):
    scripted = [
        AIMessage(content="", tool_calls=[_tc("train", {}, "c1")]),
        AIMessage(content="Trained."),
    ]
    agent = _build(tmp_path, scripted)
    thread = {"configurable": {"thread_id": "t3"}}
    agent.invoke(
        {
            "messages": [HumanMessage("train")],
            "autonomy": "guarded",
        },
        thread,
    )
    state = agent.get_state(thread)
    assert state.interrupts
    request = state.interrupts[0].value
    assert request["action_requests"][0]["name"] == "train"
    final = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), thread)
    assert "Trained" in final["messages"][-1].content


def test_batched_confirmation_two_guarded_calls_in_one_message(tmp_path):
    from sentinel.agents.registry import Registry

    models_dir = tmp_path / "models"
    registry = Registry(models_dir)
    manifest = registry._read_manifest()
    manifest["models"] = ["et-v1", "et-v2"]
    manifest["active"] = "et-v1"
    registry._write_manifest(manifest)
    for model_id in ("et-v1", "et-v2"):
        (registry.root / model_id).mkdir()
        (registry.root / model_id / "metrics.json").write_text(
            '{"rmse": 17.1, "mae": 12.0, "r2": 0.83}'
        )
        (registry.root / model_id / "provenance.json").write_text("{}")
    scripted = [
        AIMessage(content="", tool_calls=[
            _tc("promote", {"model_id": "et-v1"}, "c1"),
            _tc("delete", {"model_id": "et-v2"}, "c2"),
        ]),
        AIMessage(content="Confirmed both actions."),
    ]
    agent = _build(tmp_path, scripted)
    thread = {"configurable": {"thread_id": "batched"}}
    agent.invoke({
        "messages": [HumanMessage("promote et-v1 and delete et-v2")],
        "autonomy": "guarded",
    }, thread)
    state = agent.get_state(thread)
    assert len(state.interrupts) == 1
    assert {a["name"] for a in state.interrupts[0].value["action_requests"]} == {
        "promote", "delete"
    }
    final = agent.invoke(Command(resume={"decisions": [
        {"type": "approve"}, {"type": "approve"}
    ]}), thread)
    assert "Confirmed both actions" in final["messages"][-1].content
    assert registry.active() == "et-v1"
    assert registry.list() == ["et-v1"]


def test_autonomous_mode_still_skips_confirmation(tmp_path):
    agent = _build(tmp_path, [
        AIMessage(content="", tool_calls=[_tc("train", {}, "c1")]),
        AIMessage(content="Trained."),
    ])
    thread = {"configurable": {"thread_id": "auto"}}
    final = agent.invoke({
        "messages": [HumanMessage("train")], "autonomy": "autonomous"
    }, thread)
    assert not agent.get_state(thread).interrupts
    assert "Trained" in final["messages"][-1].content


def test_model_call_limit_ends_run_instead_of_looping_forever(tmp_path):
    """A model stuck calling a tool forever stops at the configured limit."""
    import os

    from sentinel.config import get_settings

    os.environ["SENTINEL_MODEL_CALL_RUN_LIMIT"] = "3"
    get_settings.cache_clear()
    try:
        scripted = [
            AIMessage(content="", tool_calls=[_tc("inspect", {"what": "registry"}, f"c{i}")])
            for i in range(10)
        ]
        agent = _build(tmp_path, scripted)
        thread = {"configurable": {"thread_id": "limit-test"}}
        final = agent.invoke(
            {"messages": [HumanMessage("loop forever")], "autonomy": "autonomous"},
            thread,
        )
        model_messages = [m for m in final["messages"] if isinstance(m, AIMessage)]
        assert len(model_messages) <= 4
    finally:
        del os.environ["SENTINEL_MODEL_CALL_RUN_LIMIT"]
        get_settings.cache_clear()


def test_model_call_failure_ends_gracefully_instead_of_crashing(tmp_path):
    """A provider-rejected malformed tool call produces a graceful message."""
    from sentinel.agents.agent import build_agent

    failing_model = RaisingThenFakeChatModel(
        messages=iter([AIMessage("unreachable")]),
        fail_times=10,
    )
    failing_model.exception_factory = lambda: _groq_tool_use_failed(
        ["framing", "failure_threshold", "reporting_cadence", "success_metric"]
    )
    agent = build_agent(
        chat_model=failing_model,
        train_fn=lambda cfg: (_ for _ in ()).throw(AssertionError("not reached")),
        retrain_fn=lambda *a: (_ for _ in ()).throw(AssertionError("not reached")),
        tools_chat_model=FakeChatModel(messages=iter([AIMessage("unused")])),
        ticket_dir=str(tmp_path / "tickets"),
        models_dir=str(tmp_path / "models"),
        checkpointer=MemorySaver(),
        fallback_chat_model=failing_model,
    )
    final = agent.invoke(
        {"messages": [HumanMessage("use sensible defaults")], "autonomy": "guarded"},
        {"configurable": {"thread_id": "incident"}},
    )
    last = final["messages"][-1]
    assert "framing" in last.content
    assert "failure_threshold" in last.content
    assert "BadRequestError" not in last.content
    assert "Traceback" not in last.content


def test_third_ranked_model_can_be_inspected_then_retrained(tmp_path):
    from sentinel.agents.agent import build_agent
    from sentinel.agents.registry import Registry

    models_dir = tmp_path / "models"
    registry = Registry(models_dir)
    source = tmp_path / "active.pkl"
    source.write_bytes(b"model")
    registry.register(
        family="et",
        model_path=source,
        metrics={"rmse": 15.2, "mae": 10.8, "r2": 0.86},
        leaderboard=[
            {"id": "et", "Model": "Extra Trees", "RMSE": 15.2},
            {"id": "rf", "Model": "Random Forest", "RMSE": 16.0},
            {"id": "lightgbm", "Model": "LightGBM", "RMSE": 16.4},
        ],
        provenance={
            "source": "train",
            "model_id": "et",
            "hyperparameters": {},
            "config": {"rul_cap": 125, "window": 5},
            "parent": None,
        },
        test_eval=[{"RUL": 40.0}],
    )
    retrained = []
    scripted = [
        AIMessage(
            content="",
            tool_calls=[_tc("inspect", {"what": "leaderboard"}, "inspect-1")],
        ),
        AIMessage(
            content="",
            tool_calls=[
                _tc(
                    "retrain",
                    {"model_id": "lightgbm", "hyperparameters": {}},
                    "retrain-1",
                )
            ],
        ),
        AIMessage(content="Retrained the third-ranked candidate."),
    ]
    agent = build_agent(
        chat_model=FakeChatModel(messages=iter(scripted)),
        train_fn=lambda cfg: _fake_training_run(tmp_path),
        retrain_fn=lambda model_id, *args: (
            retrained.append(model_id) or _fake_training_run(tmp_path, rmse=16.0)
        ),
        tools_chat_model=FakeChatModel(messages=iter([AIMessage("unused")])),
        ticket_dir=str(tmp_path / "tickets"),
        models_dir=str(models_dir),
        checkpointer=MemorySaver(),
        fallback_chat_model=FakeChatModel(messages=iter([AIMessage("unused")])),
    )

    final = agent.invoke(
        {
            "messages": [HumanMessage("Retrain the third-best model")],
            "autonomy": "autonomous",
        },
        {"configurable": {"thread_id": "third-ranked"}},
    )

    assert retrained == ["lightgbm"]
    assert "third-ranked" in final["messages"][-1].content
