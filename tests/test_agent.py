"""End-to-end agent trajectory with a scripted model and fake trainer."""
from __future__ import annotations

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from tests.fakes import FakeChatModel


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
    assert state.interrupts[0].value["tool"] == "train"
    final = agent.invoke(
        Command(resume={state.interrupts[0].id: "yes"}), thread
    )
    assert "Trained" in final["messages"][-1].content
