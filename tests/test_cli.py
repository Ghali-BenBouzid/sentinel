"""The CLI turn helper drives the agent and detects confirmations."""
from __future__ import annotations

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from tests.fakes import FakeChatModel


def _fake_run(tmp_path):
    from sentinel.agents.training import TrainingRun
    from sentinel.core.automl import TrainResult

    model_path = tmp_path / "s.pkl"
    model_path.write_bytes(b"m")
    return TrainingRun(
        result=TrainResult(
            leaderboard=pd.DataFrame(
                [{"Model": "Extra Trees", "RMSE": 17.1}]
            ),
            best_model=object(),
            metrics={"rmse": 17.1, "mae": 12.0, "r2": 0.83},
            model_path=model_path,
            metrics_path=tmp_path / "m.json",
        ),
        test_eval=pd.DataFrame(
            [{"unit": 1, "cycle": 200, "RUL": 40.0, "s2": 1.5}]
        ),
        predict=lambda frame: [1.0],
    )


def test_run_turn_completes_without_interrupt_in_autonomous(tmp_path):
    from sentinel.agents.__main__ import run_turn
    from sentinel.agents.agent import build_agent

    agent = build_agent(
        chat_model=FakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "train", "args": {}, "id": "c1"}
                        ],
                    ),
                    AIMessage(content="Trained et-v1."),
                ]
            )
        ),
        train_fn=lambda cfg: _fake_run(tmp_path),
        retrain_fn=lambda *args: _fake_run(tmp_path),
        tools_chat_model=FakeChatModel(
            messages=iter([AIMessage("report")])
        ),
        ticket_dir=str(tmp_path / "tickets"),
        models_dir=str(tmp_path / "models"),
        checkpointer=MemorySaver(),
    )
    thread = {"configurable": {"thread_id": "cli1"}}
    output = []
    pending = run_turn(
        agent,
        thread,
        {
            "messages": [HumanMessage("train")],
            "autonomy": "autonomous",
        },
        output.append,
    )
    assert pending is False
    assert any("Trained et-v1" in str(line) for line in output)
