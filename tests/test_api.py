"""End-to-end tests for the FastAPI/SSE surface over the V2 agent."""
from __future__ import annotations

import asyncio
import json

import httpx
import pandas as pd
from langchain_core.messages import AIMessage
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


def _sse_events(response):
    for line in response.text.splitlines():
        if line.startswith("data: "):
            yield json.loads(line[6:])


def _app(tmp_path):
    from sentinel.agents.agent import build_agent
    from sentinel.api.app import create_app

    def factory(checkpointer):
        return build_agent(
            chat_model=FakeChatModel(
                messages=iter(
                    [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "train",
                                    "args": {},
                                    "id": "c1",
                                }
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
            checkpointer=checkpointer,
        )

    return create_app(
        agent_factory=factory,
        checkpointer=MemorySaver(),
        models_dir=str(tmp_path / "models"),
    )


def _request(app, method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_autonomous_session_trains_end_to_end(tmp_path):
    app = _app(tmp_path)
    response = _request(
        app,
        "POST",
        "/sessions",
        json={"autonomy": "autonomous", "message": "train"},
    )
    assert response.status_code == 200
    assert response.headers["x-thread-id"]
    events = list(_sse_events(response))
    assert events[-1]["event"] == "done"
    assert any(
        event["event"] == "message"
        and "Trained" in event["data"].get("text", "")
        for event in events
    )


def test_guarded_session_emits_confirm_with_interrupt_id(tmp_path):
    app = _app(tmp_path)
    response = _request(
        app,
        "POST",
        "/sessions",
        json={"autonomy": "guarded", "message": "train"},
    )
    thread_id = response.headers["x-thread-id"]
    events = list(_sse_events(response))
    confirms = [
        event for event in events if event["event"] == "confirm"
    ]
    assert confirms and "interrupt" in confirms[-1]["data"]
    interrupt_id = confirms[-1]["data"]["interrupt"]
    resumed = _request(
        app,
        "POST",
        f"/sessions/{thread_id}/resume",
        json={"answers": {interrupt_id: "yes"}},
    )
    assert any(
        event["event"] == "message"
        and "Trained" in event["data"].get("text", "")
        for event in _sse_events(resumed)
    )


def test_unknown_thread_404(tmp_path):
    app = _app(tmp_path)
    assert _request(app, "GET", "/sessions/nope").status_code == 404
    assert (
        _request(
            app,
            "POST",
            "/sessions/nope/resume", json={"answer": "y"}
        ).status_code
        == 404
    )
    assert (
        _request(
            app,
            "POST",
            "/sessions/nope/message", json={"message": "x"}
        ).status_code
        == 404
    )


def test_leaderboard_empty_state(tmp_path):
    app = _app(tmp_path)
    response = _request(app, "GET", "/sessions/whatever/leaderboard")
    assert response.status_code == 200
    assert response.json() == {"active": None, "leaderboard": []}


def test_cors_header_present(tmp_path):
    app = _app(tmp_path)
    response = _request(
        app,
        "GET",
        "/sessions/x/leaderboard",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
