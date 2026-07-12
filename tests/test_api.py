"""End-to-end tests for the FastAPI/SSE surface over the V2 agent."""
from __future__ import annotations

import asyncio
import json

import httpx
import pandas as pd
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from tests.fakes import FakeChatModel, SlowFakeChatModel


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


def _app(tmp_path, checkpointer=None):
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
            fallback_chat_model=FakeChatModel(messages=iter([AIMessage("fallback unused")])),
        )

    return create_app(
        agent_factory=factory,
        checkpointer=checkpointer or MemorySaver(),
        models_dir=str(tmp_path / "models"),
    )


def _slow_app(tmp_path, delay_seconds=1.0):
    """An app whose model call blocks synchronously, like a real HTTP round trip."""
    from sentinel.agents.agent import build_agent
    from sentinel.api.app import create_app

    def factory(checkpointer):
        return build_agent(
            chat_model=SlowFakeChatModel(
                messages=iter([AIMessage("hi"), AIMessage("hi")]),
                delay_seconds=delay_seconds,
            ),
            train_fn=lambda cfg: _fake_run(tmp_path),
            retrain_fn=lambda *args: _fake_run(tmp_path),
            tools_chat_model=FakeChatModel(messages=iter([AIMessage("report")])),
            ticket_dir=str(tmp_path / "tickets"),
            models_dir=str(tmp_path / "models"),
            checkpointer=checkpointer,
            fallback_chat_model=FakeChatModel(messages=iter([AIMessage("fallback unused")])),
        )

    return create_app(
        agent_factory=factory,
        checkpointer=MemorySaver(),
        models_dir=str(tmp_path / "models"),
    )


def _renaming_app(tmp_path):
    from sentinel.agents.agent import build_agent
    from sentinel.api.app import create_app

    def factory(checkpointer):
        return build_agent(
            chat_model=FakeChatModel(messages=iter([
                AIMessage(content="", tool_calls=[{
                    "name": "rename_session",
                    "args": {"title": "FD001 health baseline"},
                    "id": "rename-1",
                }]),
                AIMessage(content="Ready."),
            ])),
            train_fn=lambda cfg: _fake_run(tmp_path),
            retrain_fn=lambda *args: _fake_run(tmp_path),
            tools_chat_model=FakeChatModel(messages=iter([AIMessage("report")])),
            ticket_dir=str(tmp_path / "tickets"),
            models_dir=str(tmp_path / "models"),
            checkpointer=checkpointer,
            fallback_chat_model=FakeChatModel(messages=iter([AIMessage("unused")])),
        )

    return create_app(
        agent_factory=factory,
        checkpointer=MemorySaver(),
        models_dir=str(tmp_path / "models"),
    )


def _seed_two_models(tmp_path):
    from sentinel.agents.registry import Registry

    registry = Registry(tmp_path / "models")
    manifest = registry._read_manifest()
    manifest.update({"models": ["et-v1", "et-v2"], "active": "et-v1"})
    registry._write_manifest(manifest)
    for model_id in ("et-v1", "et-v2"):
        (registry.root / model_id).mkdir(exist_ok=True)


def _batched_guarded_app(tmp_path):
    from sentinel.agents.agent import build_agent
    from sentinel.api.app import create_app

    def factory(checkpointer):
        return build_agent(
            chat_model=FakeChatModel(messages=iter([
                AIMessage(content="", tool_calls=[
                    {"name": "promote", "args": {"model_id": "et-v1"}, "id": "c1"},
                    {"name": "delete", "args": {"model_id": "et-v2"}, "id": "c2"},
                ]),
                AIMessage(content="Confirmed both actions."),
            ])),
            train_fn=lambda cfg: _fake_run(tmp_path),
            retrain_fn=lambda *args: _fake_run(tmp_path),
            tools_chat_model=FakeChatModel(messages=iter([AIMessage("report")])),
            ticket_dir=str(tmp_path / "tickets"),
            models_dir=str(tmp_path / "models"),
            checkpointer=checkpointer,
            fallback_chat_model=FakeChatModel(messages=iter([AIMessage("fallback unused")])),
        )

    return create_app(
        agent_factory=factory,
        checkpointer=MemorySaver(),
        models_dir=str(tmp_path / "models"),
    )


def test_slow_turns_are_not_serialized(tmp_path):
    """Two slow in-flight chat turns must run concurrently, not queue behind each other.

    Regression test: the SSE route used to run the graph's blocking sync
    .stream() call directly inside an async endpoint, which occupies the
    single event-loop thread for the whole turn - a second unrelated
    request (a different session, the autonomy toggle, a snapshot fetch)
    could not even be scheduled until the first turn fully finished. Two
    turns that each take `delay_seconds` should together take close to
    `delay_seconds`, not `2 * delay_seconds`.
    """
    delay_seconds = 0.6
    app = _slow_app(tmp_path, delay_seconds=delay_seconds)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            start = asyncio.get_event_loop().time()
            responses = await asyncio.gather(
                client.post("/sessions", json={"message": "a"}),
                client.post("/sessions", json={"message": "b"}),
            )
            return responses, asyncio.get_event_loop().time() - start

    responses, elapsed = asyncio.run(run())
    assert all(r.status_code == 200 for r in responses)
    assert elapsed < 1.5 * delay_seconds, (
        f"two {delay_seconds}s turns took {elapsed:.2f}s combined - they were "
        "serialized behind one blocking event loop instead of running concurrently"
    )


def _request(app, method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def _start_session(app, message="train", autonomy="guarded") -> str:
    response = _request(
        app, "POST", "/sessions",
        json={"message": message, "autonomy": autonomy},
    )
    list(_sse_events(response))
    return response.headers["x-thread-id"]


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


def test_autonomy_toggle_rejects_bad_value(tmp_path):
    app = _app(tmp_path)
    thread_id = _start_session(app)
    response = _request(
        app, "POST", f"/sessions/{thread_id}/autonomy", json={"autonomy": "banana"}
    )
    assert response.status_code == 400


def test_autonomy_toggle_sets_value(tmp_path):
    app = _app(tmp_path)
    thread_id = _start_session(app, autonomy="guarded")
    response = _request(
        app, "POST", f"/sessions/{thread_id}/autonomy", json={"autonomy": "autonomous"}
    )
    assert response.status_code == 200
    assert response.json() == {"autonomy": "autonomous"}
    snap = _request(app, "GET", f"/sessions/{thread_id}").json()
    assert snap["autonomy"] == "autonomous"


def test_list_sessions(tmp_path):
    app = _app(tmp_path)
    a = _start_session(app)
    b = _start_session(app)
    response = _request(app, "GET", "/sessions")
    assert response.status_code == 200
    sessions = response.json()["sessions"]
    ids = {s["thread_id"] for s in sessions}
    assert {a, b} <= ids
    for s in sessions:
        assert set(s) == {"thread_id", "autonomy", "last_message", "title"}


def test_list_sessions_uses_agent_assigned_title(tmp_path):
    app = _renaming_app(tmp_path)
    thread_id = _start_session(app)

    sessions = _request(app, "GET", "/sessions").json()["sessions"]
    session = next(item for item in sessions if item["thread_id"] == thread_id)

    assert session["title"] == "FD001 health baseline"
    snapshot = _request(app, "GET", f"/sessions/{thread_id}").json()
    assert all(
        message.get("name") != "rename_session"
        for message in snapshot["messages"]
    )


def test_list_sessions_with_sqlite_checkpointer_does_not_deadlock(tmp_path):
    """Regression: SqliteSaver.list() holds its cursor lock open for the whole
    generator; calling get_state() (which needs that same non-reentrant lock)
    while still iterating it deadlocks the thread against itself. MemorySaver
    (used by the other tests) doesn't have this lock, so it never caught this.
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    checkpointer = SqliteSaver(
        sqlite3.connect(str(tmp_path / "checkpoints.sqlite"), check_same_thread=False)
    )
    app = _app(tmp_path, checkpointer=checkpointer)
    thread_id = _start_session(app)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await asyncio.wait_for(client.get("/sessions"), timeout=3)

    response = asyncio.run(run())
    assert response.status_code == 200
    ids = {s["thread_id"] for s in response.json()["sessions"]}
    assert thread_id in ids


def test_snapshot_returns_transcript(tmp_path):
    app = _app(tmp_path)
    thread_id = _start_session(app, message="train")
    snap = _request(app, "GET", f"/sessions/{thread_id}").json()
    assert "messages" in snap
    assert snap["messages"][0]["role"] == "user"
    assert snap["messages"][0]["content"] == "train"
    for m in snap["messages"]:
        assert m["role"] in ("user", "agent", "tool_result")


def test_resume_rejects_incomplete_batched_answers(tmp_path):
    _seed_two_models(tmp_path)
    app = _batched_guarded_app(tmp_path)
    thread_id = _start_session(
        app, message="promote et-v1 and delete et-v2", autonomy="guarded"
    )
    snap = _request(app, "GET", f"/sessions/{thread_id}").json()
    assert len(snap["pending_confirmations"]) == 2
    first_id = snap["pending_confirmations"][0]["interrupt"]
    response = _request(
        app,
        "POST",
        f"/sessions/{thread_id}/resume",
        json={"answers": {first_id: "yes"}},
    )
    assert response.status_code == 400


def test_snapshot_expands_bundled_confirmations_independently(tmp_path):
    _seed_two_models(tmp_path)
    app = _batched_guarded_app(tmp_path)
    thread_id = _start_session(
        app, message="promote et-v1 and delete et-v2", autonomy="guarded"
    )
    cards = _request(app, "GET", f"/sessions/{thread_id}").json()[
        "pending_confirmations"
    ]
    assert len(cards) == 2
    assert {c["tool"] for c in cards} == {"promote", "delete"}
    for card in cards:
        assert set(card) == {"interrupt", "tool", "detail"}
