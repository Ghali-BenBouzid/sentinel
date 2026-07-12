"""FastAPI/SSE surface over the V2 agent. SSE out, POST in.

New conversation turns append a HumanMessage. Confirmation replies resume
pending interrupt ids. Session autonomy is set once and checkpointed.
"""
from __future__ import annotations

import asyncio
import json
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from ..agents.agent import build_agent
from ..agents.registry import Registry
from ..agents.training import run_retraining, run_training
from ..config import configure_langsmith, get_settings
from ..llm.provider import get_chat_model
from .presentation import stream_events, transcript_entry

_QUEUE_DONE = object()


async def _iter_in_thread(sync_iterable):
    """Drain a blocking iterator on a worker thread, yielding items as they arrive.

    The compiled graph's checkpointer (SqliteSaver) and chat model calls are
    synchronous; running them directly inside an async route would occupy the
    single event-loop thread for the whole turn and stall every other request
    on the server. This keeps that blocking work off the loop while still
    streaming each chunk to the client as soon as it's produced.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            for item in sync_iterable:
                loop.call_soon_threadsafe(queue.put_nowait, (True, item))
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, (False, exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _QUEUE_DONE)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = await queue.get()
        if item is _QUEUE_DONE:
            return
        ok, payload = item
        if not ok:
            raise payload
        yield payload


def _default_factory(checkpointer):
    return build_agent(
        chat_model=get_chat_model("smart"),
        train_fn=run_training,
        retrain_fn=run_retraining,
        tools_chat_model=get_chat_model("cheap"),
        ticket_dir="artifacts/tickets",
        models_dir="artifacts/models",
        checkpointer=checkpointer,
    )


def _sse(event: str, data) -> str:
    return (
        f"data: {json.dumps({'event': event, 'data': data})}\n\n"
    )


def _transcript(messages) -> list[dict]:
    return [
        entry
        for message in messages
        if (entry := transcript_entry(message)) is not None
    ]


def _pending_cards(state) -> list[dict]:
    if not state.interrupts:
        return []
    request = state.interrupts[0].value
    interrupt_id = state.interrupts[0].id
    return [
        {
            "interrupt": f"{interrupt_id}:{index}",
            "tool": action["name"],
            "detail": json.dumps(action["args"]),
        }
        for index, action in enumerate(request["action_requests"])
    ]


def create_app(
    agent_factory=None, checkpointer=None, models_dir="artifacts/models"
) -> FastAPI:
    """Build the API around one process-lifetime compiled agent."""
    configure_langsmith()
    app = FastAPI(title="Sentinel V2")
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(?:localhost|127\.0\.0\.1):\d+",
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-thread-id"],
    )
    registry = Registry(models_dir)
    factory = agent_factory or _default_factory
    agent = factory(checkpointer)

    thread_locks: dict[str, asyncio.Lock] = {}

    def _thread(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _lock_for(thread_id: str) -> asyncio.Lock:
        # Two mutating requests against the same thread_id now run
        # concurrently (that's the point - see _iter_in_thread above), but
        # LangGraph checkpoints for one thread aren't safe for concurrent
        # writers: each request reads the latest checkpoint as its parent,
        # so two in-flight writes silently lose one of them (e.g. an
        # autonomy toggle clicked mid-turn gets clobbered when the turn's
        # own checkpoint commits after it). Serialize same-thread writes;
        # different threads still run fully in parallel.
        return thread_locks.setdefault(thread_id, asyncio.Lock())

    async def _require(thread: dict) -> None:
        state = await asyncio.to_thread(agent.get_state, thread)
        if state.created_at is None:
            raise HTTPException(
                status_code=404, detail="unknown session"
            )

    async def _stream(graph_input, thread):
        lock = _lock_for(thread["configurable"]["thread_id"])
        async with lock:
            try:
                async for mode, chunk in _iter_in_thread(
                    agent.stream(
                        graph_input,
                        thread,
                        stream_mode=["custom", "updates"],
                    )
                ):
                    if mode == "custom":
                        yield _sse(chunk.get("type", "notify"), chunk)
                    elif mode == "updates":
                        for update in (chunk or {}).values():
                            messages = (
                                update.get("messages")
                                if isinstance(update, dict)
                                else None
                            )
                            for message in messages or []:
                                for event, data in stream_events(message):
                                    yield _sse(event, data)
            except Exception as error:  # noqa: BLE001
                yield _sse(
                    "error",
                    {
                        "message": (
                            f"{type(error).__name__}: {error}"
                        )
                    },
                )
                return
            state = await asyncio.to_thread(agent.get_state, thread)
            if state.interrupts:
                for card in _pending_cards(state):
                    yield _sse("confirm", card)
            else:
                yield _sse("done", {})

    @app.post("/sessions")
    async def start(body: dict | None = None):
        body = body or {}
        thread_id = uuid.uuid4().hex
        thread = _thread(thread_id)
        autonomy = (
            body.get("autonomy")
            or get_settings().sentinel_autonomy
        )
        graph_input = {
            "messages": [
                HumanMessage(body.get("message", "Hello"))
            ],
            "autonomy": autonomy,
        }
        return StreamingResponse(
            _stream(graph_input, thread),
            media_type="text/event-stream",
            headers={"x-thread-id": thread_id},
        )

    def _list_sessions_sync() -> list[dict]:
        seen: dict[str, dict] = {}
        # Materialize fully before the loop: SqliteSaver.list() holds its
        # cursor lock open for the whole generator, and get_state() below
        # needs that same (non-reentrant) lock - interleaving the two
        # deadlocks a thread against itself.
        for item in list(agent.checkpointer.list(None)):
            tid = item.config["configurable"]["thread_id"]
            if tid in seen:
                continue
            state = agent.get_state(_thread(tid))
            messages = state.values.get("messages", [])
            seen[tid] = {
                "thread_id": tid,
                "title": state.values.get("title"),
                "autonomy": state.values.get("autonomy"),
                "last_message": (
                    messages[-1].content if messages else None
                ),
                "_ts": state.created_at or "",
            }
        ordered = sorted(seen.values(), key=lambda s: s["_ts"], reverse=True)
        return [
            {k: v for k, v in s.items() if k != "_ts"}
            for s in ordered
        ]

    @app.get("/sessions")
    async def list_sessions():
        return {"sessions": await asyncio.to_thread(_list_sessions_sync)}

    @app.post("/sessions/{thread_id}/message")
    async def message(thread_id: str, body: dict):
        thread = _thread(thread_id)
        await _require(thread)
        graph_input = {
            "messages": [HumanMessage(body["message"])]
        }
        return StreamingResponse(
            _stream(graph_input, thread),
            media_type="text/event-stream",
            headers={"x-thread-id": thread_id},
        )

    @app.post("/sessions/{thread_id}/resume")
    async def resume(thread_id: str, body: dict):
        thread = _thread(thread_id)
        await _require(thread)
        state = await asyncio.to_thread(agent.get_state, thread)
        if not state.interrupts:
            raise HTTPException(status_code=400, detail="no confirmation is pending")
        request = state.interrupts[0].value
        interrupt_id = state.interrupts[0].id
        expected = len(request["action_requests"])
        answers = body.get("answers")
        if answers is None:
            if expected != 1:
                raise HTTPException(
                    status_code=400,
                    detail="multiple confirmations pending; use 'answers' map",
                )
            answers = {f"{interrupt_id}:0": body.get("answer", "")}
        by_index: dict[int, str] = {}
        for composite_id, answer in answers.items():
            base, _, index = composite_id.rpartition(":")
            if base != interrupt_id or not index.isdigit():
                raise HTTPException(
                    status_code=400,
                    detail=f"unknown confirmation id {composite_id!r}",
                )
            by_index[int(index)] = answer
        if sorted(by_index) != list(range(expected)):
            raise HTTPException(
                status_code=400,
                detail=f"expected answers for all {expected} pending confirmation(s), got {sorted(by_index)}",
            )
        decisions = [
            {"type": "approve"}
            if by_index[i].strip().lower() in {"y", "yes"}
            else {"type": "reject"}
            for i in range(expected)
        ]
        return StreamingResponse(
            _stream(Command(resume={"decisions": decisions}), thread),
            media_type="text/event-stream",
            headers={"x-thread-id": thread_id},
        )

    @app.post("/sessions/{thread_id}/autonomy")
    async def set_autonomy(thread_id: str, body: dict):
        thread = _thread(thread_id)
        await _require(thread)
        value = body.get("autonomy")
        if value not in ("guarded", "autonomous"):
            raise HTTPException(
                status_code=400,
                detail="autonomy must be 'guarded' or 'autonomous'",
            )
        async with _lock_for(thread_id):
            await asyncio.to_thread(agent.update_state, thread, {"autonomy": value})
        return {"autonomy": value}

    @app.get("/sessions/{thread_id}")
    async def snapshot(thread_id: str):
        thread = _thread(thread_id)
        await _require(thread)
        state = await asyncio.to_thread(agent.get_state, thread)
        messages = state.values.get("messages", [])
        return {
            "title": state.values.get("title"),
            "autonomy": state.values.get("autonomy"),
            "pending_confirmations": _pending_cards(state),
            "last_message": (
                messages[-1].content if messages else None
            ),
            "messages": _transcript(messages),
        }

    @app.get("/sessions/{thread_id}/leaderboard")
    async def leaderboard(thread_id: str):
        active = registry.active()
        if active is None:
            return {"active": None, "leaderboard": []}
        rows = registry.get(active)["metrics"].get("leaderboard", [])
        return {"active": active, "leaderboard": rows}

    return app
