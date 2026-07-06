"""FastAPI/SSE surface over the V2 agent. SSE out, POST in.

New conversation turns append a HumanMessage. Confirmation replies resume
pending interrupt ids. Session autonomy is set once and checkpointed.
"""
from __future__ import annotations

import json
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from ..agents.agent import build_agent
from ..agents.training import run_retraining, run_training
from ..config import get_settings
from ..llm.provider import get_chat_model


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


def create_app(agent_factory=None, checkpointer=None) -> FastAPI:
    """Build the API around one process-lifetime compiled agent."""
    app = FastAPI(title="Sentinel V2")
    factory = agent_factory or _default_factory
    agent = factory(checkpointer)

    def _thread(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _require(thread: dict) -> None:
        if agent.get_state(thread).created_at is None:
            raise HTTPException(
                status_code=404, detail="unknown session"
            )

    async def _stream(graph_input, thread):
        try:
            for mode, chunk in agent.stream(
                graph_input,
                thread,
                stream_mode=["custom", "updates"],
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
                            if isinstance(message, AIMessage):
                                if getattr(message, "content", ""):
                                    yield _sse(
                                        "message",
                                        {"text": message.content},
                                    )
                                for call in message.tool_calls:
                                    yield _sse(
                                        "tool_call",
                                        {
                                            "name": call["name"],
                                            "args": call["args"],
                                        },
                                    )
                            elif isinstance(message, ToolMessage):
                                yield _sse(
                                    "tool_result",
                                    {
                                        "name": message.name,
                                        "text": message.content,
                                    },
                                )
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
        state = agent.get_state(thread)
        if state.interrupts:
            for item in state.interrupts:
                yield _sse(
                    "confirm",
                    {**item.value, "interrupt": item.id},
                )
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

    @app.post("/sessions/{thread_id}/message")
    async def message(thread_id: str, body: dict):
        thread = _thread(thread_id)
        _require(thread)
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
        _require(thread)
        answers = body.get("answers")
        if answers is None:
            state = agent.get_state(thread)
            if len(state.interrupts) != 1:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "multiple confirmations pending; "
                        "use 'answers' map"
                    ),
                )
            answers = {
                state.interrupts[0].id: body.get("answer", "")
            }
        return StreamingResponse(
            _stream(Command(resume=answers), thread),
            media_type="text/event-stream",
            headers={"x-thread-id": thread_id},
        )

    @app.get("/sessions/{thread_id}")
    async def snapshot(thread_id: str):
        thread = _thread(thread_id)
        _require(thread)
        state = agent.get_state(thread)
        messages = state.values.get("messages", [])
        return {
            "autonomy": state.values.get("autonomy"),
            "pending_confirmations": [
                {"interrupt": item.id, **item.value}
                for item in state.interrupts
            ],
            "last_message": (
                messages[-1].content if messages else None
            ),
        }

    return app
