"""FastAPI surface over the resumable agent graph. SSE out, POST in.

`POST /sessions` starts a new thread and streams to the first interrupt.
`POST /sessions/{tid}/resume` feeds one answer and streams on to the next
interrupt (or to done). `GET /sessions/{tid}` reads a snapshot straight from
the checkpointer, so a client can reconnect after losing the SSE stream.

Dependencies (LLM providers, `train_fn`, `ticket_dir`) are injected the same
way the CLI and tests inject them - via `config["configurable"]` - so
`configurable_factory` is the one seam a caller (or a test) overrides.
"""

from __future__ import annotations

import json
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from ..agents.graph import build_graph
from ..agents.training import run_training
from ..llm.provider import get_provider


def _default_factory() -> dict:
    return {
        "provider_smart": get_provider("smart"),
        "provider_cheap": get_provider("cheap"),
        "train_fn": run_training,
        "ticket_dir": "artifacts/tickets",
    }


def _sse(event: str, data) -> str:
    return f"data: {json.dumps({'event': event, 'data': data})}\n\n"


def create_app(configurable_factory=None, checkpointer=None) -> FastAPI:
    app = FastAPI(title="Sentinel")
    factory = configurable_factory or _default_factory
    graph = build_graph(checkpointer=checkpointer)

    def _thread(tid: str) -> dict:
        return {"configurable": {**factory(), "thread_id": tid}}

    def _run(inp, thread):
        """Stream one graph leg, yielding SSE lines up to the next interrupt/END."""
        for mode, chunk in graph.stream(inp, thread, stream_mode=["custom", "updates"]):
            if mode == "custom":
                yield _sse(chunk.get("type", "notify"), chunk)
            elif mode == "updates":
                for _node, upd in chunk.items():
                    if isinstance(upd, dict) and upd.get("report"):
                        yield _sse("report", {"text": upd["report"]})
        state = graph.get_state(thread)
        if state.tasks and state.tasks[0].interrupts:
            yield _sse("prompt", {"text": state.tasks[0].interrupts[0].value})
        else:
            phase = (state.values.get("interview") or {}).get("phase", "done")
            yield _sse("done", {"phase": phase})

    @app.post("/sessions")
    def start():
        tid = uuid.uuid4().hex
        thread = _thread(tid)
        return StreamingResponse(
            _run({"event": "start"}, thread),
            media_type="text/event-stream",
            headers={"x-thread-id": tid},
        )

    @app.post("/sessions/{tid}/resume")
    def resume(tid: str, body: dict):
        thread = _thread(tid)
        return StreamingResponse(
            _run(Command(resume=body.get("answer", "")), thread),
            media_type="text/event-stream",
            headers={"x-thread-id": tid},
        )

    @app.get("/sessions/{tid}")
    def snapshot(tid: str):
        values = graph.get_state(_thread(tid)).values
        prog = values.get("interview") or {}
        return {
            "phase": prog.get("phase"),
            "next_prompt": prog.get("next_prompt"),
            "config": getattr(values.get("config"), "__dict__", None),
            "report": values.get("report"),
        }

    return app
