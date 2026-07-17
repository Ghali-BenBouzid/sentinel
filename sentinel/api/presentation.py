"""One presentation seam for checkpoint snapshots and live SSE updates."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def transcript_entry(message) -> dict | None:
    """Project one internal message into a stable product transcript entry."""
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AIMessage):
        visible_calls = [
            call
            for call in message.tool_calls
            if call["name"] != "rename_session"
        ]
        entry = {"role": "agent", "content": message.content}
        if visible_calls:
            entry["tool_calls"] = [
                {"name": call["name"], "args": call["args"]}
                for call in visible_calls
            ]
        return entry if message.content or visible_calls else None
    if isinstance(message, ToolMessage):
        if message.name == "rename_session":
            return None
        return {
            "role": "tool_result",
            "content": message.content,
            "name": message.name,
        }
    return None


def stream_events(message) -> list[tuple[str, dict]]:
    """Project one internal message into zero or more live product events."""
    if isinstance(message, AIMessage):
        events = []
        if message.content:
            events.append(("message", {"text": message.content}))
        events.extend(
            ("tool_call", {"name": call["name"], "args": call["args"]})
            for call in message.tool_calls
            if call["name"] != "rename_session"
        )
        return events
    if isinstance(message, ToolMessage) and message.name != "rename_session":
        return [
            (
                "tool_result",
                {"name": message.name, "text": message.content},
            )
        ]
    return []
