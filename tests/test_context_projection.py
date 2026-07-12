from langchain_core.messages import AIMessage, ToolMessage

from sentinel.agents.context import BoundedToolContextMiddleware


def _tool_exchange(index: int, payload: str):
    call_id = f"c{index}"
    return [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "inspect",
                "args": {"what": "leaderboard"},
                "id": call_id,
            }],
        ),
        ToolMessage(
            content=payload,
            tool_call_id=call_id,
            name="inspect",
        ),
    ]


def test_context_edit_clears_old_tool_results_but_keeps_recent_results():
    messages = []
    for index in range(5):
        messages.extend(_tool_exchange(index, f"payload-{index}" * 100))
    original_contents = [message.content for message in messages]
    projection = BoundedToolContextMiddleware(
        trigger_tokens=1,
        clear_at_least_tokens=100_000,
        keep_tool_results=2,
        placeholder="[Older tool result omitted]",
    )

    projected = projection._project(messages)

    tool_results = [
        message.content
        for message in projected
        if isinstance(message, ToolMessage)
    ]
    assert tool_results[:3] == ["[Older tool result omitted]"] * 3
    assert tool_results[-2][0:9] == "payload-3"
    assert tool_results[-1][0:9] == "payload-4"
    assert [message.content for message in messages] == original_contents
