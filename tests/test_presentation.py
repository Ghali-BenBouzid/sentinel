from langchain_core.messages import AIMessage, ToolMessage

from sentinel.api.presentation import stream_events, transcript_entry


def test_hidden_bookkeeping_is_consistent_live_and_in_snapshot():
    call = AIMessage(
        content="",
        tool_calls=[{
            "name": "rename_session",
            "args": {"title": "Models"},
            "id": "c1",
        }],
    )
    result = ToolMessage(
        content="Session renamed",
        tool_call_id="c1",
        name="rename_session",
    )

    assert transcript_entry(call) is None
    assert transcript_entry(result) is None
    assert stream_events(call) == []
    assert stream_events(result) == []


def test_visible_tool_is_consistent_live_and_in_snapshot():
    result = ToolMessage(
        content="Rank 3: lightgbm",
        tool_call_id="c1",
        name="leaderboard_candidate",
    )

    assert transcript_entry(result) == {
        "role": "tool_result",
        "content": "Rank 3: lightgbm",
        "name": "leaderboard_candidate",
    }
    assert stream_events(result) == [
        (
            "tool_result",
            {"name": "leaderboard_candidate", "text": "Rank 3: lightgbm"},
        )
    ]
