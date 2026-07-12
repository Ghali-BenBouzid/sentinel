"""Tests for the harness's corrective-feedback message builder."""
from __future__ import annotations

import groq
import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from sentinel.agents.harness import InvalidToolCallMiddleware, make_corrective_feedback
from tests.fakes import FakeChatModel


def _groq_tool_use_failed(missing: list[str]) -> groq.BadRequestError:
    """Build the exact exception shape Groq raises for a schema-invalid tool call."""
    fields = ", ".join(f"'{f}'" for f in missing)
    body = {
        "error": {
            "message": (
                "tool call validation failed: parameters for tool save_config "
                f"did not match schema: errors: [missing properties: {fields}]"
            ),
            "type": "invalid_request_error",
            "code": "tool_use_failed",
        }
    }
    response = httpx.Response(400, json=body, request=httpx.Request("POST", "https://api.groq.com"))
    return groq.BadRequestError(str(body), response=response, body=body)


def _registry(tmp_path, models):
    from sentinel.agents.registry import Registry

    registry = Registry(tmp_path / "models")
    for model_id in models:
        (registry.root / model_id).mkdir(parents=True)
    manifest = registry._read_manifest()
    manifest["models"] = models
    registry._write_manifest(manifest)
    return registry


def test_tier1_names_missing_fields_from_groq_error(tmp_path):
    registry = _registry(tmp_path, [])
    feedback = make_corrective_feedback(
        tools_chat_model=FakeChatModel(
            messages=iter([AIMessage("The tool call contains invalid JSON; correct it and retry.")])
        ),
        registry=registry,
    )
    error = _groq_tool_use_failed(
        ["framing", "failure_threshold", "reporting_cadence", "success_metric"]
    )
    message = feedback(error)
    assert "framing" in message
    assert "failure_threshold" in message
    assert "reporting_cadence" in message
    assert "success_metric" in message
    assert "ask the user" in message.lower()


def test_tier1_names_valid_ids_on_key_error(tmp_path):
    registry = _registry(tmp_path, ["et-v1", "et-v2"])
    feedback = make_corrective_feedback(
        tools_chat_model=FakeChatModel(messages=iter([AIMessage("unused")])),
        registry=registry,
    )
    message = feedback(KeyError("et-v99"))
    assert "et-v99" in message
    assert "et-v1" in message
    assert "et-v2" in message


def test_tier2_falls_back_to_cheap_model_for_unrecognized_errors(tmp_path):
    registry = _registry(tmp_path, [])
    feedback = make_corrective_feedback(
        tools_chat_model=FakeChatModel(
            messages=iter([AIMessage("The disk is full; try again shortly.")])
        ),
        registry=registry,
    )
    message = feedback(OSError("no space left on device"))
    assert message == "The disk is full; try again shortly."


def test_corrective_feedback_never_raises_when_cheap_model_is_unavailable(tmp_path):
    class UnavailableModel:
        def invoke(self, messages):
            raise ValueError(
                "Add a valid api_key (or auth_token) to the request headers"
            )

    feedback = make_corrective_feedback(
        tools_chat_model=UnavailableModel(),
        registry=_registry(tmp_path, []),
    )

    message = feedback(OSError("provider temporarily unavailable"))

    assert "temporarily unavailable" in message.lower()
    assert "api_key" not in message


def test_authentication_errors_are_formatted_without_secret_setup_instructions(tmp_path):
    feedback = make_corrective_feedback(
        tools_chat_model=FakeChatModel(messages=iter([AIMessage("unused")])),
        registry=_registry(tmp_path, []),
    )

    message = feedback(
        ValueError("Add a valid api_key (or auth_token) to the request headers")
    )

    assert "provider configuration" in message.lower()
    assert "api_key" not in message


class _InvalidToolCallChatModel(FakeChatModel):
    already_corrupted: bool = False

    def _generate(self, *args, **kwargs):
        result = super()._generate(*args, **kwargs)
        if self.already_corrupted:
            return result
        self.already_corrupted = True
        message = result.generations[0].message
        message.tool_calls = []
        message.invalid_tool_calls = [{
            "type": "invalid_tool_call",
            "id": "bad1",
            "name": "save_config",
            "args": "{not valid json",
            "error": "Invalid JSON",
        }]
        return result


def test_invalid_tool_call_gets_a_corrective_message_and_the_loop_continues(tmp_path):
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentState

    registry = _registry(tmp_path, [])
    feedback = make_corrective_feedback(
        tools_chat_model=FakeChatModel(
            messages=iter([AIMessage("The tool call contains invalid JSON; correct it and retry.")])
        ),
        registry=registry,
    )

    @tool
    def save_config(x: int) -> str:
        """A stand-in tool."""
        return "saved"

    model = _InvalidToolCallChatModel(messages=iter([
        AIMessage(content="calling save_config"),
        AIMessage(content="Understood, let me ask for the missing fields."),
    ]))
    agent = create_agent(
        model,
        [save_config],
        state_schema=AgentState,
        checkpointer=MemorySaver(),
        middleware=[InvalidToolCallMiddleware(feedback)],
    )
    final = agent.invoke(
        {"messages": [HumanMessage("go")]},
        {"configurable": {"thread_id": "invalid-tc"}},
    )
    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "bad1"
    assert "Invalid JSON" in tool_messages[0].content or "invalid" in tool_messages[0].content.lower()
    assert "Understood" in final["messages"][-1].content
