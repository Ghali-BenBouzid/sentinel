"""Tests for the harness's corrective-feedback message builder."""
from __future__ import annotations

import groq
import httpx
from langchain_core.messages import AIMessage

from sentinel.agents.harness import make_corrective_feedback
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
        tools_chat_model=FakeChatModel(messages=iter([AIMessage("unused")])),
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
