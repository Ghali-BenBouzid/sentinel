"""Behavioral contracts for Sentinel's production LLM prompts."""
from __future__ import annotations

from langchain_core.messages import SystemMessage


def test_main_agent_keeps_internal_tool_syntax_out_of_user_guidance():
    from sentinel.agents.agent import SYSTEM_PROMPT

    prompt = SYSTEM_PROMPT.lower()
    assert "never tell the user to invoke a tool" in prompt
    assert "never show tool names" in prompt
    assert "'how to invoke' column" in prompt
    assert "call the appropriate tool yourself" in prompt
    assert "every option must be achievable with an available tool" in prompt
    assert "do not promise exports" in prompt
    assert "confidence intervals" in prompt
    assert "sentinel has no tools for them" in prompt
    assert "say 'evaluation settings' instead of rul_cap/window" in prompt
    assert "do not promise drift findings" in prompt
    assert "example of an acceptable next-steps table" in prompt
    assert "use product language" in prompt
    assert "keep each suggested option to one action" in prompt
    assert "prefer 'compare the active model" in prompt
    assert "prefer 'generate a performance report'" in prompt


def test_main_agent_prompt_separates_execution_from_communication():
    from sentinel.agents.agent import SYSTEM_PROMPT

    prompt = SYSTEM_PROMPT.lower()
    assert "<operating_contract>" in SYSTEM_PROMPT
    assert "<conversation_workflow>" in SYSTEM_PROMPT
    assert "<user_facing_communication>" in SYSTEM_PROMPT
    assert "check that it is grounded in tool results" in SYSTEM_PROMPT
    assert "1-based rank" in prompt
    assert "continue the user's original requested action" in prompt
    assert "do not stop after inspection" in prompt


def test_report_writer_prompt_preserves_grounding_and_presentation_boundary():
    from sentinel.agents.report_writer import _SYSTEM_PROMPT

    prompt = _SYSTEM_PROMPT.lower()
    assert "decision-ready reports" in prompt
    assert "do not add a menu of next actions" in prompt
    assert "every numeric claim" in prompt
    assert "appears verbatim in metrics" in prompt
    assert "no internal implementation instructions" in prompt


def test_corrective_feedback_uses_a_private_system_contract():
    from sentinel.agents.harness import _tier2_cheap_model

    class RecordingModel:
        messages = None

        def invoke(self, messages):
            self.messages = messages

            class Response:
                content = "Retry with valid arguments."

            return Response()

    model = RecordingModel()
    assert _tier2_cheap_model(ValueError("bad input"), model) == (
        "Retry with valid arguments."
    )
    assert isinstance(model.messages[0], SystemMessage)
    system = model.messages[0].content.lower()
    assert "private corrective feedback" in system
    assert "never invent missing values" in system
    assert "agent itself can take next" in system
    assert "exactly one concise sentence" in system
