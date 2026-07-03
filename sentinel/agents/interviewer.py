"""Interviewer sub-agent - the only human-facing node.

Pattern: **the code owns the agenda, the LLM extracts structure.** A fixed
checklist (`QUESTIONS`) is asked one question at a time; the free-text answers
are then handed to the LLM once, which returns a JSON object we parse into an
`InterviewConfig`. The LLM never decides *what* to ask - it only phrases the
messy human answers into our schema.

Questions are asked through an injected `ask(prompt) -> str` callable (default
`input`), so the same node runs interactively for the captain, unattended with
scripted answers in the demo, and offline in tests.
"""

from __future__ import annotations

import json

from ..llm.provider import Provider
from . import domain_context
from .state import AgentState, InterviewConfig, append_log

# The agenda. The code owns this list; the LLM never adds to or reorders it.
QUESTIONS: list[tuple[str, str]] = [
    ("framing", "What are we trying to predict, and for what equipment?"),
    ("failure_threshold", "Below how many cycles of remaining life should we alert?"),
    ("reporting_cadence", "How often do you want status reports?"),
    ("success_metric", "What result would make this model a success to you?"),
]

# The same domain glossary the report writer uses grounds the interviewer too, so
# it normalizes a vague "success" answer into the right units/metric names
# (e.g. "within 20 cycles" -> RMSE/MAE in cycles) rather than guessing.
_EXTRACT_INSTRUCTIONS = (
    "You are configuring a predictive-maintenance training run from an "
    "interview. Use this domain glossary to interpret the answers in the correct "
    "terms and units - do NOT invent metrics or numbers the user did not give:\n\n"
    f"{domain_context.glossary()}\n\n"
    "Return ONLY a JSON object (no prose, no code fences) with keys: "
    '"framing" (string), "failure_threshold" (integer cycles), '
    '"reporting_cadence" (string), "success_metric" (string, phrased in the '
    "glossary's metric names/units when the user implies one), "
    '"rul_cap" (integer, default 125), "window" (integer, default 5). '
    "Fill rul_cap and window from the answers only if mentioned, else use the "
    "defaults. Here are the interview questions and the user's answers:\n\n"
)


def collect_config(answers: dict[str, str], provider: Provider) -> InterviewConfig:
    """Turn raw free-text answers into a structured `InterviewConfig` via the LLM.

    Defensive by design: if the model returns something unparseable, we fall
    back to safe defaults rather than crash the graph. Pure function - no I/O -
    so it is trivially testable with a fake provider.
    """
    qa = "\n".join(f"Q ({field}): {q}\nA: {answers.get(field, '')}" for field, q in QUESTIONS)
    raw = provider.complete([{"role": "user", "content": _EXTRACT_INSTRUCTIONS + qa}])
    data = _parse_json_object(raw)

    return InterviewConfig(
        framing=str(data.get("framing") or answers.get("framing", "")),
        failure_threshold=_as_int(data.get("failure_threshold"), default=30),
        reporting_cadence=str(data.get("reporting_cadence") or answers.get("reporting_cadence", "")),
        success_metric=str(data.get("success_metric") or answers.get("success_metric", "")),
        rul_cap=_as_int(data.get("rul_cap"), default=125),
        window=_as_int(data.get("window"), default=5),
    )


def _parse_json_object(text: str) -> dict:
    """Best-effort extraction of the first ``{...}`` JSON object from LLM text."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return {}


def _as_int(value, default: int) -> int:
    """Coerce a possibly-stringy/None LLM value to int, else `default`."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def interviewer_node(state: AgentState, config) -> dict:
    """Graph node: walk the checklist, extract config, emit `interview_done`."""
    cfg = config["configurable"]
    ask = cfg["ask"]
    provider = cfg["provider_smart"]

    answers = {field: ask(question) for field, question in QUESTIONS}
    interview_config = collect_config(answers, provider)
    return {
        "config": interview_config,
        "event": "interview_done",
        "log": append_log(state, f"interviewer: collected config {interview_config}"),
    }
