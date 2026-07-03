"""Interviewer sub-agent - the only human-facing node.

Pattern: **the code owns the agenda, the LLM extracts structure.** A fixed
checklist (`QUESTIONS`) is asked one question at a time; the free-text answers go
to the LLM, which both extracts a structured value *and flags whether each field
was really answered*. A weak or evasive answer ("I don't know", "you decide",
"NEVER", empty, off-topic) is not stored as junk: the field is re-asked ONCE with
guidance and a concrete default, and if it is still not answered the explained
default is used and **surfaced** to the user - nothing is defaulted silently.

Questions are asked through an injected `ask(prompt) -> str` callable (default
`input`) and defaults are announced through an injected `notify(str)` callable
(default `print`), so the same node runs interactively for the captain,
unattended with scripted answers in the demo, and offline in tests.
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
_ASKED_FIELDS = [field for field, _ in QUESTIONS]
# Advanced knobs never put to the user; extracted only if the user volunteers them.
_EXTRA_FIELDS = ["rul_cap", "window"]

# Sensible defaults used (and surfaced) when a field isn't answered.
DEFAULTS: dict[str, object] = {
    "framing": "NASA C-MAPSS turbofan Remaining Useful Life (RUL) prediction",
    "failure_threshold": 30,
    "reporting_cadence": "after every training run",
    "success_metric": "held-out RMSE under 20 cycles",
    "rul_cap": 125,
    "window": 5,
}

# Shown in the ONE re-ask, so the user gets a concrete example + the default.
_GUIDANCE: dict[str, str] = {
    "framing": "e.g. 'remaining useful life of turbofan engines, to plan maintenance'",
    "failure_threshold": "a number of cycles, e.g. 30",
    "reporting_cadence": "e.g. 'after every training run' or 'daily'",
    "success_metric": "e.g. 'held-out RMSE under 20 cycles'",
}

# The short line surfaced when a default is actually applied.
_DEFAULT_NOTES: dict[str, str] = {
    "framing": f"No clear problem framing given - assuming {DEFAULTS['framing']}.",
    "failure_threshold": f"No alert threshold given - using a default of {DEFAULTS['failure_threshold']} cycles.",
    "reporting_cadence": f"No reporting cadence given - defaulting to '{DEFAULTS['reporting_cadence']}'.",
    "success_metric": f"No success definition given - assuming '{DEFAULTS['success_metric']}'.",
    "rul_cap": f"Using the standard RUL cap of {DEFAULTS['rul_cap']} cycles.",
    "window": f"Using the default rolling-window of {DEFAULTS['window']} cycles.",
}

# The same domain glossary the report writer uses grounds the interviewer too, so
# it normalizes a vague "success" answer into the right units/metric names and can
# judge whether a reply actually answers the question.
_EXTRACT_INSTRUCTIONS = (
    "You are configuring a predictive-maintenance training run from an interview. "
    "Use this domain glossary to interpret answers in the correct terms and units - "
    "do NOT invent metrics or numbers the user did not give:\n\n"
    f"{domain_context.glossary()}\n\n"
    "For each field, decide whether the user REALLY answered it. Set answered=false "
    "when the reply is empty, a non-answer or deferral (\"I don't know\", \"you "
    'decide\", "you tell me", "whatever", or "NEVER" given as a threshold), off-topic, '
    "or does not actually provide that field. Set answered=true only for a genuine, "
    "on-topic answer.\n\n"
    "Return ONLY a JSON object (no prose, no code fences) of exactly this shape:\n"
    '{\n'
    '  "framing": {"answered": true/false, "value": string or null},\n'
    '  "failure_threshold": {"answered": true/false, "value": integer cycles or null},\n'
    '  "reporting_cadence": {"answered": true/false, "value": string or null},\n'
    '  "success_metric": {"answered": true/false, "value": string (phrased in the '
    "glossary's metric names/units when the user implies one) or null},\n"
    '  "rul_cap": {"answered": true/false, "value": integer or null},\n'
    '  "window": {"answered": true/false, "value": integer or null}\n'
    "}\n"
    "For rul_cap and window, set answered=true ONLY if the user explicitly mentioned "
    "that knob. Here are the interview questions and the user's answers:\n\n"
)


def extract_fields(answers: dict[str, str], provider: Provider) -> dict:
    """Ask the LLM to extract each field's value AND whether it was really answered.

    Pure (no I/O beyond the provider call), so it is testable with a fake provider.
    Returns the parsed per-field ``{"answered": bool, "value": ...}`` mapping; an
    unparseable reply yields ``{}``, which downstream treats as "nothing answered".
    """
    qa = "\n".join(f"Q ({field}): {q}\nA: {answers.get(field, '')}" for field, q in QUESTIONS)
    raw = provider.complete([{"role": "user", "content": _EXTRACT_INSTRUCTIONS + qa}])
    return _parse_json_object(raw)


def run_interview(ask, provider: Provider, notify=print) -> InterviewConfig:
    """Drive the interview: ask, detect non-answers, re-ask once, default + surface.

    `ask(prompt) -> str` is the human channel; `notify(str)` announces any default
    that had to be applied. Each unanswered field is re-asked at most once (no
    infinite loops); anything still unanswered falls back to the explained default.
    """
    answers = {field: ask(question) for field, question in QUESTIONS}
    fields = extract_fields(answers, provider)

    # Re-ask each not-really-answered field exactly once, with guidance + default.
    reasked = False
    for field, question in QUESTIONS:
        answered, _ = _field(fields, field)
        if not answered:
            reasked = True
            answers[field] = ask(
                f"Sorry, I need a clearer answer. {question} "
                f"({_GUIDANCE[field]}; or leave blank to use the default: {DEFAULTS[field]})"
            )
    if reasked:
        fields = extract_fields(answers, provider)

    return _build_config(fields, notify)


def _build_config(fields: dict, notify) -> InterviewConfig:
    """Assemble the config, applying and surfacing a default for any unfilled field."""
    values: dict[str, object] = {}
    for field in _ASKED_FIELDS + _EXTRA_FIELDS:
        answered, value = _field(fields, field)
        if field in ("failure_threshold", "rul_cap", "window"):
            resolved = _as_int(value) if answered else None
        else:
            resolved = str(value) if (answered and value not in (None, "")) else None
        if resolved is None:
            notify(_DEFAULT_NOTES[field])
            resolved = DEFAULTS[field]
        values[field] = resolved
    return InterviewConfig(**values)


def _field(fields: dict, name: str) -> tuple[bool, object]:
    """Read one field's ``(answered, value)``; tolerate a missing/garbage shape."""
    entry = fields.get(name)
    if isinstance(entry, dict):
        return bool(entry.get("answered")), entry.get("value")
    return False, None


def _parse_json_object(text: str) -> dict:
    """Best-effort extraction of the first ``{...}`` JSON object from LLM text."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return {}


def _as_int(value) -> int | None:
    """Coerce a possibly-stringy/None value to int, or ``None`` if it isn't one."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def interviewer_node(state: AgentState, config) -> dict:
    """Graph node: run the interview (with re-ask + surfaced defaults), emit config."""
    cfg = config["configurable"]
    ask = cfg["ask"]
    provider = cfg["provider_smart"]
    notify = cfg.get("notify", print)

    interview_config = run_interview(ask, provider, notify)
    return {
        "config": interview_config,
        "event": "interview_done",
        "log": append_log(state, f"interviewer: collected config {interview_config}"),
    }
