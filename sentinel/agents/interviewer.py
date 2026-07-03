"""Interviewer sub-agent - the only human-facing node, a turn-by-turn chatbot.

Pattern: **the code owns the agenda, the LLM owns the language.** The code walks
an ordered checklist (`QUESTIONS`) and resolves ONE field at a time; for each
user reply the LLM classifies it in context (grounded in the domain glossary) and
writes the exact thing to say next. There is no batch pass and no end-of-run dump
of assumptions - every bot turn is an immediate reply to what the user just said.

Per turn the LLM returns one of four moves for the active field:
- CLEAR: the reply answers it -> extract the value, acknowledge, next field.
- UNCLEAR: evasive/empty/off-topic -> ask again AND offer the default in the same
  breath; stay on the field. Counts toward the retry bound.
- QUESTION: the user wants an explanation -> answer it from the glossary, then
  re-ask; do NOT consume it as an answer, and do NOT count it.
- WANTS_DEFAULT: the user defers ("you decide") -> use the default, say what was
  chosen and why, next field.

After `MAX_NONANSWERS` genuine non-answers on one field the code falls back to the
default (with a one-line heads-up) and moves on, so it can never loop forever.

Each bot message is one `ask(prompt) -> str` call (return = the user's reply), so
the node runs interactively for the captain, unattended with scripted replies in
the demo, and offline in tests with a fake `ask`. `provider_smart` drives the
conversation; a final closing line, if any, goes through the injected `notify`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..llm.provider import Provider
from . import domain_context
from .state import AgentState, InterviewConfig, append_log

# The agenda. The code owns this ordered list; the LLM never reorders or adds to it.
QUESTIONS: list[tuple[str, str]] = [
    ("framing", "What are we trying to predict, and for what equipment?"),
    ("failure_threshold", "Below how many cycles of remaining life should we alert?"),
    ("reporting_cadence", "How often do you want status reports?"),
    ("success_metric", "What result would make this model a success to you?"),
]

# Advanced knobs never put to the user; they always take their default here.
_EXTRA_DEFAULTS = {"rul_cap": 125, "window": 5}

# Sensible per-field defaults (offered mid-conversation and used on fallback).
DEFAULTS: dict[str, object] = {
    "framing": "NASA C-MAPSS turbofan Remaining Useful Life (RUL) prediction",
    "failure_threshold": 30,
    "reporting_cadence": "after every training run",
    "success_metric": "held-out RMSE under 20 cycles",
    **_EXTRA_DEFAULTS,
}

# After this many genuine non-answers on ONE field, fall back to its default.
MAX_NONANSWERS = 3

CLEAR, UNCLEAR, QUESTION, WANTS_DEFAULT = "CLEAR", "UNCLEAR", "QUESTION", "WANTS_DEFAULT"

# System prompt (TIDD-EC): role + the four moves + Do/Don't + glossary + JSON
# output, with a couple of worked examples to lock the format. The glossary is the
# single source of domain facts so explanations and defaults stay correct.
_SYSTEM_PROMPT = (
    "You are Sentinel's setup assistant: a friendly, concise predictive-maintenance "
    "expert running a short spoken-style interview to configure a model-training run. "
    "You handle ONE field at a time and reply, in the moment, to whatever the user just "
    "said. Never move on until the active field is resolved.\n\n"
    "Use ONLY this glossary as your source of domain facts - do not invent metrics, "
    "numbers, or options that are not in it:\n\n"
    f"{domain_context.glossary()}\n\n"
    "Each turn you are given the ACTIVE field, its question, its DEFAULT, and the "
    "user's latest reply (with the recent exchange). Classify the reply and write your "
    "spoken response:\n"
    "- CLEAR: the reply genuinely answers THIS field. Extract the value and acknowledge "
    "in one short sentence.\n"
    "- UNCLEAR: empty, evasive, or off-topic (\"you tell me\", \"I don't know\", or "
    '"never" where it makes no sense). Immediately ask for a clearer answer AND offer '
    "the default in the same breath (e.g. \"...or I can use the default of 30 cycles - "
    'want that?").\n'
    "- QUESTION: the user is asking you something or wants an explanation before "
    "deciding (\"what does RMSE mean?\", \"explain so I can choose\", \"what are my "
    "options?\"). Answer it in plain language from the glossary, mention the default, "
    "THEN re-ask this field's question in the same reply. Do NOT treat this as an answer.\n"
    "- WANTS_DEFAULT: the user defers to you (\"you decide\", \"choose for me\", \"use "
    "the default\", \"whatever you think\"). Use the default and tell them what you "
    "chose and why in one line.\n\n"
    "DO: ground every explanation and number in the glossary/default; keep replies to "
    "1-3 warm, direct sentences; give failure_threshold as an integer number of cycles; "
    "phrase a success_metric in the glossary's metric names/units when the user implies "
    "one.\n"
    "DO NOT: move on while the field is unresolved; treat a user's question as their "
    "answer; invent facts not in the glossary.\n\n"
    "Return ONLY a JSON object (no prose, no code fences):\n"
    '{"classification": "CLEAR|UNCLEAR|QUESTION|WANTS_DEFAULT", '
    '"reply": "the exact message to say to the user now", '
    '"value": <the extracted value if CLEAR, else null>}\n\n'
    "Examples:\n"
    'Field failure_threshold, user says "50 cycles" -> '
    '{"classification": "CLEAR", "reply": "Got it - I\'ll alert below 50 cycles.", "value": 50}\n'
    'Field failure_threshold, user says "you tell me" -> '
    '{"classification": "UNCLEAR", "reply": "I need a number here - roughly how many '
    'cycles of life left should trigger an alert? Or I can use the default of 30 cycles - '
    'want that?", "value": null}\n'
    'Field success_metric, user says "what does RMSE mean?" -> '
    '{"classification": "QUESTION", "reply": "RMSE is the model\'s typical error in '
    "cycles - lower is better, and it's an accuracy measure, not a prediction of "
    'remaining life. A common target is held-out RMSE under 20 cycles. So - what result '
    'would make this a success for you?", "value": null}\n'
    'Field reporting_cadence, user says "you decide" -> '
    '{"classification": "WANTS_DEFAULT", "reply": "No problem - I\'ll send a report after '
    'every training run, which is the usual default.", "value": null}'
)


@dataclass
class Turn:
    """One LLM per-turn decision: how to classify the reply and what to say next."""

    classification: str
    reply: str
    value: object


def classify_turn(field: str, question: str, transcript: list[str], reply: str, provider: Provider) -> Turn:
    """Classify the user's `reply` to the active `field` and produce the next message.

    Pure apart from the provider call, so it is testable with a fake provider. A
    malformed reply degrades to UNCLEAR (ask again) rather than crashing the loop.
    """
    user_msg = (
        f"<active_field>{field}</active_field>\n"
        f"<question>{question}</question>\n"
        f"<default>{DEFAULTS[field]}</default>\n"
        f"<recent_exchange>\n{chr(10).join(transcript)}\n</recent_exchange>\n"
        f"<latest_user_reply>{reply}</latest_user_reply>"
    )
    raw = provider.complete(
        [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
    )
    data = _parse_json_object(raw)
    classification = data.get("classification")
    if classification not in (CLEAR, UNCLEAR, QUESTION, WANTS_DEFAULT):
        classification = UNCLEAR
    return Turn(classification=classification, reply=str(data.get("reply") or ""), value=data.get("value"))


def _resolve_field(field: str, question: str, ask, provider: Provider, preamble: str) -> tuple[object, str]:
    """Converse until `field` is resolved; return (value, closing_ack_for_next_turn).

    `preamble` (an acknowledgement carried from the previous field) is prepended to
    this field's opening question so the flow reads as one continuous chat.
    """
    default = DEFAULTS[field]
    transcript: list[str] = []
    opening = f"{preamble}\n\n{question}".strip() if preamble else question
    reply = ask(opening)
    transcript += [f"Assistant: {opening}", f"User: {reply}"]

    nonanswers = 0
    while True:
        turn = classify_turn(field, question, transcript, reply, provider)

        if turn.classification == CLEAR:
            value = _coerce(field, turn.value)
            if value is not None:
                return value, turn.reply  # ack rides along to the next field's opening
            turn.classification = UNCLEAR  # "clear" but unusable value -> treat as non-answer

        if turn.classification == WANTS_DEFAULT:
            ack = turn.reply or f"Okay, I'll go with the default: {default}."
            return default, ack

        if turn.classification == QUESTION:
            # Answer + re-ask in one message; a question is not a failed answer.
            bot_msg = turn.reply or question
        else:  # UNCLEAR
            nonanswers += 1
            if nonanswers >= MAX_NONANSWERS:
                return default, f"Let's not get stuck - I'll go with the default of {default} for now."
            bot_msg = turn.reply or f"I need a clearer answer. {question}"

        reply = ask(bot_msg)
        transcript += [f"Assistant: {bot_msg}", f"User: {reply}"]


def run_interview(ask, provider: Provider, notify=print) -> InterviewConfig:
    """Drive the whole interview as a turn-by-turn chat, one field fully at a time.

    `ask(prompt) -> str` is the human channel (each call is one bot message);
    `notify(str)` delivers a single closing line at the end, if any.
    """
    values: dict[str, object] = {}
    preamble = ""
    for field, question in QUESTIONS:
        value, ack = _resolve_field(field, question, ask, provider, preamble)
        values[field] = value
        preamble = ack  # becomes the lead-in to the next field's question
    if preamble:
        notify(preamble)  # close the conversation on the last field's acknowledgement
    return InterviewConfig(**values, **_EXTRA_DEFAULTS)


def _coerce(field: str, value: object) -> object | None:
    """Coerce an extracted value to the field's type; None means 'unusable'."""
    if field == "failure_threshold":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    text = str(value).strip() if value is not None else ""
    return text or None


def _parse_json_object(text: str) -> dict:
    """Best-effort extraction of the first ``{...}`` JSON object from LLM text."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return {}


def interviewer_node(state: AgentState, config) -> dict:
    """Graph node: run the conversational interview, emit the collected config."""
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
