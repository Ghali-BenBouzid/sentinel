"""Interviewer sub-agent - the only human-facing node, a turn-by-turn chatbot.

Pattern: **the code owns the agenda, the LLM owns the language.** The code walks
an ordered checklist (`QUESTIONS`) and resolves ONE field at a time; for each user
reply the LLM classifies it in context (grounded in the domain glossary) and
writes what to say next. There is no batch pass and no end-of-run dump - every bot
turn is an immediate reply to what the user just said.

Two behaviours are modelled on Cognireply's persona interviewer:

- **"Use all defaults?" fast path.** The very first turn offers a one-shot escape
  ("want me to just use sensible defaults so we can skip the interview?"). If the
  user accepts, every field takes its default, the config is printed, and the
  interview ends. The same instruction is honoured MID-interview: at any point,
  "use the default for everything / the rest" short-circuits the remaining
  questions (a global instruction must not get half-honoured).
- **Deduction (infer-and-confirm).** After each answer the LLM proposes, with a
  confidence score, values for still-open fields the conversation already implies.
  When the agenda reaches a field that has a confident deduced value, the bot
  CONFIRMS it ("Earlier you mentioned X - set the threshold to 30?") instead of
  asking cold, and the user accepts or corrects in one turn. Deductions are
  confidence-gated (`DEDUCE_CONFIDENCE`) and grounded - a guess just asks normally.

The state machine is pure: `start_progress()`/`advance()` carry the whole
conversation in an `InterviewProgress` dict so it can be checkpointed between
turns. The graph node `interviewer_turn` does exactly one `interrupt(next_prompt)`
per invocation (suspend, then resume with the user's reply into a single
`advance` classifier call) and loops back to itself until `phase == "done"`;
applied-default and ack lines stream out through the LangGraph stream writer.
`provider_smart` drives the conversation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field

from langgraph.types import interrupt

from ..llm.provider import Provider
from . import domain_context
from .state import AgentState, InterviewConfig, InterviewProgress, append_log

# The agenda. The code owns this ordered list; the LLM never reorders or adds to it.
QUESTIONS: list[tuple[str, str]] = [
    ("framing", "What are we trying to predict, and for what equipment?"),
    ("failure_threshold", "Below how many cycles of remaining life should we alert?"),
    ("reporting_cadence", "How often do you want status reports?"),
    ("success_metric", "What result would make this model a success to you?"),
]
_ASKED_FIELDS = [f for f, _ in QUESTIONS]

# Advanced knobs never put to the user; they always take their default here.
_EXTRA_DEFAULTS = {"rul_cap": 125, "window": 5}

# Sensible per-field defaults (offered mid-conversation and used on fallback/skip).
DEFAULTS: dict[str, object] = {
    "framing": "NASA C-MAPSS turbofan Remaining Useful Life (RUL) prediction",
    "failure_threshold": 30,
    "reporting_cadence": "after every training run",
    "success_metric": "held-out RMSE under 20 cycles",
    **_EXTRA_DEFAULTS,
}

# After this many genuine non-answers on ONE field, fall back to its default.
MAX_NONANSWERS = 3
# A deduced value is confirmed (not asked cold) only at/above this confidence.
DEDUCE_CONFIDENCE = 0.6

CLEAR, UNCLEAR, QUESTION, WANTS_DEFAULT, ALL_DEFAULTS = (
    "CLEAR",
    "UNCLEAR",
    "QUESTION",
    "WANTS_DEFAULT",
    "ALL_DEFAULTS",
)

# The up-front one-shot escape. Offered as the interviewer's first message.
GATE_QUESTION = (
    "Want me to just use sensible defaults for everything so we can skip the interview? "
    "(great for a quick demo.) Or we can go through it together to tune it - your call."
)

# Short human-facing announcement of a default that gets applied.
_DEFAULT_LINE: dict[str, str] = {
    "framing": f"Problem framing: {DEFAULTS['framing']} (default)",
    "failure_threshold": f"Alert threshold: {DEFAULTS['failure_threshold']} cycles (default)",
    "reporting_cadence": f"Reporting: {DEFAULTS['reporting_cadence']} (default)",
    "success_metric": f"Success target: {DEFAULTS['success_metric']} (default)",
}

# How the bot confirms a deduced value instead of asking cold (grounded, per field).
_CONFIRM: dict[str, str] = {
    "framing": "From what you said, I'll frame this as {v}. Sound right, or would you put it differently?",
    "failure_threshold": (
        "Earlier it sounded like alerting around {v} cycles - shall I set the threshold to {v}? "
        "(or give me a different number.)"
    ),
    "reporting_cadence": "You hinted at reporting {v} - shall I go with that?",
    "success_metric": "For success you mentioned {v} - shall I use that as the target?",
}

_gloss = domain_context.glossary()

# Gate classifier: does the user want the all-defaults shortcut? Kept tiny and
# conservative - only a clear acceptance skips the interview.
_GATE_SYSTEM_PROMPT = (
    "The user was just asked whether to SKIP a short setup interview and use sensible "
    "defaults for everything (a quick-demo shortcut), or go through the interview. "
    "Decide if their reply CLEARLY opts into the all-defaults shortcut. True only for a "
    'clear acceptance ("yes", "use defaults", "just use defaults", "skip it", "defaults '
    'are fine", "quick demo please"). Anything else - "no", "let\'s go through it", a '
    "question, or an actual answer to a setup question - is false.\n"
    'Return ONLY JSON: {"all_defaults": true|false}'
)

# Per-turn classifier (TIDD-EC): role + the five moves + Do/Don't + glossary + JSON
# output + worked examples. The glossary is the single source of domain facts.
_SYSTEM_PROMPT = (
    "You are Sentinel's setup assistant: a friendly, concise predictive-maintenance "
    "expert running a short spoken-style interview to configure a model-training run. "
    "You handle ONE field at a time and reply, in the moment, to whatever the user just "
    "said. Never move on until the active field is resolved.\n\n"
    "Use ONLY this glossary as your source of domain facts - do not invent metrics, "
    "numbers, or options that are not in it:\n\n"
    f"{_gloss}\n\n"
    "Each turn you are given the ACTIVE field, its question, its DEFAULT, the recent "
    "conversation, and (sometimes) a deduced value already shown for confirmation. "
    "Classify the user's latest reply and write your spoken response:\n"
    "- CLEAR: the reply genuinely answers THIS field (or, if a deduced value was shown, "
    "the user accepts it - then echo that value). Extract the value; acknowledge in one "
    "short sentence.\n"
    "- UNCLEAR: empty, evasive, or off-topic. Immediately ask for a clearer answer AND "
    "offer the default in the same breath (e.g. \"...or I can use the default of 30 "
    'cycles - want that?").\n'
    "- QUESTION: the user asks you something or wants an explanation before deciding. "
    "Answer it in plain language from the glossary, mention the default, THEN re-ask this "
    "field's question in the same reply. Do NOT treat this as an answer.\n"
    "- WANTS_DEFAULT: the user defers on THIS field only (\"you decide\", \"skip it\", "
    '"use the default"). Use the default and say what you chose in one line.\n'
    "- ALL_DEFAULTS: the user asks to use defaults for EVERYTHING / the rest / all future "
    "questions (\"just use defaults for everything\", \"defaults for the rest\", \"stop "
    'asking, use defaults"). Acknowledge that you\'ll fill everything else with defaults.\n\n'
    "Also, in `deduced`, propose values for any OTHER still-open interview fields "
    f"({', '.join(_ASKED_FIELDS)}) that the user's words already imply, each with a "
    "confidence 0-1. Only include a field the conversation genuinely supports; a mere "
    "guess stays below 0.6. NEVER include the active field in `deduced`, and never invent "
    "a value the conversation does not support.\n\n"
    "DO: ground every explanation and number in the glossary/default; keep replies to 1-3 "
    "warm, direct sentences; give failure_threshold as an integer number of cycles; phrase "
    "a success_metric in the glossary's metric names/units when implied.\n"
    "DO NOT: move on while the field is unresolved; treat a question as an answer; invent "
    "facts, numbers, or deductions the conversation does not support.\n\n"
    "Return ONLY a JSON object (no prose, no code fences):\n"
    '{"classification": "CLEAR|UNCLEAR|QUESTION|WANTS_DEFAULT|ALL_DEFAULTS", '
    '"reply": "the exact message to say now", '
    '"value": <the extracted value for the ACTIVE field if CLEAR, else null>, '
    '"deduced": [{"field": "<other field>", "value": <v>, "confidence": <0-1>}]}\n\n'
    "Examples:\n"
    'Active failure_threshold, user "50 cycles" -> '
    '{"classification": "CLEAR", "reply": "Got it - I\'ll alert below 50 cycles.", "value": 50, "deduced": []}\n'
    'Active failure_threshold, deduced value shown was 25, user "yeah that works" -> '
    '{"classification": "CLEAR", "reply": "Great, alerting below 25 cycles.", "value": 25, "deduced": []}\n'
    'Active framing, user "predict turbofan RUL; alert me around 25 cycles and I want RMSE under 20" -> '
    '{"classification": "CLEAR", "reply": "Got it - turbofan RUL prediction.", '
    '"value": "turbofan Remaining Useful Life prediction", '
    '"deduced": [{"field": "failure_threshold", "value": 25, "confidence": 0.9}, '
    '{"field": "success_metric", "value": "held-out RMSE under 20 cycles", "confidence": 0.85}]}\n'
    'Active success_metric, user "what does RMSE mean?" -> '
    '{"classification": "QUESTION", "reply": "RMSE is the model\'s typical error in cycles - '
    "lower is better, and it's an accuracy measure, not a prediction of remaining life. A "
    'common target is held-out RMSE under 20 cycles. So - what would success look like for '
    'you?", "value": null, "deduced": []}\n'
    'Active reporting_cadence, user "just use defaults for everything from here" -> '
    '{"classification": "ALL_DEFAULTS", "reply": "Sounds good - I\'ll fill everything else '
    'with sensible defaults.", "value": null, "deduced": []}'
)


@dataclass
class Turn:
    """One LLM per-turn decision: classification, next message, value, deductions."""

    classification: str
    reply: str
    value: object
    deduced: list[dict] = dataclass_field(default_factory=list)


def classify_gate(reply: str, provider: Provider) -> bool:
    """Return True only if the user clearly wants the all-defaults shortcut."""
    raw = provider.complete(
        [{"role": "system", "content": _GATE_SYSTEM_PROMPT}, {"role": "user", "content": reply}]
    )
    return bool(_parse_json_object(raw).get("all_defaults"))


def classify_turn(field: str, question: str, history: list[str], reply: str, deduced_value, provider: Provider) -> Turn:
    """Classify the user's `reply` to the active `field` and produce the next message.

    `deduced_value` is a value already shown for confirmation (or None). Pure apart
    from the provider call; a malformed reply degrades to UNCLEAR rather than crashing.
    """
    ded_line = f"\n<deduced_value_shown>{deduced_value}</deduced_value_shown>" if deduced_value is not None else ""
    user_msg = (
        f"<active_field>{field}</active_field>\n"
        f"<question>{question}</question>\n"
        f"<default>{DEFAULTS[field]}</default>{ded_line}\n"
        f"<recent_conversation>\n{chr(10).join(history)}\n</recent_conversation>\n"
        f"<latest_user_reply>{reply}</latest_user_reply>"
    )
    raw = provider.complete(
        [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
    )
    data = _parse_json_object(raw)
    classification = data.get("classification")
    if classification not in (CLEAR, UNCLEAR, QUESTION, WANTS_DEFAULT, ALL_DEFAULTS):
        classification = UNCLEAR
    deduced = data.get("deduced") if isinstance(data.get("deduced"), list) else []
    return Turn(classification, str(data.get("reply") or ""), data.get("value"), deduced)


def _absorb_deductions(deduced: dict, proposals: list[dict], active: str, resolved: set[str]) -> None:
    """Store confident deductions for still-open fields (Cognireply's `_deduce` gate)."""
    for p in proposals:
        if not isinstance(p, dict):
            continue
        key = p.get("field")
        if key not in _ASKED_FIELDS or key == active or key in resolved:
            continue
        try:
            conf = float(p.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        value = _coerce(key, p.get("value")) if key == "failure_threshold" else p.get("value")
        if conf >= DEDUCE_CONFIDENCE and value not in (None, ""):
            deduced[key] = value  # latest confident deduction wins


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


def start_progress() -> InterviewProgress:
    """The initial per-turn state: the gate question, nothing resolved yet."""
    return {
        "phase": "gate",
        "active_index": 0,
        "values": {},
        "deduced": {},
        "resolved": [],
        "history": [],
        "next_prompt": GATE_QUESTION,
        "nonanswers": 0,
        "notices": [],
        "config": None,
    }


def _finish(values: dict, notices: list, opener: str) -> InterviewProgress:
    """Fill every unanswered asked field with its default, recording announcements."""
    notices.append(opener)
    for f in _ASKED_FIELDS:
        if f not in values:
            notices.append(f"  - {_DEFAULT_LINE[f]}")
            values[f] = DEFAULTS[f]
    return {
        "phase": "done",
        "notices": notices,
        "config": InterviewConfig(**values, **_EXTRA_DEFAULTS),
    }


def _prompt_for(field: str, question: str, deduced: dict, preamble: str) -> str:
    """The next bot message for `field`: confirm a deduced value, or ask cold."""
    body = _CONFIRM[field].format(v=deduced[field]) if deduced.get(field) is not None else question
    return f"{preamble}\n\n{body}".strip() if preamble else body


def advance(progress: InterviewProgress, reply: str, provider: Provider) -> InterviewProgress:
    """Consume one user reply; return the next InterviewProgress. One LLM call/turn."""
    p = {**progress, "notices": []}  # notices are per-turn
    p["values"] = dict(progress["values"])  # deep-ish copy: never mutate the caller's snapshot
    p["deduced"] = dict(progress["deduced"])

    if p["phase"] == "gate":
        if classify_gate(reply, provider):
            return _finish(
                p["values"], p["notices"], "Great - using sensible defaults across the board:"
            )
        p["phase"] = "field"
        field, question = QUESTIONS[0]
        p["next_prompt"] = _prompt_for(field, question, p["deduced"], "")
        return p

    field, question = QUESTIONS[p["active_index"]]
    ded_value = p["deduced"].get(field)
    p["history"] = [*p["history"], f"Assistant: {p['next_prompt']}", f"User: {reply}"]
    turn = classify_turn(field, question, p["history"], reply, ded_value, provider)
    _absorb_deductions(p["deduced"], turn.deduced, active=field, resolved=set(p["resolved"]))

    if turn.classification == ALL_DEFAULTS:
        return _finish(p["values"], p["notices"], "Okay - filling everything else with defaults:")

    ack = ""
    if turn.classification == CLEAR:
        value = _coerce(field, turn.value)
        if value is None and ded_value is not None:
            value = _coerce(field, ded_value)
        if value is not None:
            p["values"][field] = value
            ack = turn.reply
        else:
            turn.classification = UNCLEAR
    if turn.classification == WANTS_DEFAULT:
        p["values"][field] = DEFAULTS[field]
        ack = turn.reply or f"Okay, I'll go with the default: {DEFAULTS[field]}."
    if turn.classification == QUESTION:
        p["next_prompt"] = turn.reply or question  # re-ask same field, no advance
        return p
    if turn.classification == UNCLEAR:
        p["nonanswers"] += 1
        if p["nonanswers"] < MAX_NONANSWERS:
            p["next_prompt"] = turn.reply or f"I need a clearer answer. {question}"
            return p
        p["values"][field] = DEFAULTS[field]
        ack = f"Let's not get stuck - I'll use the default of {DEFAULTS[field]}."

    # field resolved -> advance
    p["resolved"] = [*p["resolved"], field]
    p["nonanswers"] = 0
    nxt = p["active_index"] + 1
    if nxt >= len(QUESTIONS):
        if ack:
            p["notices"] = [*p["notices"], ack]
        p["phase"] = "done"
        p["config"] = InterviewConfig(**p["values"], **_EXTRA_DEFAULTS)
        return p
    p["active_index"] = nxt
    nf, nq = QUESTIONS[nxt]
    p["next_prompt"] = _prompt_for(nf, nq, p["deduced"], ack)
    return p


def _get_writer():
    """The active LangGraph stream writer, or a no-op when there's no stream
    (e.g. `graph.invoke` in tests) so streaming notices never crash the node."""
    from langgraph.config import get_stream_writer

    try:
        return get_stream_writer()
    except Exception:  # noqa: BLE001 - no active stream context
        return lambda _e: None


def interviewer_turn(state: AgentState, config) -> dict:
    """One question/answer round of the interview, then loop back to self.

    Does exactly ONE `interrupt()` per invocation: it suspends on the current
    prompt and, on resume, receives the user's reply and makes the single
    classifier call in `advance`. Because each turn is its own invocation,
    resuming re-executes only THIS turn - earlier turns are never replayed.
    """
    provider = config["configurable"]["provider_smart"]
    progress = state.get("interview") or start_progress()

    reply = interrupt(progress["next_prompt"])  # suspends; returns the saved reply on resume
    progress = advance(progress, reply, provider)

    writer = _get_writer()
    for line in progress.get("notices", []):
        writer({"type": "notify", "text": line})

    out = {
        "interview": progress,
        "log": append_log(state, "interviewer: turn"),
    }
    if progress["phase"] == "done":
        out["config"] = progress["config"]
        out["event"] = "interview_done"
    return out
