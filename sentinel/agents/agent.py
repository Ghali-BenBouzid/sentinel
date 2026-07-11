"""The V2 agent hub: a LangChain reasoning loop over DS tools."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
)

from ..config import get_settings
from ..llm.provider import get_chat_model
from . import domain_context
from .harness import (
    InvalidToolCallMiddleware,
    ModelFailureFormatterMiddleware,
    guarded_when,
    make_corrective_feedback,
)
from .registry import Registry
from .tools import make_tools


class DSAgentState(AgentState):
    """Agent messages and bookkeeping plus per-session autonomy."""

    autonomy: str


SYSTEM_PROMPT = (
    "You are Sentinel, an autonomous data-scientist agent for predictive "
    "maintenance on NASA C-MAPSS turbofan Remaining-Useful-Life prediction.\n\n"
    "Act only through your tools. Never claim to have trained, compared, "
    "promoted, or monitored anything except by calling the matching tool and "
    "reporting its result.\n"
    "Before the first training run, gather the run configuration "
    "conversationally: what to predict and for what equipment, the RUL failure "
    "threshold in cycles, reporting cadence, and success metric. Then call "
    "save_config. If the user requests sensible defaults, save them and "
    "proceed.\n"
    "Do not ask the user to confirm destructive or expensive actions yourself. "
    "The system confirms train, retrain, promote, delete, and run_monitor. Call "
    "the tool directly; respect a Declined result.\n"
    "Metrics are comparable only within one rul_cap/window configuration. The "
    "compare tool handles re-evaluation.\n"
    "A training run compares many model families but registers only the winner. To "
    "act on a non-winner (e.g. retrain the second-best model from a comparison), "
    "call inspect('leaderboard') to see the ranked models, then retrain by the "
    "chosen row's 'id'.\n\n"
    "<glossary>\n"
    + domain_context.glossary()
    + "\n</glossary>"
)


def _default_checkpointer():
    """Create an app-lifetime SQLite checkpointer."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = get_settings().checkpoint_db_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(
        sqlite3.connect(path, check_same_thread=False)
    )


def _default_fallback_model():
    """Build the alternate-provider model used after primary retries fail."""
    return get_chat_model("smart", name="anthropic")


def build_agent(
    *,
    chat_model,
    train_fn,
    retrain_fn,
    tools_chat_model,
    ticket_dir,
    models_dir,
    checkpointer=None,
    fallback_chat_model=None,
):
    """Assemble the registry, tools, and create_agent hub."""
    registry = Registry(models_dir)
    tools = make_tools(
        train_fn=train_fn,
        retrain_fn=retrain_fn,
        chat_model=tools_chat_model,
        ticket_dir=ticket_dir,
        registry=registry,
    )
    if checkpointer is None:
        checkpointer = _default_checkpointer()
    if fallback_chat_model is None:
        fallback_chat_model = _default_fallback_model()
    settings = get_settings()
    corrective_feedback = make_corrective_feedback(tools_chat_model, registry)
    middleware = [
        ModelCallLimitMiddleware(
            thread_limit=settings.sentinel_model_call_thread_limit,
            run_limit=settings.sentinel_model_call_run_limit,
        ),
        ToolCallLimitMiddleware(
            thread_limit=settings.sentinel_tool_call_thread_limit,
            run_limit=settings.sentinel_tool_call_run_limit,
        ),
        ModelFailureFormatterMiddleware(corrective_feedback),
        ModelFallbackMiddleware(fallback_chat_model),
        ModelRetryMiddleware(
            max_retries=settings.sentinel_retry_max_attempts,
            on_failure="error",
        ),
        InvalidToolCallMiddleware(corrective_feedback),
        HumanInTheLoopMiddleware(
            interrupt_on={
                name: {
                    "allowed_decisions": ["approve", "reject"],
                    "when": guarded_when(name),
                }
                for name in ("train", "retrain", "promote", "delete", "run_monitor")
            },
        ),
    ]
    return create_agent(
        chat_model,
        tools,
        system_prompt=SYSTEM_PROMPT,
        state_schema=DSAgentState,
        checkpointer=checkpointer,
        middleware=middleware,
    )
