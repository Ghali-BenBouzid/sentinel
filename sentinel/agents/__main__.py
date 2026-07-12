"""Chat with the Sentinel data-scientist agent.

    uv run python -m sentinel.agents
    uv run python -m sentinel.agents --autonomous
"""
from __future__ import annotations

import argparse

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from ..config import configure_langsmith, get_settings
from ..llm.provider import get_chat_model
from .agent import build_agent
from .training import run_retraining, run_training


def run_turn(agent, thread, inp, out=print) -> bool:
    """Stream one graph leg and report whether confirmation is pending."""
    for mode, chunk in agent.stream(
        inp, thread, stream_mode=["custom", "updates"]
    ):
        if mode == "custom":
            out(chunk.get("text", chunk))
        elif mode == "updates":
            for update in (chunk or {}).values():
                messages = (
                    update.get("messages")
                    if isinstance(update, dict)
                    else None
                )
                for message in messages or []:
                    if getattr(message, "content", ""):
                        out(message.content)
    return bool(agent.get_state(thread).interrupts)


def main() -> None:
    configure_langsmith()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="skip confirmations",
    )
    args = parser.parse_args()
    autonomy = (
        "autonomous"
        if args.autonomous
        else get_settings().sentinel_autonomy
    )

    agent = build_agent(
        chat_model=get_chat_model("smart"),
        train_fn=run_training,
        retrain_fn=run_retraining,
        tools_chat_model=get_chat_model("cheap"),
        ticket_dir="artifacts/tickets",
        models_dir="artifacts/models",
        checkpointer=None,
    )
    thread = {"configurable": {"thread_id": "cli"}}
    print(
        f"[agent] autonomy={autonomy}. "
        "Type your request (Ctrl-D to exit)."
    )
    first = True
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        graph_input = {"messages": [HumanMessage(line)]}
        if first:
            graph_input["autonomy"] = autonomy
            first = False
        pending = run_turn(agent, thread, graph_input)
        while pending:
            state = agent.get_state(thread)
            request = state.interrupts[0].value
            decisions = []
            for action in request["action_requests"]:
                answer = input(
                    f"Confirm {action['name']} ({action['args']})? [y/N] "
                )
                if answer.strip().lower() in {"y", "yes"}:
                    decisions.append({"type": "approve"})
                else:
                    decisions.append({"type": "reject"})
            pending = run_turn(
                agent,
                thread,
                Command(resume={"decisions": decisions}),
            )


if __name__ == "__main__":
    main()
