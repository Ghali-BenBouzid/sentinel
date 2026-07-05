"""Run the agent graph end to end against FD001.

    uv run python -m sentinel.agents            # scripted interview (unattended)
    uv run python -m sentinel.agents --interactive   # you answer the questions

Provider is chosen by `SENTINEL_LLM_PROVIDER` (default: groq, the free tier):

    groq       -> set GROQ_API_KEY        (free tier, zero cost - the default)
    anthropic  -> set ANTHROPIC_API_KEY

The graph runs interview -> train -> report -> monitor, printing each phase so
the flow is visible.
"""

from __future__ import annotations

import argparse

from langgraph.types import Command

from ..config import get_settings
from ..llm.provider import get_provider
from .graph import build_graph
from .training import run_training

# Canned answers so the demo runs unattended (and CI-safe) by default. The first
# reply takes the interviewer's up-front "use all defaults?" fast path, so the
# demo skips the interview and heads straight into training - the clearest thing
# to show unattended. The remaining answers are the fallback in case a run instead
# goes through the interview; each is a clear one-turn reply. Use --interactive to
# go through the conversation (and try front-loading info to see deduction).
SCRIPTED_ANSWERS = [
    "Yes please - just use sensible defaults for everything, quick demo.",
    "Remaining useful life of NASA C-MAPSS turbofan engines, so we can plan maintenance.",
    "Alert when an engine has 30 cycles of life left.",
    "A report after every training run.",
    "Held-out RMSE under 20 cycles.",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interactive", action="store_true", help="answer the interview yourself instead of scripted"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    provider_name = get_settings().sentinel_llm_provider
    print(f"[agent] provider={provider_name} (set SENTINEL_LLM_PROVIDER in env or .env)\n")

    configurable = {
        "provider_smart": get_provider("smart"),
        "provider_cheap": get_provider("cheap"),
        "train_fn": run_training,
        "ticket_dir": "artifacts/tickets",
        "thread_id": "cli",
    }
    graph = build_graph()
    thread = {"configurable": configurable}
    answers = None if args.interactive else iter(SCRIPTED_ANSWERS)

    print("[1/4] interview ...")
    inp = {"event": "start"}
    while True:
        for mode, chunk in graph.stream(inp, thread, stream_mode=["custom", "updates"]):
            if mode == "custom":
                print(f"  {chunk.get('text', chunk)}")
        state = graph.get_state(thread)
        if not state.tasks:  # no pending interrupt -> graph is done
            break
        prompt = state.tasks[0].interrupts[0].value
        reply = input(prompt + "\n> ") if args.interactive else next(answers, "")
        if not args.interactive:
            print(f"  Q: {prompt}\n  A: {reply}")
        inp = Command(resume=reply)

    final = graph.get_state(thread).values

    print("\n===== REPORT =====")
    print(final.get("report", "(no report)"))
    print("\n===== MONITOR =====")
    alerts = final.get("alerts", [])
    print(f"{len(alerts)} readings flagged; {sum(a['decision'] == 'alert' for a in alerts)} alerts filed.")
    for a in alerts[:10]:
        print(f"  unit {a['unit']}: predicted RUL {a['predicted_rul']} -> {a['decision']}")

    print("\n===== TRACE =====")
    for line in final.get("log", []):
        print(f"  {line}")


if __name__ == "__main__":
    main()
