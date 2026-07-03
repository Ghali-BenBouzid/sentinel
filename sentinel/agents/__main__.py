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

from ..config import get_settings
from ..llm.provider import get_provider
from .graph import build_graph
from .training import run_training

# Canned answers so the demo runs unattended (and CI-safe) by default. The LLM
# still extracts structure from these - only the human typing is skipped.
SCRIPTED_ANSWERS = [
    "Remaining useful life of NASA C-MAPSS turbofan engines, so we can plan maintenance.",
    "Alert when an engine has 30 cycles of life left.",
    "A report after every training run.",
    "Held-out RMSE under 20 cycles.",
]


def _scripted_ask():
    answers = iter(SCRIPTED_ANSWERS)

    def ask(question: str) -> str:
        answer = next(answers, "")
        print(f"  Q: {question}\n  A: {answer}")
        return answer

    return ask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interactive", action="store_true", help="answer the interview yourself instead of scripted"
    )
    args = parser.parse_args()

    provider_name = get_settings().sentinel_llm_provider
    print(f"[agent] provider={provider_name} (set SENTINEL_LLM_PROVIDER in env or .env)\n")

    ask = input if args.interactive else _scripted_ask()
    configurable = {
        "ask": ask,
        "provider_smart": get_provider("smart"),
        "provider_cheap": get_provider("cheap"),
        "train_fn": run_training,
        "ticket_dir": "artifacts/tickets",
    }

    graph = build_graph()
    print("[1/4] interview ...")
    final = graph.invoke({"event": "start"}, config={"configurable": configurable})

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
