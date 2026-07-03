"""Monitor sub-agent - post-training, steps through incoming readings.

MVP scope (per the design doc): a simple per-reading threshold check, no
wake-policy system. We replay the held-out FD001 test rows one at a time as if
they were live sensor readings, predict RUL for each, and `decide` what to do:

- **alert** + mock action (write a ticket file) when predicted RUL is at/under
  the interviewer's `failure_threshold`,
- **report** when it is within a warning band (`WARN_FACTOR` x threshold),
- **ok** otherwise (nothing recorded).

`decide` is a pure function so the threshold logic is unit-testable without a
model. The mock action is deliberately a local file write - no real ticketing.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import domain_context
from .state import AgentState, append_log

WARN_FACTOR = 2.0  # report (not alert) when RUL is within this multiple of the threshold

# The threshold is a Remaining-Useful-Life count, so ticket wording uses the same
# unit the glossary defines for the dataset (single source of truth for "cycles").
_RUL_UNITS = domain_context.DATASETS["fd001"].units


def decide(predicted_rul: float, threshold: int, warn_factor: float = WARN_FACTOR) -> str:
    """Return ``"alert"`` / ``"report"`` / ``"ok"`` for one predicted RUL."""
    if predicted_rul <= threshold:
        return "alert"
    if predicted_rul <= threshold * warn_factor:
        return "report"
    return "ok"


def _write_ticket(ticket_dir: Path, unit: int, predicted_rul: float, threshold: int) -> Path:
    """Mock action: drop a maintenance ticket file locally (no real system)."""
    ticket_dir.mkdir(parents=True, exist_ok=True)
    path = ticket_dir / f"ticket_unit_{unit}.json"
    path.write_text(
        json.dumps(
            {
                "unit": unit,
                "predicted_rul": round(predicted_rul, 1),
                "threshold": threshold,
                "message": (
                    f"Unit {unit} predicted RUL {predicted_rul:.1f} {_RUL_UNITS} "
                    f"<= {threshold} {_RUL_UNITS}; schedule maintenance."
                ),
            },
            indent=2,
        )
    )
    return path


def run_monitor(test_eval, predict, threshold: int, ticket_dir: Path) -> list[dict]:
    """Step through readings, decide per row, and file tickets on alerts.

    Prediction is batched once (fast) then walked row by row - the "stepping
    through readings" is the per-row `decide` loop. Returns one record per
    non-ok reading (the significant events the orchestrator/report care about).
    """
    preds = list(predict(test_eval))
    events: list[dict] = []
    for (_, row), pred in zip(test_eval.iterrows(), preds):
        pred = float(pred)
        action = decide(pred, threshold)
        if action == "ok":
            continue
        unit = int(row["unit"])
        record = {
            "unit": unit,
            "predicted_rul": round(pred, 1),
            "actual_rul": float(row["RUL"]) if "RUL" in row else None,
            "decision": action,
        }
        if action == "alert":
            record["ticket"] = str(_write_ticket(ticket_dir, unit, pred, threshold))
        events.append(record)
    return events


def monitor_node(state: AgentState, config) -> dict:
    """Graph node: run the monitor over the training run's held-out rows."""
    cfg = config["configurable"]
    ticket_dir = Path(cfg.get("ticket_dir", "artifacts/tickets"))
    run = state["train_run"]
    threshold = state["config"].failure_threshold

    events = run_monitor(run.test_eval, run.predict, threshold, ticket_dir)
    alerts = [e for e in events if e["decision"] == "alert"]
    line = f"monitor: {len(events)} readings flagged, {len(alerts)} alerts (tickets in {ticket_dir})"
    return {
        "alerts": events,
        "event": "monitor_done",
        "log": append_log(state, line),
    }
