"""Graph state and the config the interviewer collects.

`AgentState` is the single dict that flows through every LangGraph node. Nodes
read the fields they need and return a partial dict that LangGraph merges back
in (last write wins, plain overwrite semantics - no reducers needed here).

Dependencies the nodes need but that aren't *data* (the LLM providers, the
training function, where to drop tickets, how to ask interview questions) do NOT
live in state - they're injected via LangGraph's `RunnableConfig["configurable"]`
so the graph stays a pure state machine and the tests can swap in fakes. See
`graph.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


@dataclass
class InterviewConfig:
    """The structured setup the interviewer extracts before any training runs.

    The free-text fields (`framing`, `reporting_cadence`, `success_metric`) are
    for the human/report; the numeric ones are the actual knobs that drive the
    DS core and the monitor.
    """

    framing: str  # plain-language: what are we predicting, for what equipment
    failure_threshold: int  # RUL (cycles) below which the monitor raises an alert
    reporting_cadence: str  # e.g. "after every training run", "daily"
    success_metric: str  # e.g. "held-out RMSE under 20 cycles"
    rul_cap: int = 125  # DS-core knob: piecewise-linear RUL cap
    window: int = 5  # DS-core knob: rolling-feature window


class AgentState(TypedDict, total=False):
    """Everything the graph passes between nodes.

    `event` is the wake signal the orchestrator routes on (see `graph.route`);
    sub-agents set it to the significant thing that just happened, never to raw
    progress. Everything else is the data those sub-agents produce.
    """

    event: str  # significant event driving routing (interview_done, run_finished, ...)
    config: InterviewConfig  # produced by the interviewer
    train_run: Any  # `training.TrainingRun` bundle produced by the trainer
    report: str  # produced by the report writer
    alerts: list[dict]  # produced by the monitor (one entry per alert/report)
    error: str  # set if training failed
    log: list[str]  # human-readable trace of what each node did


def append_log(state: AgentState, line: str) -> list[str]:
    """Return the running log with `line` appended (nodes overwrite `log`)."""
    return [*state.get("log", []), line]
