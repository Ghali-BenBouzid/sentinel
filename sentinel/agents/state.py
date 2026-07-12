"""The structured run config the agent collects before training.

In V1 this file also held the graph's `AgentState`/`InterviewProgress` and a log
helper. Under the V2 agent the graph state is the message history (see
`sentinel/agents/agent.py`), so all that is gone - this file now carries only the
`InterviewConfig` shape that the `save_config` tool persists and the trainer/
report read.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InterviewConfig:
    """The structured setup collected before any training runs."""

    framing: str
    failure_threshold: int
    reporting_cadence: str
    success_metric: str
    rul_cap: int = 125
    window: int = 5
