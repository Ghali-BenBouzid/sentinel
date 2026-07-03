"""The Sentinel agent layer (Milestone 2).

A LangGraph graph wrapping the M1 DS core: an orchestrator that routes on
significant events, plus interviewer / report-writer / monitor sub-agents. See
`docs/learning/02-agent-layer.md` and `docs/pdm-agent-design.md`.
"""

from .graph import build_graph, route
from .state import AgentState, InterviewConfig

__all__ = ["build_graph", "route", "AgentState", "InterviewConfig"]
