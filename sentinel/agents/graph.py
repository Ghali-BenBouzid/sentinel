"""The LangGraph graph: orchestrator + three sub-agents + a trainer node.

Shape is hub-and-spoke. The **orchestrator** is the hub: every sub-agent routes
back to it, and it dispatches the next one based on the `event` in state. It is
"woken by events, not progress" - the router only ever looks at `state["event"]`,
which sub-agents set to significant milestones (interview_done, run_finished,
run_failed, report_ready, monitor_done). There are no progress ticks in the
graph to react to, which is exactly the point.

    START -> orchestrator --(event)--> interviewer / trainer / report_writer / monitor / END
                  ^                                   |
                  +-----------------------------------+   (every sub-agent returns to the hub)

The **trainer** node is not an LLM sub-agent - it is the M1 DS-core invocation
(via `config["configurable"]["train_fn"]`). The orchestrator dispatches it and
is woken by its `run_finished` / `run_failed` event, matching the design's event
vocabulary.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .interviewer import interviewer_node
from .monitor import monitor_node
from .report_writer import report_writer_node
from .state import AgentState, append_log

# event -> next node. `None`/"start" means "no config yet, go interview".
_ROUTES = {
    None: "interviewer",
    "start": "interviewer",
    "interview_done": "trainer",
    "run_finished": "report_writer",
    "run_failed": "report_writer",
    "report_ready": "monitor",
    "monitor_done": END,
    "failed_reported": END,
}


def route(state: AgentState) -> str:
    """Pick the next node from the last significant event (the wake signal)."""
    event = state.get("event")
    if event not in _ROUTES:
        raise ValueError(f"orchestrator got unknown event {event!r}")
    return _ROUTES[event]


def orchestrator_node(state: AgentState) -> dict:
    """The hub. Owns no work of its own - routing happens on the outgoing edge."""
    return {"log": append_log(state, f"orchestrator: event={state.get('event')!r}")}


def trainer_node(state: AgentState, config) -> dict:
    """Run the M1 DS core via the injected `train_fn`; emit run_finished/run_failed."""
    train_fn = config["configurable"]["train_fn"]
    try:
        run = train_fn(state["config"])
    except Exception as exc:  # noqa: BLE001 - surface any DS-core failure as an event
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "event": "run_failed",
            "log": append_log(state, f"trainer: run FAILED ({type(exc).__name__})"),
        }
    m = run.result.metrics
    line = f"trainer: run finished, held-out RMSE={m['rmse']:.2f} R2={m['r2']:.3f}"
    return {"train_run": run, "event": "run_finished", "log": append_log(state, line)}


def build_graph():
    """Assemble and compile the StateGraph. Dependencies come in via config."""
    graph = StateGraph(AgentState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("interviewer", interviewer_node)
    graph.add_node("trainer", trainer_node)
    graph.add_node("report_writer", report_writer_node)
    graph.add_node("monitor", monitor_node)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", route)
    # Every sub-agent reports back to the hub, which re-routes on the new event.
    for node in ("interviewer", "trainer", "report_writer", "monitor"):
        graph.add_edge(node, "orchestrator")

    return graph.compile()
