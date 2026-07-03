"""Report writer sub-agent.

The simplest sub-agent, and the one to rebuild as a learning exercise (see
`docs/learning/02-agent-layer.md`): one function, `write_report`, that turns a
finished AutoML run into a plain-language report using only the `Provider` seam.

`report_writer_node` is the thin LangGraph wrapper around it - it pulls the
`TrainResult` and the cheap provider out of the graph, calls `write_report`, and
records the text back into state. The wiring does not depend on how
`write_report` builds its prompt, which is what makes it safe to rewrite.
"""

from __future__ import annotations

from ..core.automl import TrainResult
from ..llm.provider import Provider
from .state import AgentState, InterviewConfig, append_log


def write_report(
    result: TrainResult,
    provider: Provider,
    config: InterviewConfig | None = None,
) -> str:
    """Turn a finished AutoML run into a short plain-language report.

    Inputs:
      - `result`: the M1 `TrainResult` - `.leaderboard` (a ranked DataFrame),
        `.best_model` (the fitted estimator), and `.metrics` (held-out
        ``rmse``/``mae``/``r2`` on the FD001 test set).
      - `provider`: the LLM seam; only `.complete(messages)` is used.
      - `config`: optional interview context, so the report can speak to the
        user's stated framing and success metric.

    Output: the report text the LLM returns (a few short paragraphs). No side
    effects - the caller decides what to do with the string.
    """
    lb = result.leaderboard
    cols = [c for c in ["Model", "MAE", "RMSE", "R2"] if c in lb.columns]
    leaderboard_text = lb[cols].head(5).to_string(index=False) if cols else lb.head(5).to_string()
    m = result.metrics

    context = ""
    if config is not None:
        context = (
            f"The user framed this as: {config.framing}\n"
            f"They defined success as: {config.success_metric}\n"
        )

    prompt = (
        "You are a predictive-maintenance analyst. Write a short, plain-language "
        "report (3-4 short paragraphs, no jargon dumps) on a model-training run "
        "for turbofan Remaining Useful Life (RUL) prediction.\n\n"
        f"{context}\n"
        f"Best model: {type(result.best_model).__name__}\n"
        f"Held-out test metrics: RMSE={m['rmse']:.2f} cycles, "
        f"MAE={m['mae']:.2f} cycles, R2={m['r2']:.3f}\n\n"
        f"Top of the model comparison leaderboard:\n{leaderboard_text}\n\n"
        "Explain which model won and how well it predicts, what the error means "
        "in practical terms (cycles of remaining life), and whether it meets the "
        "stated success bar. Do not invent numbers beyond those given."
    )
    return provider.complete([{"role": "user", "content": prompt}])


def report_writer_node(state: AgentState, config) -> dict:
    """Graph node: write a report for the finished (or failed) run."""
    cheap = config["configurable"]["provider_cheap"]

    if state.get("error"):
        report = (
            "Training did not complete. The run failed with:\n"
            f"{state['error']}\n\nNo model was produced, so there is nothing to monitor."
        )
        return {
            "report": report,
            "event": "failed_reported",
            "log": append_log(state, "report_writer: reported failure"),
        }

    run = state["train_run"]
    report = write_report(run.result, cheap, state.get("config"))
    return {
        "report": report,
        "event": "report_ready",
        "log": append_log(state, "report_writer: wrote run report"),
    }
