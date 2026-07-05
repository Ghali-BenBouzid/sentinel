"""Report writer sub-agent.

A worked example of a well-engineered, grounding-constrained data-explaining
prompt (see `docs/learning/02-agent-layer.md` for the why). `write_report` turns
a finished AutoML run into a plain-language report using only the `Provider`
seam, and is hard-constrained against fabricating or transforming numbers - the
failure that once made a weak model "explain" RMSE by taking its square root.

The prompt follows the TIDD-EC framework (Task, Instructions, Do, Don't,
Examples, Context): a system message carries the role + the Do/Don't rules, and
a user message carries the Context (the domain glossary), the grounded data, and
the task. It reads its domain knowledge from `domain_context.py`, so adding a new
metric or dataset there flows into the report with no change here.

`report_writer_node` is the thin LangGraph wrapper - it pulls the `TrainResult`
and the cheap provider out of the graph, calls `write_report`, and records the
text back into state. The wiring does not depend on how the prompt is built.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..core.automl import TrainResult
from ..llm.provider import Provider
from . import domain_context
from .state import AgentState, InterviewConfig, append_log

# System message: role + the hard Do/Don't rules (TIDD-EC). These are what keep a
# weak model honest - every reported number must be reproduced verbatim, never
# derived. The specific bans (square root, relabelling) target real observed
# failures, not hypotheticals.
_SYSTEM_PROMPT = (
    "You are a predictive-maintenance analyst who writes short, honest, "
    "plain-language reports about model-training runs for a non-expert reader.\n\n"
    "You reason ONLY from the numbers and glossary you are given. Follow these "
    "rules without exception.\n\n"
    "DO:\n"
    "- State only numeric values that appear verbatim in the METRICS block you are given.\n"
    "- Use the exact metric names and units from the GLOSSARY (RMSE is Root Mean Squared "
    "Error, measured in cycles; MAE is Mean Absolute Error, in cycles; R2 is unitless).\n"
    "- When you want to say how far off the model is on average, quote the provided MAE.\n"
    "- Make clear these scores are on a HELD-OUT TEST SET, and explain why in ONE brief clause: "
    "the model never saw this data during training, so the scores reflect real-world performance "
    "on new engines rather than memorised training data. You may note the model was chosen by "
    "cross-validation on the training data, but do NOT cite any cross-validation numbers (none are "
    "given to you).\n"
    "- Explain the result in practical terms (cycles of remaining engine life) using only "
    "the glossary and the provided metrics.\n\n"
    "DO NOT:\n"
    "- Do NOT compute, derive, transform, recompute, or infer any new number. Never take a "
    "square root, square, ratio, sum, average, or percentage of a provided metric, and never "
    "invent counts or 'approximately X' figures that are not in the METRICS block.\n"
    "- Do NOT treat RMSE as a value to convert into an error - RMSE is ALREADY the error "
    "magnitude in cycles. Do not take its square root or otherwise transform it.\n"
    "- Do NOT describe any metric (RMSE, MAE, or R2) as a prediction of remaining life or of when "
    "the equipment will fail. Metrics measure how ACCURATE the model is, not how long an engine "
    "will last. Never write anything like 'the model predicts it will fail in 17 cycles' from a "
    "metric - that confuses an error measure with a forecast.\n"
    "- Do NOT rename or relabel metrics (RMSE is Root Mean Squared Error, never 'Mean Squared "
    "Error').\n"
    "- Do NOT compute how far a metric is above or below a target or threshold (e.g. never write "
    "'2.91 cycles under the target' or 'N above the goal'). Just state plainly whether the target "
    "is met or not - the difference is a calculation, and calculations are forbidden.\n"
    "- If a number you would like to cite is not in the METRICS block, omit that claim rather "
    "than inventing or calculating it."
)


# Words that fix the direction of a free-text success target ("RMSE under 20").
_BELOW_WORDS = ("under", "below", "less than", "lower than", "at most", "no more than", "within", "<")
_ABOVE_WORDS = ("above", "over", "greater than", "higher than", "at least", "no less than", ">")
_METRIC_WORDS = {"rmse": "rmse", "mae": "mae", "r-squared": "r2", "r squared": "r2", "r2": "r2"}


def _success_verdict(success_metric: str | None, metrics: dict[str, float]) -> bool | None:
    """Decide met / not-met in CODE, so the weak LLM never does the comparison.

    A weak model gets numeric comparisons wrong (it once called RMSE 17.09 "not
    under 20"), so we do it here and hand it the conclusion to phrase. Returns
    True/False, or None when the free-text target can't be parsed confidently -
    then the report states target + achieved value without asserting a verdict.

    ponytail: regex for the common "METRIC under/over N" phrasing; None-fallback
    otherwise - never assert a verdict we are not sure of.
    """
    if not success_metric:
        return None
    t = success_metric.lower()
    key = next((v for k, v in _METRIC_WORDS.items() if k in t), None)
    if key is None or key not in metrics:
        return None
    # Strip the metric name before reading the target number, so the "2" in "R2"
    # is not mistaken for the threshold.
    cleaned = t
    for word in _METRIC_WORDS:
        cleaned = cleaned.replace(word, " ")
    num = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if num is None:
        return None
    target, value = float(num.group(1)), float(metrics[key])
    below = any(w in t for w in _BELOW_WORDS)
    above = any(w in t for w in _ABOVE_WORDS)
    if below and not above:
        return value <= target
    if above and not below:
        return value >= target
    # No explicit direction word: fall back to the metric's nature - error metrics
    # (RMSE/MAE) are upper bounds, R2 is a lower bound.
    if key in ("rmse", "mae"):
        return value <= target
    return value >= target  # r2


def write_report(
    result: TrainResult,
    provider: Provider,
    config: InterviewConfig | None = None,
    best_model_name: str | None = None,
) -> str:
    """Turn a finished AutoML run into a short, grounded plain-language report.

    Inputs:
      - `result`: the M1 `TrainResult` - `.leaderboard` (a ranked DataFrame),
        `.best_model` (the fitted estimator), and `.metrics` (held-out
        ``rmse``/``mae``/``r2`` on the FD001 test set).
      - `provider`: the LLM seam; only `.complete(messages)` is used.
      - `config`: optional interview context, so the report can speak to the
        user's stated framing and success metric.

    Output: the report text the LLM returns (a few short paragraphs). No side
    effects - the caller decides what to do with the string. The prompt is
    hard-constrained so every number in the report traces back to `result.metrics`.
    """
    m = result.metrics
    # The best-model name may arrive as a serializable string (from checkpointed
    # state, where the estimator itself can't survive); fall back to the live
    # estimator's class name for direct/CLI callers that still pass a TrainResult.
    best_model_name = best_model_name or type(result.best_model).__name__
    # Leaderboard is reduced to model *names* only: the METRICS block is the single
    # numeric source, so there is no second table of numbers for the model to
    # (mis)transcribe. Every number the report may cite lives in one place.
    lb = result.leaderboard
    if "Model" in lb.columns:
        ranked = ", ".join(str(name) for name in lb["Model"].head(5))
    else:
        ranked = ", ".join(str(name) for name in lb.index[:5])

    goal = ""
    success_check = "SUCCESS CHECK: no explicit success target was given, so do not include a verdict."
    if config is not None:
        goal = (
            "\n<user_goal>\n"
            f"How the user framed the problem: {config.framing}\n"
            f"How the user defined success: {config.success_metric}\n"
            "</user_goal>\n"
        )
        verdict = _success_verdict(config.success_metric, m)
        if verdict is None:
            success_check = (
                "SUCCESS CHECK: state the user's target and the model's achieved value plainly; "
                "do not assert whether it is met (the target has no clear numeric rule to check)."
            )
        else:
            success_check = (
                "SUCCESS CHECK (already decided in code - state THIS verdict, do NOT re-compare the "
                f"numbers yourself): the model {'MEETS' if verdict else 'DOES NOT MEET'} the user's "
                "stated target."
            )

    user_prompt = (
        "<glossary>\n"
        f"{domain_context.glossary()}\n"
        "</glossary>\n\n"
        "<run>\n"
        f"Best model: {best_model_name}\n"
        "METRICS - the best model's scores on the HELD-OUT TEST SET (FD001 engines the "
        "model never saw during training); these are the ONLY numbers you may cite, verbatim:\n"
        f"- RMSE = {m['rmse']:.2f} cycles\n"
        f"- MAE = {m['mae']:.2f} cycles\n"
        f"- R2 = {m['r2']:.3f}\n"
        "</run>\n\n"
        "<models_compared>\n"
        "(ranked by cross-validation on the training data; best first - names only, no numbers)\n"
        f"{ranked}\n"
        "</models_compared>\n"
        f"{goal}"
        f"<success_check>\n{success_check}\n</success_check>\n"
        "TASK: Write a short, clean report for a non-expert - about three tight paragraphs of "
        "flowing prose. No preamble (do not open with 'Here is a report' and do not narrate your "
        "own process, e.g. 'I'll focus on...'), no bullet points, and never announce 'the following "
        "metrics' and then omit them - weave every number directly into a sentence. Follow this "
        "shape, filling each <...> slot ONLY from the METRICS block (verbatim, never calculated):\n"
        "1. The winning model, and that it was selected by cross-validation from several model "
        "families (you may name the winner; do NOT paste the list of model names, and never call "
        "the winning model one of the 'other' models). Then state these scores are on a held-out "
        "test set and, in one clause, why that matters - the model never saw that data in training.\n"
        "2. Accuracy in practical terms: on average it is off by <MAE> cycles (MAE), and its typical "
        "error size is <RMSE> cycles (RMSE). Add one sentence on what that means for planning "
        "maintenance.\n"
        "3. The fit and the verdict: state R2 = <R2>, and in ONE plain sentence explain that R2 runs "
        "from 1 (a perfect fit) down to 0 (no better than always guessing the average), placing this "
        "value in that range. Then give the success verdict using the SUCCESS_CHECK block above "
        "word-for-word - do NOT compare the numbers yourself and do NOT compute any distance."
    )
    return provider.complete(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )


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

    ts = state["train_state"]
    # Rebuild only what `write_report` reads from serializable state: metrics dict,
    # a leaderboard frame (names only), and the best-model NAME (the estimator
    # itself did not cross the checkpoint boundary).
    result = TrainResult(
        leaderboard=pd.DataFrame(ts["leaderboard"]),
        best_model=None,
        metrics=ts["metrics"],
        model_path=Path(ts["model_path"]),
        metrics_path=Path(ts["model_path"]),  # unused by write_report
    )
    report = write_report(result, cheap, state.get("config"), best_model_name=ts["best_model_name"])
    return {
        "report": report,
        "event": "report_ready",
        "log": append_log(state, "report_writer: wrote run report"),
    }
