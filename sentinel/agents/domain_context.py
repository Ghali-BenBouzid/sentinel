"""Domain context / glossary - the single source of plain-language truth.

The data-interacting agents (report writer, interviewer) reason about datasets,
metrics, models, and techniques. A weaker LLM will *invent* interpretations if the
prompt doesn't explain what these things mean - that's exactly how the report
writer once "explained" RMSE by taking its square root. This module is the fix:
one place that explains each concept in plain language, rendered into the prompts
as grounding context.

Extending it is deliberately a one-liner: to support a new dataset, metric, model,
or technique, add one entry to the matching dict below - nothing else changes.
The render helpers and the prompts pick it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    """Plain-language explanation of one supported dataset."""

    name: str
    description: str  # 2-4 lines: what it is, target, and the units, for a non-expert
    target: str
    units: str


@dataclass(frozen=True)
class MetricInfo:
    """Plain-language, honest explanation of one evaluation metric."""

    key: str  # the key as it appears in a TrainResult.metrics dict, e.g. "rmse"
    name: str  # the correct full name
    units: str
    definition: str  # one line
    interpretation: str  # how to read it HONESTLY (units, what NOT to do with it)


# --- DATASETS ------------------------------------------------------------
# Add a dataset = add one DatasetInfo entry keyed by its short id.
DATASETS: dict[str, DatasetInfo] = {
    "fd001": DatasetInfo(
        name="NASA C-MAPSS FD001 turbofan engines",
        description=(
            "Run-to-failure logs from a fleet of simulated turbofan jet engines. Each engine "
            "records 21 sensors on every operating 'cycle' (one run/flight cycle) until it fails. "
            "We predict Remaining Useful Life (RUL): how many cycles an engine has left before "
            "failure. The RUL target is capped at 125 cycles (rul_cap) - a healthy engine's exact "
            "remaining life is not predictable from sensors, so anything further out than 125 "
            "cycles is simply labelled 125. The model reads rolling-window features (the recent "
            "mean, spread, and trend of each sensor over the last few cycles), so it sees how "
            "readings are drifting rather than a single instant."
        ),
        target="Remaining Useful Life (RUL)",
        units="cycles",
    ),
}

# --- METRICS -------------------------------------------------------------
# Add a metric = add one MetricInfo entry keyed by its metrics-dict key.
METRICS: dict[str, MetricInfo] = {
    "rmse": MetricInfo(
        key="rmse",
        name="Root Mean Squared Error (RMSE)",
        units="cycles",
        definition="the typical size of the model's prediction error, in the target's own units.",
        interpretation=(
            "This IS the error magnitude, already in cycles - do not derive, transform, "
            "recompute, or take the square root of it (it is already a 'root' value). It weighs "
            "large misses more heavily than MAE, so it is usually a bit larger than the MAE."
        ),
    ),
    "mae": MetricInfo(
        key="mae",
        name="Mean Absolute Error (MAE)",
        units="cycles",
        definition="the average absolute gap between predicted and actual RUL.",
        interpretation=(
            "This is the honest 'on average, off by about X cycles' number. When you want to say "
            "how far off the model is on average, quote the MAE - do not invent a different value."
        ),
    ),
    "r2": MetricInfo(
        key="r2",
        name="R-squared, the coefficient of determination (R2)",
        units="unitless, on a 0-to-1 scale",
        definition="the share of the variation in RUL that the model explains.",
        interpretation=(
            "Closer to 1 is better (1.0 would be perfect); around 0 means no better than always "
            "guessing the average. It is a proportion, not a count of cycles."
        ),
    ),
}

# --- MODELS --------------------------------------------------------------
# Placeholder for future entries: model-family id -> one plain-language line.
# Add a model = add one entry, e.g. "et": "Extra Trees - an ensemble of ...".
MODELS: dict[str, str] = {}

# --- TECHNIQUES ----------------------------------------------------------
# Placeholder for future entries: technique id -> one plain-language line.
# Add a technique = add one entry, e.g. "rolling_window": "Rolling-window features ...".
TECHNIQUES: dict[str, str] = {}


def render_dataset(dataset: str = "fd001") -> str:
    """Render one dataset's explanation as a plain-language block."""
    d = DATASETS[dataset]
    return f"{d.name}: {d.description}\nPrediction target: {d.target} (measured in {d.units})."


def render_metrics(keys: tuple[str, ...] | None = None) -> str:
    """Render the given metrics' honest explanations (all known metrics if None)."""
    keys = keys or tuple(METRICS)
    lines = []
    for key in keys:
        m = METRICS[key]
        lines.append(f"- {m.name} - units: {m.units}. {m.definition} {m.interpretation}")
    return "\n".join(lines)


def render_models(keys: tuple[str, ...] | None = None) -> str:
    """Render known model explanations (empty string until MODELS is populated)."""
    keys = keys or tuple(MODELS)
    return "\n".join(f"- {MODELS[k]}" for k in keys)


def render_techniques(keys: tuple[str, ...] | None = None) -> str:
    """Render known technique explanations (empty until TECHNIQUES is populated)."""
    keys = keys or tuple(TECHNIQUES)
    return "\n".join(f"- {TECHNIQUES[k]}" for k in keys)


def glossary(dataset: str = "fd001", metric_keys: tuple[str, ...] | None = None) -> str:
    """Assemble the plain-language grounding block injected into agent prompts.

    Includes the dataset explanation and the metric glossary; models/techniques
    are appended only once those dicts have entries, so the block grows itself.
    """
    parts = ["DATASET:", render_dataset(dataset), "", "METRICS:", render_metrics(metric_keys)]
    if MODELS:
        parts += ["", "MODELS:", render_models()]
    if TECHNIQUES:
        parts += ["", "TECHNIQUES:", render_techniques()]
    return "\n".join(parts)
