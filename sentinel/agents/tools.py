"""Registry-backed data-science tools for the agent harness.

Tools return strings for expected errors. Runtime dependencies are closed over
by ``make_tools``; confirmation policy is handled by agent middleware.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from langchain_core.tools import tool

from ..core import automl
from . import monitor as monitor_mod
from .registry import Registry
from .report_writer import write_report
from .state import InterviewConfig
from .training import TrainingRun

_FAMILY = {
    "extra trees regressor": "et",
    "extra trees": "et",
    "light gradient boosting machine": "lightgbm",
    "lightgbm": "lightgbm",
    "random forest regressor": "rf",
    "random forest": "rf",
}


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def _family_of(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    return _FAMILY.get(normalized, normalized.replace(" ", "-"))


def _reevaluate(
    registry: Registry, model_id: str, rul_cap: int, window: int
) -> dict:
    """Re-score a model through its persisted predictor and readings."""
    predict = registry.load_predict(model_id)
    readings = pd.DataFrame(registry.readings(model_id))
    predictions = predict(readings)
    return automl._regression_metrics(readings["RUL"], list(predictions))


def make_tools(
    *,
    train_fn,
    retrain_fn,
    chat_model,
    ticket_dir,
    registry: Registry,
) -> list:
    """Build the registry-backed tool collection."""
    config_path = registry.root.parent / "config.json"

    def _load_config() -> InterviewConfig | None:
        if config_path.exists():
            return InterviewConfig(**json.loads(config_path.read_text()))
        return None

    def _unknown(model_id: str) -> str:
        return (
            f"No model '{model_id}' in the registry. "
            f"Known: {registry.list()}"
        )

    @tool
    def save_config(
        framing: str,
        failure_threshold: int,
        reporting_cadence: str,
        success_metric: str,
        rul_cap: int = 125,
        window: int = 5,
    ) -> str:
        """Persist the run configuration gathered from the user."""
        config = InterviewConfig(
            framing,
            failure_threshold,
            reporting_cadence,
            success_metric,
            rul_cap,
            window,
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config.__dict__, indent=2))
        return f"Saved config: {config.__dict__}"

    @tool
    def train(
        rul_cap: int = 125,
        window: int = 5,
        models: list[str] | None = None,
    ) -> str:
        """Train the model shelf and register its winner."""
        config = _load_config() or InterviewConfig(
            "RUL",
            30,
            "each run",
            "RMSE under 20",
            rul_cap,
            window,
        )
        config.rul_cap, config.window = rul_cap, window
        run: TrainingRun = train_fn(config)
        family = _family_of(run.result.leaderboard.iloc[0]["Model"])
        model_id = registry.register(
            family=family,
            model_path=run.result.model_path,
            metrics=run.result.metrics,
            leaderboard=_records(run.result.leaderboard),
            provenance={
                "source": "train",
                "model_id": family,
                "hyperparameters": {},
                "config": {"rul_cap": rul_cap, "window": window},
                "parent": None,
            },
            test_eval=_records(run.test_eval),
        )
        metrics = run.result.metrics
        return (
            f"Trained and registered {model_id} (now active): "
            f"RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f} "
            f"R2={metrics['r2']:.3f}."
        )

    @tool
    def retrain(
        model_id: str,
        hyperparameters: dict,
        rul_cap: int = 125,
        window: int = 5,
    ) -> str:
        """Retrain one model with explicit hyperparameters."""
        run: TrainingRun = retrain_fn(
            model_id, hyperparameters, rul_cap, window
        )
        registered_id = registry.register(
            family=_family_of(model_id),
            model_path=run.result.model_path,
            metrics=run.result.metrics,
            leaderboard=_records(run.result.leaderboard),
            provenance={
                "source": "retrain",
                "model_id": model_id,
                "hyperparameters": hyperparameters,
                "config": {"rul_cap": rul_cap, "window": window},
                "parent": registry.active(),
            },
            test_eval=_records(run.test_eval),
        )
        metrics = run.result.metrics
        return (
            f"Retrained and registered {registered_id}: "
            f"RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f} "
            f"R2={metrics['r2']:.3f}."
        )

    @tool
    def evaluate(model_id: str) -> str:
        """Report a registered model's held-out metrics."""
        try:
            info = registry.get(model_id)
        except KeyError:
            return _unknown(model_id)
        metrics = info["metrics"]
        return (
            f"{model_id}: RMSE={metrics['rmse']:.2f} "
            f"MAE={metrics['mae']:.2f} R2={metrics['r2']:.3f}."
        )

    @tool
    def compare(
        model_id_a: str,
        model_id_b: str,
        rul_cap: int | None = None,
        window: int | None = None,
    ) -> str:
        """Compare two models on one evaluation configuration."""
        try:
            config_a = registry.provenance(model_id_a)["config"]
            config_b = registry.provenance(model_id_b)["config"]
            metrics_a = registry.get(model_id_a)["metrics"]
            metrics_b = registry.get(model_id_b)["metrics"]
        except KeyError as error:
            missing = str(error).strip("'")
            return _unknown(missing)
        if config_a != config_b:
            if rul_cap is None or window is None:
                active = registry.active()
                if active is None:
                    return (
                        "Cannot compare across different training configs "
                        "without a common rul_cap/window; pass them to compare."
                    )
                common = registry.provenance(active)["config"]
                rul_cap, window = common["rul_cap"], common["window"]
            metrics_a = _reevaluate(
                registry, model_id_a, rul_cap, window
            )
            metrics_b = _reevaluate(
                registry, model_id_b, rul_cap, window
            )
        delta = {
            key: round(metrics_b[key] - metrics_a[key], 3)
            for key in ("rmse", "mae", "r2")
        }
        return (
            f"{model_id_a}: RMSE={metrics_a['rmse']:.2f} "
            f"R2={metrics_a['r2']:.3f} | "
            f"{model_id_b}: RMSE={metrics_b['rmse']:.2f} "
            f"R2={metrics_b['r2']:.3f} | "
            f"delta(b-a): RMSE={delta['rmse']} MAE={delta['mae']} "
            f"R2={delta['r2']}."
        )

    @tool
    def inspect(what: str) -> str:
        """Inspect the system: the model list, the leaderboard, or a model's provenance.

        - "registry" / "models" / "list": the active model and the registered ids.
        - "leaderboard": the ranked model-comparison from the active model's training
          run. Each row has a retrainable PyCaret `id`, the friendly `Model` name, and
          CV metrics, so runner-up models are visible - e.g. to retrain the second-best,
          read row index 1 and retrain its `id`.
        - "<model_id>": that model's provenance JSON.
        """
        if what in ("registry", "models", "list"):
            return (
                f"active={registry.active()} models={registry.list()}"
            )
        if what in ("leaderboard", "models_compared", "ranking"):
            active = registry.active()
            if active is None:
                return "No active model yet; train one first to get a leaderboard."
            leaderboard = registry.get(active)["metrics"].get("leaderboard", [])
            if not leaderboard:
                return "No leaderboard was recorded for the active model."
            return json.dumps(leaderboard)
        try:
            return json.dumps(registry.provenance(what))
        except KeyError:
            return _unknown(what)

    @tool
    def promote(model_id: str) -> str:
        """Set a registered model as active."""
        try:
            registry.set_active(model_id)
        except KeyError:
            return _unknown(model_id)
        return f"Promoted {model_id}; it is now the active model."

    @tool
    def delete(model_id: str) -> str:
        """Remove an inactive registered candidate."""
        try:
            registry.remove(model_id)
        except KeyError:
            return _unknown(model_id)
        except ValueError as error:
            return str(error)
        return f"Deleted {model_id}."

    @tool
    def write_report_tool() -> str:
        """Write a grounded report over the active model."""
        active = registry.active()
        if active is None:
            return "No active model to report on; train one first."
        info = registry.get(active)
        result = automl.TrainResult(
            leaderboard=pd.DataFrame(info["metrics"]["leaderboard"]),
            best_model=None,
            metrics={
                key: info["metrics"][key]
                for key in ("rmse", "mae", "r2")
            },
            model_path=None,
            metrics_path=None,
        )
        return write_report(
            result,
            chat_model,
            _load_config(),
            best_model_name=active,
        )

    write_report_tool.name = "write_report"

    @tool
    def run_monitor() -> str:
        """Monitor active-model readings and file alert tickets."""
        active = registry.active()
        if active is None:
            return "No active model to monitor; train one first."
        config = _load_config()
        threshold = config.failure_threshold if config else 30
        predict = registry.load_predict(active)
        readings = pd.DataFrame(registry.readings(active))
        events = monitor_mod.run_monitor(
            readings, predict, threshold, Path(ticket_dir)
        )
        alerts = [
            event for event in events if event["decision"] == "alert"
        ]
        return (
            f"Monitored {len(readings)} readings: {len(events)} flagged, "
            f"{len(alerts)} alerts filed."
        )

    return [
        save_config,
        train,
        retrain,
        evaluate,
        compare,
        inspect,
        promote,
        delete,
        write_report_tool,
        run_monitor,
    ]
