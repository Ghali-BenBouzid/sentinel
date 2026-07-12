"""Disk-backed model registry: the single source of truth for trained models.

Once retraining exists there are many models (the original winner plus tuned
candidates), so "the current best" / "et-v2" / "the winner" must resolve to
something concrete. This is that something: a small directory per model plus a
manifest naming which one is active.

Layout:
    <models_dir>/
      manifest.json
      <id>/
        model.pkl
        metrics.json
        provenance.json
        readings.json

Only native-JSON data crosses in and out. The model itself is rehydrated on
demand via ``load_predict(id)`` so nothing heavy is handed back to the caller.
"""
from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd


def _pycaret_load_predict(
    pkl_path: Path,
) -> Callable[[pd.DataFrame], list[float]]:
    """Load a persisted PyCaret pipeline and return a prediction function."""
    from pycaret.regression import load_model, predict_model

    model = load_model(str(pkl_path.with_suffix("")))

    def predict(frame: pd.DataFrame) -> list[float]:
        preds = predict_model(model, data=frame)
        col = (
            preds["prediction_label"]
            if "prediction_label" in preds
            else preds.iloc[:, -1]
        )
        return [float(v) for v in col]

    return predict


class Registry:
    """A directory of trained models plus a manifest of which one is active."""

    def __init__(self, models_dir: str | Path) -> None:
        self.root = Path(models_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        self._lock = threading.RLock()
        if not self._manifest_path.exists():
            self._write_manifest({"active": None, "models": []})

    def _read_manifest(self) -> dict:
        return json.loads(self._manifest_path.read_text())

    def _write_manifest(self, manifest: dict) -> None:
        self._manifest_path.write_text(json.dumps(manifest, indent=2))

    def _dir(self, model_id: str) -> Path:
        return self.root / model_id

    def _require(self, model_id: str) -> None:
        if model_id not in self._read_manifest()["models"]:
            raise KeyError(model_id)

    def _next_id(self, family: str) -> str:
        existing = [
            model_id
            for model_id in self._read_manifest()["models"]
            if model_id.rsplit("-v", 1)[0] == family
        ]
        return f"{family}-v{len(existing) + 1}"

    def register(
        self,
        *,
        family: str,
        model_path: str | Path,
        metrics: dict,
        leaderboard: list[dict],
        provenance: dict,
        test_eval: list[dict],
    ) -> str:
        """Copy a saved model in, write its JSON sidecars, and return its id."""
        model_id = self._next_id(family)
        model_dir = self._dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        source = Path(model_path)
        if source.suffix != ".pkl":
            source = source.with_suffix(".pkl")
        shutil.copyfile(source, model_dir / "model.pkl")
        (model_dir / "metrics.json").write_text(
            json.dumps(
                {
                    **{key: float(value) for key, value in metrics.items()},
                    "leaderboard": leaderboard,
                },
                indent=2,
            )
        )
        registered_provenance = {
            **provenance,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (model_dir / "provenance.json").write_text(
            json.dumps(registered_provenance, indent=2)
        )
        (model_dir / "readings.json").write_text(json.dumps(test_eval, indent=2))

        manifest = self._read_manifest()
        manifest["models"].append(model_id)
        if manifest["active"] is None:
            manifest["active"] = model_id
        self._write_manifest(manifest)
        return model_id

    def get(self, model_id: str) -> dict:
        self._require(model_id)
        metrics = json.loads((self._dir(model_id) / "metrics.json").read_text())
        return {
            "id": model_id,
            "metrics": metrics,
            "provenance": self.provenance(model_id),
        }

    def provenance(self, model_id: str) -> dict:
        self._require(model_id)
        return json.loads((self._dir(model_id) / "provenance.json").read_text())

    def readings(self, model_id: str) -> list[dict]:
        self._require(model_id)
        return json.loads((self._dir(model_id) / "readings.json").read_text())

    def list(self) -> list[str]:
        return list(self._read_manifest()["models"])

    def active(self) -> str | None:
        return self._read_manifest()["active"]

    def set_active(self, model_id: str) -> None:
        with self._lock:
            self._require(model_id)
            manifest = self._read_manifest()
            manifest["active"] = model_id
            self._write_manifest(manifest)

    def remove(self, model_id: str) -> None:
        with self._lock:
            self._require(model_id)
            if self._read_manifest()["active"] == model_id:
                raise ValueError(f"{model_id} is the active model; promote another first")
            shutil.rmtree(self._dir(model_id))
            manifest = self._read_manifest()
            manifest["models"].remove(model_id)
            self._write_manifest(manifest)

    def load_predict(
        self, model_id: str
    ) -> Callable[[pd.DataFrame], list[float]]:
        self._require(model_id)
        return _pycaret_load_predict(self._dir(model_id) / "model.pkl")
