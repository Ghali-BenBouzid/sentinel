"""train_one: train a single named model with explicit hyperparameters.

PyCaret is faked so the test is offline: we assert train_one wires setup ->
create_model(model_id, **hp) -> finalize -> predict -> save correctly and
returns a one-row leaderboard with held-out metrics.
"""
from __future__ import annotations

import pandas as pd


def test_train_one_trains_named_model_with_hyperparameters(tmp_path, monkeypatch):
    import sentinel.core.automl as A

    calls = {}

    def fake_setup(**kw):
        calls["setup"] = kw

    def fake_create_model(mid, **kw):
        calls["create_model"] = (mid, kw)
        return "FITTED"

    def fake_finalize_model(model):
        calls["finalize"] = model
        return "FINAL"

    def fake_predict_model(model, data):
        out = data.copy()
        out["prediction_label"] = [40.0] * len(data)
        return out

    def fake_save_model(model, path):
        calls["save"] = path
        (tmp_path / "retrain_et.pkl").write_bytes(b"x")

    monkeypatch.setattr(A, "setup", fake_setup)
    monkeypatch.setattr(A, "create_model", fake_create_model)
    monkeypatch.setattr(A, "finalize_model", fake_finalize_model)
    monkeypatch.setattr(A, "predict_model", fake_predict_model)
    monkeypatch.setattr(A, "save_model", fake_save_model)

    train_df = pd.DataFrame(
        {
            "unit": [1, 1],
            "cycle": [1, 2],
            "s2": [0.1, 0.2],
            "RUL": [50.0, 49.0],
        }
    )
    test_df = pd.DataFrame(
        {"unit": [2], "cycle": [10], "s2": [0.3], "RUL": [45.0]}
    )

    stages = []
    result = A.train_one(
        "et",
        {"n_estimators": 500, "max_depth": 12},
        train_df,
        target="RUL",
        test_df=test_df,
        artifacts_dir=str(tmp_path),
        ignore_features=["unit", "cycle"],
        on_stage=lambda stage, detail="": stages.append(stage),
    )

    assert calls["create_model"] == (
        "et",
        {"n_estimators": 500, "max_depth": 12},
    )
    assert calls["setup"]["ignore_features"] == ["unit", "cycle"]
    assert list(result.leaderboard["Model"]) == ["et"]
    assert set(result.metrics) == {"rmse", "mae", "r2"}
    assert str(result.model_path).endswith("retrain_et.pkl")
    assert "evaluating" in stages and "saving" in stages
