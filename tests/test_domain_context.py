"""Tests for the domain-context glossary.

These lock in the two things the glossary exists to guarantee: the metric names
and units are correct (so a weak model can't relabel RMSE as "Mean Squared
Error"), and the honest-interpretation notes carry the guardrails the report
prompt relies on. Also checks the render helpers stay in step with the data, so
extending a dict is genuinely a one-liner.
"""

from __future__ import annotations

from sentinel.agents import domain_context as dc


def test_metric_names_and_units_are_correct():
    assert dc.METRICS["rmse"].name == "Root Mean Squared Error (RMSE)"
    assert dc.METRICS["rmse"].units == "cycles"
    assert dc.METRICS["mae"].name == "Mean Absolute Error (MAE)"
    assert dc.METRICS["mae"].units == "cycles"
    assert "coefficient of determination" in dc.METRICS["r2"].name.lower()


def test_rmse_interpretation_forbids_transformation():
    # The exact guardrail that kills the "square root of RMSE" bug at the source.
    note = dc.METRICS["rmse"].interpretation.lower()
    assert "do not derive" in note or "do not" in note
    assert "square root" in note


def test_metric_interpretations_forbid_prediction_framing():
    # Guards the "the model predicts it will fail in 17.09 cycles" bug: an error
    # metric must never read as a remaining-life forecast.
    for key in ("rmse", "mae", "r2"):
        assert "not a prediction" in dc.METRICS[key].interpretation.lower()


def test_dataset_explains_rul_and_cap_and_units():
    d = dc.DATASETS["fd001"]
    assert d.units == "cycles"
    assert d.target == "Remaining Useful Life (RUL)"
    assert "125" in d.description  # the rul_cap is explained
    assert "cycle" in d.description.lower()


def test_glossary_block_carries_dataset_and_all_metrics():
    block = dc.glossary()
    assert "DATASET:" in block and "METRICS:" in block
    for m in dc.METRICS.values():
        assert m.name in block
    # MODELS/TECHNIQUES are empty placeholders, so their headers stay out for now.
    assert "MODELS:" not in block
    assert "TECHNIQUES:" not in block


def test_glossary_grows_when_a_model_entry_is_added(monkeypatch):
    # Extensibility: adding one entry to a dict surfaces in the rendered block,
    # with no other change. (Patched so the real glossary stays untouched.)
    monkeypatch.setitem(dc.MODELS, "et", "Extra Trees - an ensemble of randomized trees.")
    block = dc.glossary()
    assert "MODELS:" in block
    assert "Extra Trees" in block
