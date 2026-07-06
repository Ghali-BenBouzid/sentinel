"""The DS tools: registry-backed, rail-guarded, string-returning."""
from __future__ import annotations

import pandas as pd


class FakeChat:
    def invoke(self, messages, **kw):
        class _Response:
            content = "Report body."

        return _Response()


def _fake_training_run(tmp_path, rmse=17.1, window=5):
    from sentinel.agents.training import TrainingRun
    from sentinel.core.automl import TrainResult

    model_path = tmp_path / "src.pkl"
    model_path.write_bytes(b"m")
    result = TrainResult(
        leaderboard=pd.DataFrame([{"Model": "Extra Trees", "RMSE": rmse}]),
        best_model=object(),
        metrics={"rmse": rmse, "mae": 12.0, "r2": 0.83},
        model_path=model_path,
        metrics_path=tmp_path / "m.json",
    )
    test_eval = pd.DataFrame(
        [{"unit": 1, "cycle": 200, "RUL": 40.0, "s2": 1.5}]
    )
    return TrainingRun(
        result=result,
        test_eval=test_eval,
        predict=lambda frame: [1.0],
    )


def _tools(tmp_path):
    from sentinel.agents.registry import Registry
    from sentinel.agents.tools import make_tools

    registry = Registry(tmp_path / "models")
    tools = make_tools(
        train_fn=lambda cfg: _fake_training_run(tmp_path),
        retrain_fn=lambda mid, hp, rul_cap, window: _fake_training_run(
            tmp_path, rmse=16.0
        ),
        chat_model=FakeChat(),
        ticket_dir=str(tmp_path / "tickets"),
        registry=registry,
    )
    return {tool.name: tool for tool in tools}, registry


def _invoke(tool, args, autonomy="autonomous"):
    """Call a structured tool including the InjectedState."""
    return tool.invoke({**args, "state": {"autonomy": autonomy}})


def test_confirm_autonomous_proceeds_and_streams(monkeypatch):
    from sentinel.agents import tools as T

    seen = []
    monkeypatch.setattr(T, "get_stream_writer", lambda: seen.append)
    assert T.confirm("promote", "et-v1", "autonomous") is None
    assert seen and seen[0]["type"] == "auto_approved"


def test_confirm_guarded_declined_returns_string(monkeypatch):
    from sentinel.agents import tools as T

    monkeypatch.setattr(T, "interrupt", lambda payload: "no")
    out = T.confirm("promote", "et-v1", "guarded")
    assert isinstance(out, str) and "did not approve" in out


def test_train_registers_winner_and_activates(tmp_path):
    tools, registry = _tools(tmp_path)
    message = _invoke(tools["train"], {"rul_cap": 125, "window": 5})
    assert registry.active() == "et-v1"
    assert "et-v1" in message


def test_retrain_registers_candidate(tmp_path):
    tools, registry = _tools(tmp_path)
    _invoke(tools["train"], {})
    _invoke(
        tools["retrain"],
        {"model_id": "et", "hyperparameters": {"n_estimators": 500}},
    )
    assert set(registry.list()) == {"et-v1", "et-v2"}


def test_promote_moves_active(tmp_path):
    tools, registry = _tools(tmp_path)
    _invoke(tools["train"], {})
    _invoke(tools["retrain"], {"model_id": "et", "hyperparameters": {}})
    _invoke(tools["promote"], {"model_id": "et-v2"})
    assert registry.active() == "et-v2"


def test_delete_active_refused_with_message(tmp_path):
    tools, registry = _tools(tmp_path)
    _invoke(tools["train"], {})
    out = _invoke(tools["delete"], {"model_id": "et-v1"})
    assert "active" in out.lower()
    assert registry.list() == ["et-v1"]


def test_unknown_model_returns_message_not_raise(tmp_path):
    tools, _ = _tools(tmp_path)
    out = _invoke(tools["evaluate"], {"model_id": "ghost"})
    assert "ghost" in out and "registry" in out.lower()


def test_compare_across_configs_reevaluates(tmp_path, monkeypatch):
    tools, registry = _tools(tmp_path)
    _invoke(tools["train"], {"rul_cap": 125, "window": 5})
    _invoke(
        tools["retrain"],
        {
            "model_id": "et",
            "hyperparameters": {},
            "rul_cap": 100,
            "window": 5,
        },
    )
    from sentinel.agents import tools as Tmod

    monkeypatch.setattr(
        Tmod,
        "_reevaluate",
        lambda reg, mid, rul_cap, window: {
            "rmse": 15.0,
            "mae": 10.0,
            "r2": 0.9,
        },
    )
    out = _invoke(
        tools["compare"],
        {
            "model_id_a": "et-v1",
            "model_id_b": "et-v2",
            "rul_cap": 125,
            "window": 5,
        },
    )
    assert "et-v1" in out and "et-v2" in out
