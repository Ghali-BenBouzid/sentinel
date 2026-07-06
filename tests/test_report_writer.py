"""write_report drives a chat model (no Provider seam any more)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class FakeChat:
    """Minimal chat model stand-in: invoke(messages) returns content."""

    def __init__(self, text):
        self.text = text

    def invoke(self, messages, **kw):
        class _Response:
            pass

        response = _Response()
        response.content = self.text
        return response


def test_write_report_uses_chat_model():
    from sentinel.agents.report_writer import write_report
    from sentinel.core.automl import TrainResult

    result = TrainResult(
        leaderboard=pd.DataFrame([{"Model": "Extra Trees", "RMSE": 17.1}]),
        best_model=object(),
        metrics={"rmse": 17.1, "mae": 12.0, "r2": 0.83},
        model_path=Path("x"),
        metrics_path=Path("y"),
    )
    out = write_report(
        result, FakeChat("Report body."), best_model_name="Extra Trees"
    )
    assert out == "Report body."
