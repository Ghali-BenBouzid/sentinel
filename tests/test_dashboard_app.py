"""Streamlit-runtime tests for the dashboard view (app.py).

These run the REAL app script via Streamlit's AppTest, with the LLM and training
faked (no key, no PyCaret). They lock in the three UI-bug fixes:

- the run advances on its own after each answer (no "type again to load the next
  question" race),
- the report + monitor actually render once training finishes (no dead st.stop),
- a fresh session (browser refresh) reconnects to the running graph instead of
  restarting from the Start screen.

Skipped entirely when streamlit is not installed (the `dashboard` extra), so CI
without that extra stays green.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import pytest

pytest.importorskip("streamlit")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import sentinel.agents.training as training_mod  # noqa: E402
import sentinel.llm.provider as provider_mod  # noqa: E402
from sentinel.agents.training import TrainingRun  # noqa: E402
from sentinel.core.automl import TrainResult  # noqa: E402

APP = "sentinel/dashboard/app.py"


class QueueProvider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, messages, **kwargs):
        i = min(self.calls, len(self.replies) - 1)
        self.calls += 1
        return self.replies[i]


class FakeProvider:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, messages, **kwargs):
        return self.reply


def _turn(c, r, v=None):
    return json.dumps({"classification": c, "reply": r, "value": v, "deduced": []})


def _gate(b):
    return json.dumps({"all_defaults": b})


def _fake_run(*a, **k):
    lb = pd.DataFrame({"Model": ["Extra Trees"], "MAE": [11.9], "RMSE": [17.1], "R2": [0.82]})
    result = TrainResult(
        leaderboard=lb,
        best_model=object(),
        metrics={"rmse": 17.1, "mae": 11.9, "r2": 0.82},
        model_path="artifacts/rul_model.pkl",
        metrics_path="artifacts/metrics.json",
    )
    test_eval = pd.DataFrame({"unit": [1, 2], "RUL": [10, 90]})
    return TrainingRun(result=result, test_eval=test_eval, predict=lambda f: list(f["RUL"]))


def _install_fakes(smart_replies, report="Report: model is good."):
    """Patch the source factories app.py imports (done before AppTest runs it)."""
    smart = QueueProvider(smart_replies)
    provider_mod.get_provider = lambda tier: smart if tier == "smart" else FakeProvider(report)
    training_mod.run_training = _fake_run


@pytest.fixture(autouse=True)
def _clear_resource_cache():
    # The runner lives in st.cache_resource (so a refresh reconnects); clear it
    # between tests so runs don't leak into each other.
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _click_start(at):
    for b in at.button:
        if "Start" in b.label:
            b.click().run()
            return
    raise AssertionError("no Start button on the start screen")


def _body(at):
    return " ".join(m.value for m in at.markdown)


def test_run_self_advances_to_report_and_monitor():
    # Bugs 2 & 3: after each answer the app must advance on its own (no extra
    # nudge), and once training finishes the report + monitor must render.
    _install_fakes(
        [_gate(False), _turn("CLEAR", "ok", "turbofan RUL"), _turn("CLEAR", "ok", 30),
         _turn("CLEAR", "ok", "each run"), _turn("CLEAR", "ok", "RMSE < 20")]
    )
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    _click_start(at)
    assert not at.exception, at.exception

    answers = iter(["no, tune it", "turbofan RUL", "30", "each run", "RMSE < 20"])
    # Each answer is one user submit; the app must self-advance to the next
    # question (or into training) within that run - no external re-run loop.
    for _ in range(5):
        assert at.chat_input, "expected a chat input awaiting the next question"
        at.chat_input[0].set_value(next(answers)).run()
        assert not at.exception, at.exception
        if "tickets filed" in _body(at):
            break

    body = _body(at)
    assert "Report: model is good." in body, f"report not rendered:\n{body}"
    assert "tickets filed" in body, f"monitor summary not rendered:\n{body}"


def test_refresh_reconnects_without_restarting():
    # Bug 1: a brand-new session (browser refresh) must reconnect to the running
    # graph, not drop back to the Start screen.
    _install_fakes([_gate(True)])
    at1 = AppTest.from_file(APP, default_timeout=90)
    at1.run()
    _click_start(at1)

    time.sleep(0.5)  # let the (fake, instant) run make progress
    at2 = AppTest.from_file(APP, default_timeout=90)
    at2.run()
    assert not at2.exception, at2.exception
    assert not any("Start" in b.label for b in at2.button), "refresh restarted from the Start screen"
