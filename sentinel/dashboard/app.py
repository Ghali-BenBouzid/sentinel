"""Streamlit dashboard: watch the M2 agent graph run end to end, live.

This is a throwaway demo view. All Streamlit calls live here; the graph itself is
driven by GraphRunner (sentinel/dashboard/runner.py), which is framework-agnostic.

Run it:

    uv sync --extra dashboard
    uv run streamlit run sentinel/dashboard/app.py

Design (learned from three UI bugs in the first cut):

- **The page is a pure function of the runner's transcript.** Every rerun rebuilds
  the whole view from `runner.history()` + a few state flags. Nothing important
  lives in `st.session_state`, so a browser refresh rebuilds the exact same page.
- **The runner is held in `st.cache_resource`**, a process-global singleton that
  survives reruns *and* fresh sessions (refreshes) - the background graph thread
  keeps running and the refreshed page reconnects to it.
  (ponytail: global singleton = single-user demo; fine here, not for multi-user.)
- **One rerun point drives progress:** while the run is active and not waiting on
  the user, `sleep + st.rerun()` polls the worker. When a prompt is pending we
  show the chat box and wait; when the run is done we render and settle.
  (ponytail: a 0.5s busy-poll, fine for one viewer; a push channel if it scaled.)
"""

from __future__ import annotations

import time

import streamlit as st

from sentinel.agents.graph import build_graph
from sentinel.agents.training import run_training
from sentinel.config import get_settings
from sentinel.dashboard.runner import GraphRunner
from sentinel.llm.provider import get_provider

st.set_page_config(page_title="Sentinel", page_icon="🛰️", layout="centered")


@st.cache_resource
def _holder() -> dict:
    """Process-global home for the live GraphRunner (survives reruns + refreshes)."""
    return {}


def _build_runner() -> GraphRunner:
    return GraphRunner(
        graph=build_graph(),
        provider_smart=get_provider("smart"),
        provider_cheap=get_provider("cheap"),
        train_fn=run_training,
        ticket_dir="artifacts/tickets",
    )


def _reset() -> None:
    _holder().pop("runner", None)


st.title("🛰️ Sentinel")
st.caption("Predictive-maintenance agent - interview → train → report → monitor, live.")

holder = _holder()
runner: GraphRunner | None = holder.get("runner")

# --- start screen ---------------------------------------------------------
if runner is None:
    provider = get_settings().sentinel_llm_provider
    st.write(
        "This runs the real agent graph against NASA C-MAPSS FD001. "
        f"LLM provider: **{provider}**. Training runs PyCaret live, so it takes a few minutes."
    )
    if st.button("Start", type="primary"):
        try:
            r = _build_runner()
            r.start()
            holder["runner"] = r
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - show config/key errors in the UI, not the console
            st.error(f"Could not start: {type(exc).__name__}: {exc}")
    st.stop()

# --- 1. interview transcript (pure function of history) -------------------
st.subheader("1 · Interview")
for ev in runner.history():
    if ev.kind == "prompt":
        st.chat_message("assistant").write(ev.payload)
    elif ev.kind == "notify":
        st.chat_message("assistant").write(f"_{ev.payload}_")
    elif ev.kind == "answer":
        st.chat_message("user").write(ev.payload)

# Unexpected thread failure (not a graph-reported training failure): dead-end.
if runner.error is not None:
    st.error(f"The run failed unexpectedly: {runner.error}")
    st.button("Start over", on_click=_reset)
    st.stop()

# Waiting on the user's answer: show the chat box and wait (worker is blocked).
if runner.pending_prompt() is not None:
    reply = st.chat_input("Your answer")
    if reply:
        runner.answer(reply)
        st.rerun()
    st.stop()

# --- still running: show live status, then poll -------------------------
if not runner.done:
    if runner.saw("training_started"):
        st.subheader("2 · Training")
        elapsed = int(runner.training_elapsed() or 0)
        st.info(f"Comparing model families with PyCaret... ({elapsed}s elapsed)")
    else:
        st.caption("Thinking...")
    time.sleep(0.5)
    st.rerun()

# --- done: render the finished run --------------------------------------
final = runner.final_state()

if final.get("event") == "failed_reported":
    st.subheader("2 · Training")
    st.error("Training did not complete.")
    st.subheader("3 · Report")
    st.write(final.get("report", "(no report)"))
    st.button("Start over", on_click=_reset)
    st.stop()

# 2 · training results
run = final["train_run"]
st.subheader("2 · Training")
st.dataframe(run.result.leaderboard, width="stretch")
m = run.result.metrics
c1, c2, c3 = st.columns(3)
c1.metric("RMSE", f"{m['rmse']:.1f}")
c2.metric("MAE", f"{m['mae']:.1f}")
c3.metric("R²", f"{m['r2']:.2f}")

# 3 · report
st.subheader("3 · Report")
st.write(final.get("report", "(no report)"))

# 4 · monitor + tickets
st.subheader("4 · Monitor")
alerts = final.get("alerts", [])
n_alerts = sum(a["decision"] == "alert" for a in alerts)
st.write(f"{len(alerts)} readings flagged · {n_alerts} maintenance tickets filed.")
for a in alerts:
    icon = "🔴" if a["decision"] == "alert" else "🟠"
    line = f"{icon} unit {a['unit']}: predicted RUL {a['predicted_rul']} → {a['decision']}"
    if a.get("ticket"):
        line += f"  · ticket: `{a['ticket']}`"
    st.write(line)

st.divider()
st.button("Start over", on_click=_reset)
