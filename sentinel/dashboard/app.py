"""Streamlit dashboard: watch the M2 agent graph run end to end, live.

This is a throwaway demo view. All Streamlit calls live here; the graph itself is
driven by GraphRunner (sentinel/dashboard/runner.py), which is framework-agnostic.

Run it:

    uv sync --extra dashboard
    uv run streamlit run sentinel/dashboard/app.py

Two interaction modes, split by phase (see the dashboard design doc):
- Interview  -> rerun-driven chat (one rerun per user message).
- Train/monitor -> an in-script poll loop that updates placeholders in place,
  so the long live training run does not fight Streamlit's rerun model.
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


def _drain_into_chat(runner: GraphRunner) -> None:
    """Move any new prompt/notify events into the persistent chat transcript."""
    for ev in runner.poll():
        if ev.kind == "prompt":
            st.session_state.chat.append(("assistant", ev.payload))
        elif ev.kind == "notify":
            st.session_state.chat.append(("assistant", f"_{ev.payload}_"))
        elif ev.kind == "error":
            st.session_state.chat.append(("assistant", f"**Error:** {ev.payload}"))


def _wait_for(runner: GraphRunner, pred, timeout: float = 30.0) -> bool:
    """Spin (draining events) until pred() is true or timeout. UI-thread only."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        _drain_into_chat(runner)
        if pred():
            return True
        time.sleep(0.2)
    return False


def _start_run() -> None:
    """Construct providers + GraphRunner and start the worker thread."""
    runner = GraphRunner(
        graph=build_graph(),
        provider_smart=get_provider("smart"),
        provider_cheap=get_provider("cheap"),
        train_fn=run_training,
        ticket_dir="artifacts/tickets",
    )
    runner.start()
    st.session_state.runner = runner
    st.session_state.chat = []


st.title("🛰️ Sentinel")
st.caption("Predictive-maintenance agent - interview → train → report → monitor, live.")

# --- start screen ---------------------------------------------------------
if "runner" not in st.session_state:
    provider = get_settings().sentinel_llm_provider
    st.write(
        "This runs the real agent graph against NASA C-MAPSS FD001. "
        f"LLM provider: **{provider}**. Training runs PyCaret live, so it takes a few minutes."
    )
    if st.button("Start", type="primary"):
        try:
            _start_run()
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - show config/key errors in the UI, not the console
            st.error(f"Could not start: {type(exc).__name__}: {exc}")
    st.stop()

runner: GraphRunner = st.session_state.runner
_drain_into_chat(runner)

# --- 1. interview chat ----------------------------------------------------
st.subheader("1 · Interview")
for role, msg in st.session_state.chat:
    st.chat_message(role).write(msg)

if runner.error is not None:
    st.error(f"The run failed unexpectedly: {runner.error}")
    st.stop()

if runner.pending_prompt() is not None:
    reply = st.chat_input("Your answer")
    if reply:
        st.session_state.chat.append(("user", reply))
        runner.answer(reply)
        st.rerun()
    st.stop()  # wait for the user; nothing further to render this run

# If the interview is still resolving (LLM thinking) and training hasn't begun,
# spin briefly for the next prompt so the chat feels responsive.
if not runner.saw("training_started") and not runner.done:
    with st.spinner("Thinking..."):
        if _wait_for(
            runner,
            lambda: runner.pending_prompt() is not None or runner.saw("training_started") or runner.done,
        ):
            st.rerun()

# --- 2. training ----------------------------------------------------------
if runner.saw("training_started"):
    st.subheader("2 · Training")
    if not runner.saw("training_finished") and not runner.done:
        started = time.time()
        with st.status("Comparing model families with PyCaret...", expanded=True) as status:
            while not runner.saw("training_finished") and not runner.done:
                for ev in runner.poll():
                    if ev.kind == "notify":
                        st.write(ev.payload)
                status.update(label=f"Training... ({int(time.time() - started)}s elapsed)")
                time.sleep(0.5)
            status.update(label="Training complete", state="complete")
        st.rerun()

# From here on we need the finished graph state.
if not runner.done:
    st.stop()

final = runner.final_state()

# A training failure is reported by the graph itself; render it and stop.
if final.get("event") == "failed_reported":
    st.subheader("2 · Training")
    st.error("Training did not complete.")
    st.subheader("3 · Report")
    st.write(final.get("report", "(no report)"))
    st.stop()

# Successful run: show the leaderboard + headline metrics.
run = final["train_run"]
st.dataframe(run.result.leaderboard, width="stretch")
m = run.result.metrics
c1, c2, c3 = st.columns(3)
c1.metric("RMSE", f"{m['rmse']:.1f}")
c2.metric("MAE", f"{m['mae']:.1f}")
c3.metric("R²", f"{m['r2']:.2f}")

# --- 3. report ------------------------------------------------------------
st.subheader("3 · Report")
st.write(final.get("report", "(no report)"))

# --- 4. monitor + tickets -------------------------------------------------
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
