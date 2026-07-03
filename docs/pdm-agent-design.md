# Predictive-maintenance agent layer - design

This is the design the Milestone 2 agent layer implements.
It is recorded here so the project carries its own rationale, not just the code.

## The three layers

```
DS core (M1, done)  ->  agent layer (M2, this)  ->  Streamlit dashboard (later)
```

- **DS core** (`sentinel/core/*.py`): deterministic data prep, feature engineering, and AutoML.
  No LLM, independently runnable. See `docs/learning/01-ds-core.md`.
- **Agent layer** (`sentinel/agents/`, `sentinel/llm/`): a LangGraph graph that wraps the DS core -
  interviews the user, supervises training, monitors incoming data, and reports.
- **Dashboard**: out of scope for this milestone.

## The agent layer is a LangGraph graph

- **Orchestrator**: owns state and routing, dispatches sub-agents.
  Woken only by significant events (run finished / failed, best-model-changed, monitor alert) -
  never by raw progress ticks.
- **Interviewer sub-agent**: the only human-facing surface.
  Conversationally extracts structured config (dataset framing, failure threshold, reporting cadence,
  success definition) before any training runs.
  Pattern: the code owns the agenda/checklist, the LLM extracts and phrases free-text answers into that structure.
- **Report writer sub-agent**: turns AutoML runs and significant events (leaderboard, best model, metrics)
  into a plain-language report.
- **Monitor sub-agent**: post-training, steps through incoming (simulated) sensor readings and decides
  report / alert / action.
  MVP action is a mock (writes a ticket file locally).
  No wake-policy system - that is explicitly post-MVP; a simple per-reading threshold check is enough.

## LLM provider seam

`sentinel/llm/provider.py` is a small Python `Protocol` (`complete(messages: list[dict], **kwargs) -> str`)
with two concrete implementations:

- `AnthropicProvider` (primary, `anthropic` SDK).
- `GroqProvider` (free tier, OpenAI-compatible `groq` SDK) for cost-free development/testing.

Which provider is active is a config/env choice (`SENTINEL_LLM_PROVIDER=anthropic|groq`).
This seam is explicitly **not** a routing/orchestration framework - just an interface so the graph never
imports a vendor SDK directly.
A cheap/small model is used for the report writer; a stronger model for the interviewer, where extracting
structure from free text needs more judgment.

## Explicitly out of scope for this milestone

The dashboard, an append-only event stream, real (non-mock) actions, and dataset-agnostic support.
All later milestones or out of MVP scope.
