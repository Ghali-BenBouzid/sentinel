# Deferred feature: real-environment simulator

Status: idea captured for a later design phase.

Do not implement this directly from this note.

Run a dedicated design pass first.

## Intent

Make Sentinel demonstrable as a system connected to a realistic fleet of machines rather than as a chat interface that occasionally replays a static test set.

An independently runnable simulator process represents multiple machines and their sensors.

It sends a continuous stream of readings to a public ingestion API exposed by Sentinel.

The application includes an integration panel that clearly identifies the source as a simulated real environment connected through the same API an external system would use.

The demo should exercise the complete operational loop from incoming telemetry through monitoring, detection, agent interpretation, and visible action or failure reporting.

## Proposed experience

1. The operator opens Sentinel's integration panel.
2. The panel explains that the source is an external machine-fleet simulator using Sentinel's ingestion API.
3. The operator starts or connects the simulator.
4. The UI shows connection health, machines, sensors, event throughput, and current simulation scenario.
5. Readings begin arriving continuously.
6. Sentinel detects operational and data-quality conditions, explains them, and surfaces resulting alerts or actions.
7. The operator can stop, reset, or select a scenario for a repeatable demonstration.

The simulator must remain independently runnable so it proves the API boundary rather than becoming hidden frontend mock data.

## Scenarios to cover

- Normal operation across several machines with distinct sensor profiles.
- Gradual degradation leading toward a maintenance threshold.
- Abrupt equipment anomaly or failure signature.
- Covariate or operating-condition drift.
- Concept or degradation-pattern drift where practical.
- Expected sensor stops reporting entirely.
- Intermittent missing readings.
- Delayed, duplicated, malformed, stale, or out-of-order telemetry.
- Implausible values and stuck sensors.
- A new or unknown sensor appears.
- Machine disconnect and reconnect.
- Burst traffic or a temporary ingestion outage.
- Model or agent monitoring failure.
- Tool, ticket, or downstream action failure.

Failures inside Sentinel must be visible as first-class operational events.

The demo must not silently turn an internal failure into an apparently healthy state.

## Data sources

The first version may replay held-out C-MAPSS readings because they already match Sentinel's predictive-maintenance domain.

The design should also support generated or transformed readings so scenarios are not limited to anomalies present in the held-out dataset.

Synthetic scenarios should be deterministic from a seed and carry explicit ground-truth labels for later evaluation.

Never use held-out test outcomes to train or select a model.

Replaying held-out readings for operational demonstration must remain separate from model training and selection.

## Likely system boundary

The eventual design should separate at least four concerns:

- A simulator process that owns virtual machines, clocks, scenario generation, and emission rate.
- A versioned Sentinel telemetry-ingestion API with machine identity, sensor identity, timestamps, and idempotency semantics.
- A monitoring pipeline that validates data quality, applies the active model, detects drift or anomalies, and records outcomes.
- An integration UI that observes and controls the demo without pretending the browser itself is the data source.

The control plane and data plane should be distinct.

Starting a local external process from a browser requires a trusted backend control endpoint or a separately managed simulator service.

That security and deployment boundary must be decided during design rather than hidden behind a button.

## Important design questions

- Does the UI start a simulator managed by Sentinel, or only connect to a simulator the operator started separately?
- Is ingestion HTTP, streaming HTTP, MQTT, another protocol, or an adapter-based combination?
- What is the canonical telemetry envelope and schema-version strategy?
- How are machines provisioned and authenticated?
- What delivery guarantees, ordering rules, and idempotency behavior does ingestion promise?
- Which checks are deterministic data-quality rules and which belong to learned detectors or the agent?
- How does Sentinel distinguish machine degradation, sensor failure, pipeline failure, model failure, and agent failure?
- Which simulated conditions create alerts, maintenance tickets, retraining suggestions, or automatic actions?
- How are ground truth and detection latency recorded so the simulation can become an evaluation harness?
- How does a demo reset all simulator and Sentinel state reproducibly?

## Safety and product constraints

- The integration panel must label simulated data unambiguously.
- Demo controls must not be confused with controls for real industrial machinery.
- Any future real external action remains guarded and auditable.
- Raw telemetry, derived features, model predictions, agent conclusions, and actions need separate provenance.
- Unknown or failed states must display as unknown or failed, never as healthy.
- The simulator should exercise the public integration contract rather than import Sentinel internals.

## Connection to existing work

- `sentinel/agents/monitor.py` contains the current per-reading monitoring behavior.
- `sentinel/agents/tools.py` exposes the current guarded `run_monitor` tool.
- `docs/superpowers/specs/2026-07-06-v2-agentic-ds-design.md` documents the current model registry and stored-readings approach.
- `docs/learning/02-agent-layer.md` explains the original held-out-reading replay monitor.
- `docs/learning/06-sse-vs-websockets.html` discusses future continuous-monitor notification pressure.
- `docs/HANDOFF.md` already identifies genuinely autonomous continuous monitoring and real actions as later milestones.

This feature should become its own spec before implementation because it introduces a durable ingestion contract, a long-running process, operational state, failure semantics, and a new safety boundary.
