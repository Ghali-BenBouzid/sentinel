# V2 Spec 1 - the agentic data scientist (design)

This is the design for Sentinel V2, first slice.
It replaces V1's fixed hub-and-spoke event router with a genuine tool-calling reasoning agent that can be delegated data-science work in natural language.
Last updated: 2026-07-06.

## Goal (and non-goals)

Turn Sentinel from a deterministic pipeline (interview -> train -> report -> monitor) into an agent you can talk to and delegate work to.
The anchor use case, the thing every design decision is measured against:

> "Retrain just Extra Trees with 500 estimators and compare it to the current best. If it wins, promote it."

Stated in one line:
a LangGraph `create_react_agent` becomes the graph's hub, owns a set of data-science tools (train, retrain-with-params, compare, inspect, report, monitor, promote), and decides which to call from the conversation - guarded by confirmation rails with an autonomous override.

Explicit non-goals for this slice:

- **No planner / multi-step autonomous investigation loop.**
  The agent reacts to the user's task turn by turn.
  "Get held-out RMSE under 15 on your own" (the agent iterating unattended toward a target) is a later spec.
  Autonomous *mode* here only means "skip confirmations", not "plan and pursue a goal".
- **No auth / multi-tenant model.**
  Single-user learning backend, same as V1.
  `thread_id` is the session identifier.
- **No new datasets or model families.**
  Same FD001 case study and the same PyCaret model shelf as M1/V1.

## Why replace the orchestrator (not bolt an agent on)

V1's graph is a deterministic pipeline wearing agent clothing.
`route()` dispatches on `state["event"]` through a fixed table; the "sub-agents" are single-purpose LLM calls with no reasoning, tools, or planning.
That fixed router is exactly the scaffolding the V2 vision wants gone.

The load-bearing insight of this design: replacing the orchestrator does **not** throw away V1's tested logic.
The dumb thing that dies is the `route()` event table.
The actual work - `train_and_evaluate`, the report writer, the monitor - survives, repackaged as **tools** the reasoning agent calls.
We lose the router, keep the muscle, gain the brain.

The trade we are accepting with eyes open: an LLM reasoning loop is non-deterministic.
V1 always asked the same fields in the same order; the agent may converse differently each run and occasionally call a tool we did not anticipate.
We manage that with tight typed tool schemas, confirmation rails on anything with external consequence, and offline tests over a scripted fake model.
Predictability for flexibility - the right trade for a delegatable agent.

## Architecture

`create_react_agent` (from `langgraph.prebuilt`, the prebuilt available in the installed langgraph 1.2.7) is the hub.
It is a ReAct loop: the chat model sees the conversation plus the bound tool schemas, emits tool calls, a `ToolNode` runs them, results feed back, and the loop continues until the model answers in prose.

```
        user turn (interrupt) 
              |
              v
   +----------------------+       tool call
   |  create_react_agent  | -----------------> [ ToolNode: run tool, append result ]
   |  (chat model + ReAct)| <-----------------
   +----------------------+       tool result
              |
         prose answer -> back to the user (interrupt for the next turn)
```

What V1 constructs disappear:
`orchestrator_node`, `route()`, the `_ROUTES` table, `route_interview`, and the trainer/report/monitor *nodes* (their logic moves into tools).
The `interviewer.py` `advance()` state machine (~400 lines: gate, deduction, defaults, non-answer bounds) dissolves entirely - a reasoning agent conversing *is* the interview, and it calls `save_config` when it has what it needs.

What stays, untouched or nearly so:
the checkpointer substrate (`SqliteSaver` default, `MemorySaver` in tests), `interrupt()` for human turns, the streaming seam (`get_stream_writer` for progress events), `domain_context.glossary()` grounding, and the DS core (`sentinel/core/*`) plus its callback/stage-event hooks.

## The LLM seam evolves

`sentinel/llm/provider.py` today returns a custom `Provider` (a `complete(messages) -> str` Protocol implemented over the raw `anthropic` / `groq` SDKs).
Tool-calling needs the richer `bind_tools` interface that a `Provider.complete` cannot express, so the seam changes shape:

| | V1 | V2 |
|---|---|---|
| Returns | `Provider` (`complete()->str`) | LangChain `BaseChatModel` |
| Impls | `AnthropicProvider`, `GroqProvider` (raw SDKs) | `ChatAnthropic`, `ChatGroq` (langchain wrappers) |
| Config | `get_settings()` selects | `get_settings()` selects (unchanged) |
| Rule | vendor SDK imports confined to this file | still confined to this file |

New dependencies: `langchain-groq`, `langchain-anthropic` (thin wrappers over the `groq` / `anthropic` SDKs we already depend on).
`get_provider()` becomes `get_chat_model()` returning a `BaseChatModel`, config-selected exactly as today.
The "never import a vendor SDK outside `provider.py`" rule is preserved - the file just builds chat models now instead of Provider objects.
`Provider`, `AnthropicProvider`, `GroqProvider` and the old `get_provider` are deleted.

Model choice: the agent needs a tool-calling-capable model.
Groq's `llama-3.3-70b-versatile` and Anthropic's Claude models both qualify.
Config gains an optional model-name override; the defaults are a tool-capable model per provider.

## State shrinks (and serialization gets safe by construction)

V1's `AgentState` carried `event`, `config`, `train_state`, `report`, `alerts`, `error`, `log`.
Under the agent, the graph's state is the **message history** (`create_react_agent`'s `MessagesState`) plus a disk-backed model registry.
Config, training results, reports, and alerts become artifacts the tools read and write on disk - never graph-state fields.

This makes V1's hardest constraint disappear by construction.
The msgpack checkpoint serializer (no pickle fallback) rejected DataFrames, estimators, and closures; V1 handled it with the `to_state()` discipline.
Now nothing heavy is *ever* in state: the only references that cross the checkpoint boundary are LangChain messages (natively serializable) and string model ids.
The registry is the single source of truth for artifacts, addressed by id.

## Tools (`sentinel/agents/tools.py`)

Each tool is a typed function (`@tool` with a pydantic-validated signature) that wraps existing DS-core / sub-agent logic and reads/writes the registry.
Dependencies (the `train_fn`, the chat model for report/interview grounding, `ticket_dir`, `autonomy`) come from the injected runtime config, not tool arguments - same dependency-injection discipline as V1.

| Tool | Signature (args the model fills) | Does | Rail |
|---|---|---|---|
| `save_config` | `framing, failure_threshold, reporting_cadence, success_metric, rul_cap?, window?` | Persist the run config the agent gathered conversationally. Replaces the interviewer machine. | - |
| `train` | `rul_cap?, window?, models?` | Full comparison; registers every candidate + the winner; returns the leaderboard summary. Wraps `train_and_evaluate`. | confirm |
| `retrain` | `model_id, hyperparameters, rul_cap?, window?` | Parameterized single-model train (NEW DS-core entrypoint); registers a candidate; returns its held-out metrics. | confirm |
| `evaluate` | `model_id` | Held-out metrics for a registered model. | - |
| `compare` | `model_id_a, model_id_b` | Side-by-side metrics + deltas. | - |
| `inspect` | `what` | Read-only: leaderboard / dataset stats / a model's params / the registry listing. | - |
| `promote` | `model_id` | Set the registry's `active` model. | confirm |
| `delete` | `model_id` | Remove a registered candidate. | confirm |
| `write_report` | (none) | Grounded report over the active model's metrics. Wraps `report_writer`, keeps the no-fabrication prompt. | - |
| `run_monitor` | (none) | Step held-out readings through the active model; emit alerts / mock tickets. Wraps `monitor`. | - |

Invalid tool calls (bad model id, malformed hyperparameters) return a descriptive error string the agent can recover from - never a crash that truncates the stream.

## DS-core addition: parameterized retraining

The anchor use case needs something the DS core cannot do: train one named model with specific hyperparameters.
`train_and_evaluate` trains a fixed candidate list and picks the winner.
We add a sibling in `sentinel/core/automl.py`:

```python
def train_one(
    model_id: str,                # "et", "lightgbm", ...
    hyperparameters: dict,        # {"n_estimators": 500, "max_depth": 12}
    train_df, target, test_df,
    artifacts_dir, ignore_features=None, session_id=42, fold=3,
    on_stage=None,
) -> TrainResult:
    ...
```

It runs `setup` + `create_model(model_id, **hyperparameters)`, finalizes, evaluates on the held-out test set, and returns the same `TrainResult` shape as `train_and_evaluate`.
The existing stage-event callback seam is reused so retraining streams progress too.
It is testable offline with a faked PyCaret, same as the existing core helpers.

## Model registry (`sentinel/agents/registry.py`)

The new subsystem the multi-model world requires.
Once retraining exists there are many models (the original winner plus candidates), so "the current best" / "et-v2" / "the winner" need to resolve to something.

Layout on disk:

```
artifacts/models/
  manifest.json                 # { "active": "<id>", "models": ["<id>", ...] }
  <id>/
    model.pkl                   # the PyCaret pipeline
    metrics.json                # held-out rmse/mae/r2 + cv leaderboard row
    provenance.json             # { source, model_id, hyperparameters, config, parent, created_at }
```

Ids are short and human-legible: `et-v1`, `et-v2`, `lightgbm-v1`.
The registry API is small and native-JSON: `register(...) -> id`, `get(id)`, `list()`, `active()`, `set_active(id)`, `remove(id)`, `load_predict(id)`.
Tools are the only writers.
Nothing heavy leaves it: callers get a string id or a small metrics dict; the model itself is rehydrated on demand via `load_predict(id)` (the V1 `training.load_predict` pattern, generalized to the registry).

## Safety rails and autonomy

Two modes, one seam.

**Guarded (default).**
`train`, `retrain`, `promote`, and `delete` stop and ask the human `y/n` before running.
`train`/`retrain` are guarded because they burn real compute (minutes of PyCaret); `promote`/`delete` because they mutate the active system.
The read/additive tools (`evaluate`, `compare`, `inspect`, `write_report`, `run_monitor`) run freely.
Confirmation happens via `interrupt()`, so it works identically over the CLI and the HTTP/SSE surface and survives a checkpoint.

**Autonomous.**
A user-set mode that skips every confirmation - "delegate and walk away."
Even so, each auto-approved action streams a notice (`{"type": "auto_approved", "tool": ...}`) so an unattended run is still auditable.

The seam is a single helper used by every guarded tool:

```python
def require_confirmation(action: str, detail: str, autonomy: str) -> None:
    if autonomy == "autonomous":
        get_stream_writer()({"type": "auto_approved", "tool": action, "detail": detail})
        return
    answer = interrupt({"type": "confirm", "tool": action, "detail": detail})
    if str(answer).strip().lower() not in {"y", "yes"}:
        raise ToolDenied(f"user declined {action}")
```

`autonomy` is injected per session via `config["configurable"]["autonomy"]`, defaulting from `get_settings()` (`SENTINEL_AUTONOMY=guarded|autonomous`, `guarded` default).
A `ToolDenied` returns a clean "user declined" message to the agent, which continues the conversation rather than crashing.

## Surfaces: CLI and HTTP/SSE

Both are in this slice.

**CLI** (`python -m sentinel.agents`): a chat loop.
Read a line, feed it to the agent, stream tool-calls / stage events / the prose answer, and when the agent `interrupt()`s (a question or a confirmation) prompt the user and resume with `Command(resume=...)`.
`--autonomous` sets autonomous mode.

**HTTP/SSE** (`sentinel/api/app.py`): adapt the V1 surface to the agent.
`POST /sessions` (optional `{"autonomy": "..."}`) starts a session and streams until the first interrupt.
`POST /sessions/{id}/resume` (`{"answer": "..."}`) posts one turn - a chat message *or* a `y/n` confirmation - and streams on.
`GET /sessions/{id}` reads a snapshot from the checkpointer.
The SSE event vocabulary extends V1's: `message` (agent prose), `tool_call` / `tool_result`, `stage` / `model_training` / `model_trained` (unchanged training progress), `confirm` (a rail awaiting y/n), `auto_approved`, `done`, `error`.

## Testing (offline, same discipline as V1)

No live LLM, no PyCaret, no network - the whole agent runs on fakes.

- **`FakeChatModel`**: a `BaseChatModel` stub that emits a scripted sequence of tool calls then a final answer, so a test drives an exact agent trajectory deterministically.
- **Fake `train_fn`** (the V1 pattern): tools that train use the injected training function; tests inject one returning a canned `TrainResult`, so no PyCaret runs.
- **Per-tool unit tests**: each tool against a temp registry - `retrain` registers a candidate, `compare` computes deltas, `inspect` reads, `promote` moves `active`.
- **Registry round-trip**: `register` / `get` / `set_active` / `remove` / `load_predict` on disk under `tmp_path`.
- **Rail tests**: guarded mode `interrupt`s and a `no` raises `ToolDenied`; autonomous mode skips the interrupt and streams `auto_approved`.
- **End-to-end trajectory**: a scripted `FakeChatModel` drives "train -> retrain et -> compare -> promote" through the real graph + `MemorySaver`, asserting the registry ends with the promoted model active - the V2 analogue of V1's full-graph wiring test.

## What this unlocks (deliberately deferred)

Once this substrate exists, later specs are additive on top of it, not rewrites:

- the HTTP surface already exists (built here);
- an autonomous **planner** loop (agent pursues "get RMSE under 15" unattended) is new agent behavior over the same tools;
- more tools (feature engineering, new datasets, hyperparameter search) are registry-backed additions;
- richer rails (budgets, allow/deny lists) extend `require_confirmation`.

The foundation is: a reasoning hub, a typed tool layer, a model registry, and a rail seam - all tested offline.
