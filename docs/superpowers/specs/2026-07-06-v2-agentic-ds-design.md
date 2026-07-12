# V2 Spec 1 - the agentic data scientist (design)

This is the design for Sentinel V2, first slice.
It replaces V1's fixed hub-and-spoke event router with a genuine tool-calling reasoning agent that can be delegated data-science work in natural language.
Last updated: 2026-07-06.

## Goal (and non-goals)

Turn Sentinel from a deterministic pipeline (interview -> train -> report -> monitor) into an agent you can talk to and delegate work to.
The anchor use case, the thing every design decision is measured against:

> "Retrain just Extra Trees with 500 estimators and compare it to the current best. If it wins, promote it."

Stated in one line:
LangChain's `create_agent` (the maintained successor to the deprecated `langgraph.prebuilt.create_react_agent`) becomes the graph's hub, owns a set of data-science tools (train, retrain-with-params, compare, inspect, report, monitor, promote), and decides which to call from the conversation - guarded by confirmation rails with an autonomous override.

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

`create_agent` (from `langchain.agents`, langchain 1.3.x) is the hub.
It is the maintained successor to `langgraph.prebuilt.create_react_agent`, which is deprecated since LangGraph v1.0; we install `langchain` and use `create_agent` rather than build a foundation on a deprecated prebuilt.
It is a ReAct loop: the chat model sees the conversation plus the bound tool schemas, emits tool calls, a tool node runs them, results feed back, and the loop continues until the model answers in prose.

```
   new user message                                confirmation reply
   invoke({messages:[HumanMessage]})               Command(resume="yes")
              |                                            |
              v                                            v
   +----------------------+       tool call        [ interrupt() inside a
   |     create_agent     | --------------------->   guarded tool, mid-loop ]
   |  (chat model + ReAct)| <---------------------
   +----------------------+       tool result
              |
         prose answer -> loop ends, wait for the next user message
```

Two distinct mechanisms drive the graph, and the design keeps them separate (this is the subtlety a naive "resume on every turn" gets wrong):

- **A new conversation turn is a fresh `invoke`** with a new `HumanMessage` on the *same* `thread_id`.
  `create_agent` ends its run after the prose answer - it does not stay suspended waiting for the next message.
  The checkpointer preserves the full message history under the thread, so the next turn just appends a `HumanMessage` and re-invokes; the agent sees the whole conversation.
- **`Command(resume=...)` is used *only* to answer a confirmation** raised by `interrupt()` inside a guarded tool mid-loop.
  It is never how a chat turn is delivered.

What V1 constructs disappear:
`orchestrator_node`, `route()`, the `_ROUTES` table, `route_interview`, and the trainer/report/monitor *nodes* (their logic moves into tools).
The `interviewer.py` `advance()` state machine (~400 lines: gate, deduction, defaults, non-answer bounds) dissolves entirely - a reasoning agent conversing *is* the interview, and it calls `save_config` when it has what it needs.

What stays, untouched or nearly so:
the checkpointer substrate (`SqliteSaver` default, `MemorySaver` in tests) - which now also preserves conversation history across turns - `interrupt()` for mid-tool confirmations, the streaming seam (`get_stream_writer` for progress events), `domain_context.glossary()` grounding, and the DS core (`sentinel/core/*`) plus its callback/stage-event hooks.

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
Under the agent, the graph's state is the **message history** (`create_agent`'s state) plus one extra field, `autonomy`, and a disk-backed model registry.
Config, training results, reports, and alerts become artifacts the tools read and write on disk - never graph-state fields.

The custom state schema subclasses `create_agent`'s own state base, not a bare `MessagesState`:
the schema is `class DSAgentState(langchain.agents.middleware.AgentState): autonomy: str` - it inherits the `messages` (and any bookkeeping keys `create_agent` needs) and adds our one field.
`create_agent` uses `system_prompt=` (a string or `SystemMessage`) where the old prebuilt used `prompt=`; there is no `version` kwarg.

This makes V1's hardest constraint disappear by construction.
The msgpack checkpoint serializer (no pickle fallback) rejected DataFrames, estimators, and closures; V1 handled it with the `to_state()` discipline.
Now nothing heavy is *ever* in state: the only references that cross the checkpoint boundary are LangChain messages (natively serializable) and string model ids.
The registry is the single source of truth for artifacts, addressed by id.

## Tools (`sentinel/agents/tools.py`)

Each tool is a typed function (`@tool` with a pydantic-validated signature) that wraps existing DS-core / sub-agent logic and reads/writes the registry.
Stateless dependencies (the `train_fn`, the chat model for report grounding, `ticket_dir`) come from the injected runtime config, not tool arguments - same dependency-injection discipline as V1.
The `autonomy` mode is the exception: it is per-session and must survive across invocations, so it lives in *checkpointed graph state* and reaches the rail via `InjectedState`, not `configurable` (see Safety rails below).

| Tool | Signature (args the model fills) | Does | Rail |
|---|---|---|---|
| `save_config` | `framing, failure_threshold, reporting_cadence, success_metric, rul_cap?, window?` | Persist the run config the agent gathered conversationally. Replaces the interviewer machine. | - |
| `train` | `rul_cap?, window?, models?` | Full comparison; registers **the winner** as a loadable model and records the full leaderboard *metrics*; returns the leaderboard summary. Wraps `train_and_evaluate`. | confirm |
| `retrain` | `model_id, hyperparameters, rul_cap?, window?` | Parameterized single-model train (NEW DS-core entrypoint); registers a candidate; returns its held-out metrics. | confirm |
| `evaluate` | `model_id` | Held-out metrics for a registered model. | - |
| `compare` | `model_id_a, model_id_b, rul_cap?, window?` | Side-by-side metrics + deltas, **only across a common evaluation config** (`rul_cap`/`window`, see below). | - |
| `inspect` | `what` | Read-only: leaderboard / dataset stats / a model's params / the registry listing. | - |
| `promote` | `model_id` | Set the registry's `active` model. | confirm |
| `delete` | `model_id` | Remove a registered candidate; **refuses the active model** (see below). | confirm |
| `write_report` | (none) | Grounded report over the active model's metrics. Wraps `report_writer`, keeps the no-fabrication prompt. | - |
| `run_monitor` | (none) | Step the active model's stored readings (`registry.readings(active_id)`) through it; emit alerts / mock tickets. Wraps `monitor`. Guarded because writing a ticket is the monitor's external *action*. | confirm |

Invalid tool calls (bad model id, malformed hyperparameters) and a declined confirmation both **return** a descriptive string the agent can recover from - never a raised exception.
This matters concretely: `create_agent`'s tool node only converts tool-*validation* errors into messages and re-raises anything else, so a custom exception from a denied confirmation would terminate the graph.
Guarded tools therefore return the denial string rather than raising (see the rail helper below).

Two integrity rules the tools enforce, because the agent will otherwise stumble into them:

- **Metrics are only comparable within one evaluation config.**
  `rul_cap` caps the RUL target and `window` changes the features, so a model trained under `rul_cap=100` has RMSE/MAE on a *different target scale* than one under `rul_cap=125` - a raw delta between them is meaningless and could promote a "better" number that is only better because its target was easier.
  Every model's `provenance.json` records its evaluation config (`rul_cap`, `window`).
  When the two models' configs match, `compare` deltas their stored metrics directly (the fast path).
  When they differ, it **re-evaluates both models on one common held-out target** via `evaluate` before deltaing.
  The common config is chosen deterministically: the `rul_cap`/`window` the caller passes to `compare` if given, else the active model's config; if the configs differ *and* neither an explicit config nor an active model is available, `compare` returns a message asking the caller to name a `rul_cap`/`window` rather than guessing.
- **The active model cannot be deleted out from under the system.**
  `registry.remove(id)` (and the `delete` tool) refuse to remove the currently-active model, returning a message telling the agent to promote another model first.
  This keeps `manifest.json.active` from ever pointing at a missing directory, which would break `write_report`, `evaluate`, and `run_monitor`.

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
    provenance.json             # { source, model_id, hyperparameters, config (rul_cap, window), parent, created_at }
    readings.json               # the held-out monitor readings (test_eval) for THIS model's config
```

Ids are short and human-legible: `et-v1`, `et-v2`, `lightgbm-v1`.
The registry API is small and native-JSON: `register(...) -> id`, `get(id)`, `list()`, `active()`, `set_active(id)`, `remove(id)`, `load_predict(id)`, `readings(id)`.
Tools are the only writers.
Nothing heavy leaves it: callers get a string id, a small metrics dict, or the readings records; the model itself is rehydrated on demand via `load_predict(id)` (the V1 `training.load_predict` pattern, generalized to the registry).

**Why `readings.json` is stored, not reconstructed.**
V1's monitor consumed the held-out `test_eval` frame (one row per FD001 test unit at its last cycle) that training produced.
Now that state is reduced to messages, the monitor needs a defined source for those rows.
They are config-dependent (the `window` feature knob changes them), so each model persists the exact readings built under *its own* config when it is registered - `run_monitor` loads `registry.readings(active_id)`.
`test_eval` is small (a few hundred rows) and native-JSON, so storing it per model is cheaper and more deterministic than re-loading FD001 and re-featurizing on every monitor call.

## Safety rails and autonomy

Two modes, one seam.

**Guarded (default).**
`train`, `retrain`, `promote`, `delete`, and `run_monitor` stop and ask the human `y/n` before running.
`train`/`retrain` are guarded because they burn real compute (minutes of PyCaret); `promote`/`delete` because they mutate the active system; `run_monitor` because writing a ticket is an external action.
The read/analytic tools (`evaluate`, `compare`, `inspect`, `write_report`) run freely.
Confirmation happens via `interrupt()`, so it works identically over the CLI and the HTTP/SSE surface and survives a checkpoint.

**Autonomous.**
A user-set mode that skips every confirmation - "delegate and walk away."
Even so, each auto-approved action streams a notice (`{"type": "auto_approved", "tool": ...}`) so an unattended run is still auditable.

**Autonomy is per-session state, not runtime config.**
`config["configurable"]` is passed fresh on every `invoke`/`resume` and is *not* checkpointed, so storing autonomy there would silently revert to the default on the next request.
Instead `autonomy` is written into checkpointed graph state when the session starts (`create_agent`'s state schema is extended with an `autonomy` field), and the rail reads it via `InjectedState` - so it persists for the life of the session without the client having to resend it.
The session-start value comes from the request (`--autonomous` on the CLI, the `autonomy` field on `POST /sessions`), defaulting from `get_settings()` (`SENTINEL_AUTONOMY=guarded|autonomous`, `guarded` default).

The seam is a single helper used by every guarded tool.
It **returns** on approval and **returns a denial string** on refusal - it never raises, because `create_agent`'s tool node would re-raise a custom exception and terminate the graph:

```python
def confirm(action: str, detail: str, autonomy: str) -> str | None:
    """Return None if the action may proceed, or a denial string the tool returns as-is."""
    if autonomy == "autonomous":
        get_stream_writer()({"type": "auto_approved", "tool": action, "detail": detail})
        return None
    answer = interrupt({"type": "confirm", "tool": action, "detail": detail})
    if str(answer).strip().lower() in {"y", "yes"}:
        return None
    return f"Declined: the user did not approve {action} ({detail})."

# in a guarded tool (autonomy comes from InjectedState):
def promote(model_id: str, state: Annotated[dict, InjectedState]) -> str:
    denial = confirm("promote", model_id, state["autonomy"])
    if denial:
        return denial            # clean message back to the agent, graph continues
    registry.set_active(model_id)
    return f"Promoted {model_id}; it is now the active model."
```

**Multiple confirmations in one turn.**
The model can emit two guarded tool calls in a single response (e.g. `retrain` then `promote`).
`create_agent` runs them as separate tool tasks, so both `interrupt()`s pend at once, and LangGraph then rejects a scalar `Command(resume="yes")` - it requires a map from interrupt id to answer.
So the confirmation contract is id-addressed, not scalar:
the `confirm` event carries the `interrupt` id (from `get_state(thread).interrupts`), and a resume supplies `Command(resume={<id>: "yes", ...})`.
A scalar answer is accepted only as a convenience when exactly one confirmation is pending (the common case).
Both transports read the pending interrupt ids from the checkpointer and echo them in the `confirm` event, so a client always answers the specific interrupt(s) rather than positionally.

## Surfaces: CLI and HTTP/SSE

Both are in this slice.

Both drive the graph through the same two mechanisms (new-message `invoke` vs. confirmation `resume`); the difference is just transport.

**CLI** (`python -m sentinel.agents`): a chat loop.
Read a line and `invoke` the agent with it as a `HumanMessage`; stream tool-calls / stage events / the prose answer.
If the run *interrupts* on one or more confirmations, prompt `y/n` for each pending interrupt and `Command(resume={id: answer})` to continue the *same* run; otherwise the run ends and we wait for the next line (a fresh `invoke`).
`--autonomous` sets the session's autonomy at start.

**HTTP/SSE** (`sentinel/api/app.py`): adapt the V1 surface to the agent.
`POST /sessions` (optional `{"autonomy": "..."}`, `{"message": "..."}`) starts a session, sets autonomy in state, and streams until the run ends or interrupts.
The endpoints mirror the two mechanisms explicitly rather than overloading one:
`POST /sessions/{id}/message` (`{"message": "..."}`) delivers a new conversation turn (a fresh `invoke` appending a `HumanMessage`);
`POST /sessions/{id}/resume` (`{"answers": {"<interrupt_id>": "y"}}`, or `{"answer": "y"}` when a single confirmation is pending) answers the pending confirmation(s) via `Command(resume=...)`.
`GET /sessions/{id}` reads a snapshot from the checkpointer (including any pending confirmation interrupt ids).
The SSE event vocabulary extends V1's: `message` (agent prose), `tool_call` / `tool_result`, `stage` / `model_training` / `model_trained` (unchanged training progress), `confirm` (a rail awaiting y/n, carrying its `interrupt` id), `auto_approved`, `done`, `error`.

## Testing (offline, same discipline as V1)

No live LLM, no PyCaret, no network - the whole agent runs on fakes.

- **`FakeChatModel`**: a `BaseChatModel` stub that emits a scripted sequence of tool calls then a final answer, so a test drives an exact agent trajectory deterministically.
- **Fake `train_fn`** (the V1 pattern): tools that train use the injected training function; tests inject one returning a canned `TrainResult`, so no PyCaret runs.
- **Per-tool unit tests**: each tool against a temp registry - `retrain` registers a candidate, `compare` computes deltas, `inspect` reads, `promote` moves `active`.
- **Registry round-trip**: `register` / `get` / `set_active` / `remove` / `load_predict` / `readings` on disk under `tmp_path`; `remove` refuses the active model.
- **Comparable-metrics test**: `compare` of two models with different `rul_cap` re-evaluates on a common target rather than deltaing the stored (incomparable) numbers.
- **Rail tests**: guarded mode `interrupt`s and a `no` makes the tool *return* a denial string (the graph continues, the active model is unchanged); autonomous mode skips the interrupt and streams `auto_approved`.
- **Batched-confirmation test**: a `FakeChatModel` that emits two guarded calls in one turn produces two pending interrupts, and a mapped `Command(resume={id: answer})` resolves both - guarding against the scalar-resume failure.
- **Autonomy persistence test**: a session started autonomous stays autonomous across a follow-up `invoke` (proves it is read from checkpointed state, not re-supplied config).
- **End-to-end trajectory**: a scripted `FakeChatModel` drives "train -> retrain et -> compare -> promote" through the real graph + `MemorySaver` across *multiple* `invoke` turns, asserting history carries over and the registry ends with the promoted model active - the V2 analogue of V1's full-graph wiring test.

## What this unlocks (deliberately deferred)

Once this substrate exists, later specs are additive on top of it, not rewrites:

- the HTTP surface already exists (built here);
- an autonomous **planner** loop (agent pursues "get RMSE under 15" unattended) is new agent behavior over the same tools;
- more tools (feature engineering, new datasets, hyperparameter search) are registry-backed additions;
- richer rails (budgets, allow/deny lists) extend `require_confirmation`.

The foundation is: a reasoning hub, a typed tool layer, a model registry, and a rail seam - all tested offline.
