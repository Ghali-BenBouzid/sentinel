# Learning note 04 - building an agentic data scientist

This note explains why Sentinel replaced its fixed event router with a
tool-calling reasoning agent.
It covers the small state schema, checkpointed autonomy, confirmation rails,
model registry, and id-addressed confirmation resume.

The deterministic data-science core did not change.
What changed is who decides which operation should happen next.

---

## 1. From an event router to a reasoning hub

V1 had a LangGraph graph, but its control flow was fixed.
An orchestrator read `state["event"]` and selected the next node from a routing
table.
The sequence was always interview, train, report, monitor.

That design was resumable and testable, but it could not handle a request such
as:

> Retrain Extra Trees with 500 estimators, compare it to the current best, and
> promote it if it wins.

Supporting that request in a fixed router would require adding more events,
branches, and state-machine transitions.
Every new data-science operation would expand the control-flow table.

V2 makes LangChain's `create_agent` the hub.
The chat model receives the conversation and typed tool schemas.
It emits a tool call, the tool runs, and its result becomes another message in
the reasoning loop.
The model continues until it produces a prose answer.

```text
HumanMessage
    |
    v
create_agent -> tool call -> tool result
    ^                          |
    +--------------------------+
    |
    v
AIMessage
```

This changes control flow without discarding the tested implementation.
The data loader, feature builder, AutoML wrapper, report writer, and monitor
remain ordinary Python functions.
They are now reached through tools rather than graph nodes.

The tradeoff is explicit.
The old router was deterministic but rigid.
The reasoning loop is flexible but depends on model judgment.
Typed tool inputs, confirmation rails, and scripted offline tests constrain that
judgment at the system boundary.

---

## 2. What `create_agent` supplies

`langchain.agents.create_agent` builds the maintained ReAct-style loop.
It binds tools to a chat model, executes requested calls, appends tool results to
the message history, and calls the model again.
Sentinel does not implement that loop itself.

The assembly point stays small:

```python
return create_agent(
    chat_model,
    tools,
    system_prompt=SYSTEM_PROMPT,
    state_schema=DSAgentState,
    checkpointer=checkpointer,
)
```

The important arguments are:

- `chat_model`: a tool-calling LangChain chat model.
- `tools`: the typed operations returned by `make_tools`.
- `system_prompt`: domain rules and behavioral constraints.
- `state_schema`: the agent's native state plus Sentinel's one extra field.
- `checkpointer`: conversation and interrupt persistence.

The custom state schema subclasses the agent middleware's own schema:

```python
from langchain.agents.middleware import AgentState


class DSAgentState(AgentState):
    autonomy: str
```

This is different from creating a new `TypedDict` with only `messages`.
`AgentState` owns the message field and any bookkeeping keys that
`create_agent` needs.
Subclassing it preserves that contract while adding only the checkpointed value
Sentinel owns.

State is intentionally small.
It contains LangChain messages and the string autonomy mode.
DataFrames, estimators, prediction closures, and training dataclasses never
cross the checkpoint boundary.

---

## 3. Tools are the application boundary

The agent can only perform data-science work by calling tools.
That gives the system a narrow place to enforce invariants.

Sentinel exposes ten tools:

- `save_config`
- `train`
- `retrain`
- `evaluate`
- `compare`
- `inspect`
- `promote`
- `delete`
- `write_report`
- `run_monitor`

The tools close over runtime dependencies such as the registry, training
functions, report chat model, and ticket directory.
Those dependencies are not model-supplied arguments.
The model sees only the operation's domain inputs.

Expected failures return descriptive strings.
A missing model id, deletion of the active model, or declined confirmation does
not raise.
This matters because the default tool node converts schema-validation failures
into tool messages but re-raises other exceptions.
An exception for a routine denial would terminate the graph instead of letting
the agent explain the outcome or choose another action.

Returning a string keeps the failure inside the reasoning loop:

```python
try:
    registry.set_active(model_id)
except KeyError:
    return f"No model '{model_id}' in the registry."
return f"Promoted {model_id}; it is now the active model."
```

Unexpected programming and infrastructure failures are still exceptions.
The return-not-raise rule applies to expected domain outcomes.

---

## 4. Checkpointed autonomy through `InjectedState`

Guarded mode is the default.
Training, retraining, promotion, deletion, and monitoring require human
approval.
Read-only inspection, evaluation, comparison, and report writing do not.

Autonomous mode skips confirmations.
It still emits an `auto_approved` custom event so unattended operation remains
auditable.

Autonomy is a property of the session.
It must survive from one conversation turn to the next.
Putting it in `config["configurable"]` would be incorrect because runtime config
is supplied fresh on every invocation and is not checkpointed.

Instead the first invocation writes `autonomy` into `DSAgentState`.
Guarded tools receive the current state through an injected argument:

```python
def promote(
    model_id: str,
    state: Annotated[dict, InjectedState],
) -> str:
    denial = confirm("promote", model_id, state["autonomy"])
    if denial:
        return denial
    ...
```

`InjectedState` hides the argument from the model's tool schema.
The model supplies `model_id`.
LangGraph supplies the checkpointed state.

This separation prevents a tool call from choosing its own safety mode.
It also means follow-up messages do not need to resend autonomy.

---

## 5. The confirmation rail

All guarded tools share one helper:

```python
def confirm(action: str, detail: str, autonomy: str) -> str | None:
    if autonomy == "autonomous":
        get_stream_writer()({
            "type": "auto_approved",
            "tool": action,
            "detail": detail,
        })
        return None
    answer = interrupt({
        "type": "confirm",
        "tool": action,
        "detail": detail,
    })
    if str(answer).strip().lower() in {"y", "yes"}:
        return None
    return f"Declined: the user did not approve {action} ({detail})."
```

In guarded mode, `interrupt()` suspends the graph in the middle of the tool.
The checkpointer stores the pending work.
A later resume supplies the answer and the tool continues.

The helper returns a denial string rather than raising a custom exception.
The tool returns that string as its normal result.
The agent then sees a clean `ToolMessage` and can respond appropriately.

The CLI and API use the same rail.
Only their transport differs.

---

## 6. Why confirmations are addressed by id

A model may emit more than one guarded tool call in one response.
LangGraph can then have multiple interrupts pending at the same time.
A scalar resume answer is ambiguous because it does not identify which
confirmation it answers.

Each pending interrupt has a stable id.
Clients resume with a mapping:

```python
Command(
    resume={
        "interrupt-id-for-retrain": "yes",
        "interrupt-id-for-promote": "no",
    }
)
```

The API includes that id in every `confirm` SSE event.
The CLI reads all pending interrupts and prompts for each one.

For convenience, the HTTP API accepts `{"answer": "yes"}` when exactly one
interrupt is pending.
When several are pending, callers must provide the `answers` map.
This prevents positional or ordering assumptions from becoming part of the
protocol.

---

## 7. The registry is the source of truth

A single saved model path was enough when every run replaced the previous
winner.
Parameterized retraining creates multiple candidates, so names such as
"current best" and `et-v2` need durable meaning.

The registry stores each model under a human-readable id:

```text
artifacts/models/
  manifest.json
  et-v1/
    model.pkl
    metrics.json
    provenance.json
    readings.json
  et-v2/
    ...
```

`manifest.json` lists models and names the active one.
The first registered model becomes active.
Promotion changes the active id.
Deletion refuses the active model so the manifest can never point at a missing
directory.

Provenance records the model family, hyperparameters, parent, and evaluation
configuration.
Metrics are comparable only when `rul_cap` and `window` match.
When configurations differ, the compare tool re-evaluates both models on a
common target rather than subtracting incomparable stored scores.

Each model also stores the held-out readings produced by its feature
configuration.
The monitor can therefore reload the active model and its exact evaluation rows
without placing either object in graph state.

Tools are the registry's only writers.
The model never manipulates registry files directly.

---

## 8. Two ways to continue a session

A completed agent turn is not suspended.
The next user message is a fresh invocation on the same thread:

```python
agent.invoke(
    {"messages": [HumanMessage("compare the candidates")]},
    thread,
)
```

The checkpointer restores earlier messages, and the new message is appended.

A confirmation is different.
It resumes a tool that is currently suspended at `interrupt()`:

```python
agent.invoke(Command(resume={interrupt_id: "yes"}), thread)
```

The API exposes this distinction directly:

- `/sessions/{id}/message` starts a new conversation turn.
- `/sessions/{id}/resume` answers pending confirmations.

Keeping these operations separate avoids accidentally treating ordinary chat as
an interrupt response.

---

## Exercises

1. Add a read-only `list_tickets` tool that returns ticket summaries without
   requiring confirmation.
   Decide which filesystem failures are expected outcomes and should become
   strings.

2. Make `train` guarded only for the first training run in a session.
   Identify what additional checkpointed state is required and why registry
   contents alone may not express session history.

3. Add a budget rail that refuses a third retrain in one session.
   Keep the counter out of runtime config and test that it survives a follow-up
   invocation.

4. Extend the compare tool with an explicit named evaluation profile.
   Preserve the rule that raw metrics from different target configurations are
   never subtracted directly.
