# Sentinel context

Sentinel is a predictive-maintenance workspace whose agent operates over durable
model artifacts. The model registry is the source of truth for trained models,
leaderboards, provenance, readings, and future long-form artifacts.

## Conversation context

The checkpointed message history is the durable conversation transcript. It is
not itself the model's working-context contract. Agent assembly uses
`BoundedToolContextMiddleware` to project that history into a bounded request
context, clearing stale tool payloads while
preserving recent results. Request-only context editing must not mutate the
checkpoint or make old UI transcript entries disappear.

Large structured values stay in durable artifacts. Tools return compact,
decision-ready results or stable references. For example, the full leaderboard
is served to the UI from Registry through the leaderboard API, while ordinal
agent actions resolve one candidate through `leaderboard_candidate`.

Live SSE updates and hydrated snapshots share the presentation projection in
`sentinel/api/presentation.py`; internal LangChain messages are not a frontend
interface.

Persistent LLM summarization is intentionally deferred. It may be introduced
only after its summaries can coexist with an intact durable transcript and when
normal conversational history, rather than tool payloads, becomes the measured
source of context pressure.
