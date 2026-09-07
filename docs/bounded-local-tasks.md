# Bounded local tasks

Stage 4.2 builds schema-constrained local understanding on top of the processor
supervisor. A bounded task is not an agent: it receives a task and an approved,
sanitized context package, performs one inference operation, and may only return
a validated derived `Claim` or a typed `ActionFailure`.

## Supported contracts

The initial registry contains two version `1.0.0` contracts:

| Kind | Output | Purpose |
| --- | --- | --- |
| `episode_summarization` | `EpisodeSummaryOutput` | Summarize observed episode facts without inventing actions or outcomes. |
| `intent_classification` | `IntentClassificationOutput` | Produce a non-authoritative classification hint that cannot approve or execute an action. |

Each `BoundedTaskContract` fixes the output model, prompt instruction, token,
latency, RAM and VRAM ceilings, minimum confidence, and failure behavior. The
effective inference budget is the intersection of the caller's limits and the
contract limits; a contract can never expand caller authority or resources.
Bounded generation uses temperature `0.0`; sampling is not caller-controlled.

## Validation boundary

Model text is untrusted. The supervisor accepts it only when the complete
response is one JSON object accepted by the contract's strict Pydantic model.
Markdown fences, commentary, unknown fields, invalid enum values, out-of-range
confidence and tool-call-shaped objects fail with
`ActionFailure(code="INVALID_MODEL_OUTPUT")`. Valid output below the contract's
confidence threshold fails with `ActionFailure(code="LOW_CONFIDENCE")`.

The raw response is not copied into failure messages. A failed response never
becomes a `Claim`.

The prompt labels context as untrusted data and forbids tool or capability
requests. More importantly, this is an architectural boundary rather than a
prompt-only control: `ProcessorSupervisor` exposes no actuator or capability
registry to the inference engine, and bounded output is parsed only as a task
result schema.

## Usage

```python
from rai.inference import BoundedTaskKind

result = await supervisor.process_bounded(
    BoundedTaskKind.EPISODE_SUMMARIZATION,
    task,
    context,
    budget,
    cancellation,
)
```

The returned claim keeps the sanitized context package as provenance, preserves
the highest retained input data class, uses validated model confidence, and
records the task kind and contract version in `epistemic_status`.

## Extension rules

Entity extraction, salience estimation, privacy-risk elevation and routing hints
must be added as separate registry entries with dedicated strict output models.
They must not reuse permissive tool-call parsers. Routing output remains a hint;
deterministic policy makes the final `LOCAL`, `ASK`, `ESCALATE` or `DENY`
decision in Stage 4.5. Persistent result caching is a separate follow-up because
its key must include the task contract, model artifact, normalized input and
policy versions.
