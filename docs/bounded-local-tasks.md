# Bounded local tasks

Stage 4.2 builds schema-constrained local understanding on top of the processor
supervisor. A bounded task is not an agent: it receives a task and an approved,
sanitized context package, performs one inference operation, and may only return
a validated derived `Claim` or a typed `ActionFailure`.

All six version `1.0.0` contracts and persistent policy-aware result caching are
implemented.

## Supported contracts

The registry contains six version `1.0.0` contracts:

| Kind | Output | Purpose |
| --- | --- | --- |
| `episode_summarization` | `EpisodeSummaryOutput` | Summarize observed episode facts without inventing actions or outcomes. |
| `intent_classification` | `IntentClassificationOutput` | Produce a non-authoritative classification hint that cannot approve or execute an action. |
| `entity_extraction` | `EntityExtractionOutput` | Extract at most 32 explicit entity mentions, each tied to an approved manifest source. |
| `salience_estimation` | `SalienceEstimationOutput` | Produce a normalized score and a deterministic low, medium or high score band. |
| `privacy_risk_elevation` | `PrivacyRiskElevationOutput` | Preserve or elevate the retained input classification; never downgrade it. |
| `routing_hint` | `RoutingHintOutput` | Suggest `LOCAL`, `ASK`, `ESCALATE` or `DENY` without dispatching or authorizing work. |

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
becomes a `Claim` or cache entry.

Task-specific validation also runs before claim creation. Entity source IDs must
exist in the retained, approved context manifest, salience labels must match the
documented score bands (`low` below `0.33`, `medium` below `0.67`, then `high`),
and privacy-risk output cannot downgrade the highest retained input data class.
An accepted privacy elevation becomes the derived claim's data class.

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

## Persistent result cache

`SQLiteBoundedResultCache` stores only canonical JSON produced from an output
that already passed its strict bounded-task schema. Failures, low-confidence or
partial output, cancellation and deadline results are never stored. A cache hit
is not accepted directly: the supervisor validates the cached JSON again
against the current contract, retained classification and approved source IDs,
then creates a new `Claim` bound to the current `ContextPackage` provenance.
Corrupt or incompatible entries are invalidated and become normal misses.

The version `1.0.0` key contract hashes a canonical request containing the
deterministic prompt, retained data class, output-token ceiling and provider
allow-list. It also binds the model name, immutable model artifact version,
bounded-task contract version, prompt version and policy version. Consequently,
changes to any execution or privacy boundary produce a miss. Public
`CacheLookupMetadata` exposes only the opaque digest, versions, task/model
identity, timestamps and typed hit/miss reason; it never exposes the prompt or
normalized input.

The local SQLite store defaults to 512 entries and a 24-hour TTL. Expired rows
are removed on access and writes evict least-recently-used rows over capacity.
The database lives under the RAI XDG cache directory and is created with mode
`0600`. Cache I/O failures fail open as misses because the cache is disposable;
all inference policy and validation checks still apply.

Container-managed caching requires an immutable artifact identity. It remains
disabled when `model_artifact_version` is absent:

```python
container = ApplicationContainer(
    config={
        "local_ai": {
            "backend": "ollama",
            "model": "qwen3:8b",
            "model_artifact_version": "sha256:<model-manifest-digest>",
            "result_cache": {
                "enabled": True,
                "ttl_seconds": 86400,
                "max_entries": 512,
            },
        }
    }
)
```

Changing weights under the same artifact version violates the cache contract.
Operators must update the version (preferably to a manifest digest), or disable
the cache with `result_cache.enabled = False`.

## Extension rules

New bounded tasks require separate registry entries with dedicated strict output
models and adversarial contract tests. They must not reuse permissive tool-call
parsers. Routing output remains a hint; deterministic policy makes the final
`LOCAL`, `ASK`, `ESCALATE` or `DENY` decision in Stage 4.5. Contract or prompt
changes must advance their versions so cached output cannot cross the new
validation boundary.
