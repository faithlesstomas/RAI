# Rich Assistant M4–M6 evaluation

This document records the reproducible acceptance evidence for the M4–M6
implementation in Issue #35. Graphiti is deliberately excluded.

## Reproduce the benchmark

Run the frozen corpus through a selected product backend:

```console
uv sync --extra dev --extra docs --extra inference-llama
rai assistant benchmark-memory \
  --backend ollama \
  --model qwen3.5:2b \
  --max-input-tokens 4096 \
  --max-output-tokens 256 \
  --max-latency-seconds 60 \
  --trials 3 \
  --energy auto \
  --output assistant-retrieval-benchmark.json
```

The command uses an isolated temporary SQLite store. The output fixes the
corpus, model-artifact fingerprint, prompt and judge versions, token, latency
and context budgets, local dense implementation, RRF weights, graph depth and
relation-confidence threshold. Retrieval failures and answer-utilization
failures remain separate fields.

The checked-in v1 manifests remain historical evidence:

- [FunctionGemma 270M](evaluation/assistant-m4-m6-functiongemma-270m.json)
- [TinyLlama 1.1B](evaluation/assistant-m4-m6-tinyllama-1.1b.json)

They are not acceptance evidence. Protocol v1 measured partially serialized
payloads, omitted failed generations from accuracy and energy denominators,
mixed raw model answers with deterministic grounding overrides, used one trial
and labelled rather than constructed history horizons.

The acceptance manifests use artifact `rai-assistant-retrieval-benchmark-v2`,
corpus `rai-assistant-retrieval-floor-v2` and judge
`deterministic-phrase-and-abstention-v3`:

- [Qwen 3.5 2B](evaluation/assistant-m4-m6-qwen3.5-2b-v2.json)
- [Qwen 3.5 4B](evaluation/assistant-m4-m6-qwen3.5-4b-v2.json)
- [Gemma 4 E4B](evaluation/assistant-m4-m6-gemma4-e4b-v2.json)

Each manifest contains three trials, 126 retrieval measurements (seven cases,
six channels, three trials) and 30 routing measurements. Each records raw model
text separately from the delivered grounded answer. Headline accuracy and
energy coverage use all attempted cases. The benchmark measures the final
serialized evidence layer and constructs actual short, medium and long reply
histories with distractors. All three manifests have a non-empty fingerprint,
zero retrieval failures, zero answer-evaluation failures and energy coverage
1.0.

Judge v3 remains deterministic. It accepts conservative Polish inflection,
recognizes explicit forms such as `nie ma informacji` and `brak jest danych`,
and rejects answers that turn a future plan into a completed event. It does not
use the answer model to grade itself.

## M4 retrieval result

All channels reached mean recall 1.000. Raw turns and claims reached mean
precision 0.643; dense, RRF and graph reached 0.386. Final grounded answer
accuracy was 1.000 for every channel and model. The raw-model column is still
load-bearing evidence: it shows where deterministic evidence grounding changed
an otherwise incorrect or unsafe model answer.

| Model | Raw turns | Claims | Summary | Dense | RRF | Graph |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.5 2B, final | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Qwen 3.5 2B, raw model | 1.000 | 1.000 | 1.000 | 0.905 | 0.952 | 0.905 |
| Qwen 3.5 4B, final | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Qwen 3.5 4B, raw model | 0.952 | 1.000 | 0.857 | 1.000 | 1.000 | 1.000 |
| Gemma 4 E4B, final | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Gemma 4 E4B, raw model | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The equal final scores do not make the representations equivalent. Claims use
about 286 serialized context characters on average. Raw turns use about 246,
but retain less structured semantics. Grounded summaries use about 942 and
select exactly the same source IDs as claims in every trial
(`distinct_from_claims_rate = 0`). Dense, RRF and graph use about 749 characters,
have lower precision and differ from claim BM25 in 71.4% of cases without a
quality gain. Claim BM25 therefore remains the default compact-memory channel.

Grounding now consumes all eligible evidence representations. Direct,
high-confidence user facts recovered from raw episodic turns and semantic
claims carried by source-covered summaries can ground a personal answer just
like durable claims. Uncertain, quoted, hearsay or low-confidence sources remain
ineligible. This avoids both hallucination and the earlier false `no memory`
override when raw or summary evidence was present.

## M5 routing and write-back

For every model and trial, adaptive and always-memory policies reached source
accuracy 1.000 and final answer accuracy 1.000 on short, medium and long
histories under the same 8,000-character budget.

| Horizon | Adaptive context chars | Always-memory chars | Quality result |
|---|---:|---:|---|
| Short | 743.0 | 1,001.5 | both 1.000; adaptive uses 25.8% less context |
| Medium | 1,719.5 | 1,719.5 | both 1.000 |
| Long | 2,075.0 | 1,861.0 | both 1.000; adaptive uses 11.5% more context |

The short-history objective is met: the real `recent_conversation` route is as
accurate and cheaper. The long-history objective is met on quality and fixed
budget, but not on context minimization; raw fallback evidence can be larger
than the compact-only baseline. This is recorded as a tuning opportunity, not
misreported as a cost win.

The route and sufficiency decision are recomputed after character-budget
pruning. Explicit memory questions with no eligible evidence abstain before
model generation. Verified derived write-back requires active same-scope
sources, numeric confidence for every source, complete source coverage,
verifier and policy identity, and provenance-bearing `SUPPORTS` edges.
Forgetting or deleting a supporting source recursively invalidates its derived
state. `FORGET` also removes the source exchange from raw FTS and recent model
context while preserving the separate audit record.

## M6 activation decision

The evaluated dense, weighted-RRF and bounded-graph channels are complete as
opt-in experiments, but none is promoted into the default context builder.
They did not improve final answer quality over claim BM25 across three local
models, selected more material, and had lower precision. Grounded summaries
also add size without selecting evidence distinct from claims.

Graph traversal still enforces its security contract: only `ALLOW`, eligible,
provenance-bearing relations at or above confidence 0.75 can participate. The
corpus includes a denied edge, and selection-integrity tests prove that it
cannot affect a trusted path. Manifests expose graph node and relation IDs.

A learned local embedding, cross-encoder or personalized PageRank should be
added only after a target workload demonstrates a gap that claim BM25 and the
bounded graph baseline cannot cover. It must rerun this protocol under equal
budgets before activation.

## Energy interpretation

The workstation exposes readable `amdgpu:PPT` power rather than CPU package
RAPL. The manifests therefore label scope `unattributed:amdgpu:PPT` and record
the exact sysfs sensor. Coverage is 1.0 over all attempted generations, but the
joule totals are whole-device GPU readings and are not attributable CPU/model
energy. They support within-run diagnostics, not portable efficiency claims.
