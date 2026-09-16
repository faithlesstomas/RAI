# Rich Assistant M4–M6 evaluation

This document records the reproducible evidence used to close the M4–M6
implementation work for Issue #35. Graphiti is deliberately excluded.

## Reproduce the benchmark

Install the declared local backend and run the frozen corpus:

```console
uv sync --extra dev --extra docs --extra inference-llama
rai assistant benchmark-memory \
  --backend llama \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --max-output-tokens 48 \
  --max-latency-seconds 30 \
  --energy auto \
  --output assistant-retrieval-benchmark.json
```

The command uses an isolated temporary SQLite store. The output fixes the
corpus, model artifact hash, prompt and judge versions, token/latency/context
budgets, local dense implementation, RRF weights, graph depth and relation
confidence threshold. Retrieval failure and answer-utilization failure remain
separate fields.

The checked-in live manifests are:

- [FunctionGemma 270M](evaluation/assistant-m4-m6-functiongemma-270m.json)
- [TinyLlama 1.1B](evaluation/assistant-m4-m6-tinyllama-1.1b.json)

Both runs used the seven-case `rai-assistant-retrieval-floor-v1` corpus and the
independent `deterministic-phrase-and-abstention-v1` judge. The route comparison
also covers short, medium and long histories under the same 8,000-character
budget.

## Result and activation decision

| Model | Channel | Mean recall | Mean precision | Answer accuracy |
|---|---:|---:|---:|---:|
| FunctionGemma 270M | claims BM25 | 1.000 | 0.643 | 0.500 |
| FunctionGemma 270M | dense / RRF / graph | 1.000 | 0.386 | 0.500 |
| TinyLlama 1.1B | claims BM25 | 1.000 | 0.643 | 0.429 |
| TinyLlama 1.1B | dense / RRF / graph | 1.000 | 0.386 | 0.429 |

The feature-hashing dense baseline, weighted RRF and bounded graph traversal do
not beat claim BM25 on this target workload. They therefore remain opt-in and
are not wired into the default product context builder. This is a negative
benchmark result, not a hidden promotion decision. A future learned local
embedding may replace the dependency-free dense baseline through the
`DenseEmbeddingProvider` boundary and must rerun the same comparison.

The graph test admits only `ALLOW`, eligible, provenance-bearing relations at
or above confidence 0.75 before traversal. The corpus includes a denied edge;
selection-integrity tests prove that it cannot enter a path. Selected graph
paths expose both node and relation IDs in `AssistantContextManifest`.

## M5 routing and write-back

The adaptive router and always-memory baseline both achieve source recall 1.0
for the frozen short, medium and long cases. For the long case the adaptive
policy records a raw-evidence fallback because compact-memory sufficiency is
below threshold; the baseline does not. The manifest records the policy
version, coverage, evidence quality, confidence, rejected route and final
budget. Answer accuracy matches the baseline for every completed comparison;
on the FunctionGemma long-history case the fallback answers correctly while the
always-memory backend produces an evaluation failure. Explicit memory questions
with no eligible evidence now abstain before model generation.

Derived write-back is an explicit verified operation rather than a side effect
of assistant prose. It requires active source memories, complete source
coverage, identical profile/purpose/domain, non-broadened privacy, a verifier
identity and policy version. The stored inferred claim and every `SUPPORTS`
edge carry provenance. Deleting the primary source removes the derived state.

## Energy interpretation

Both recorded runs have measurement coverage 1.0, but this workstation exposes
CPU package RAPL energy only to root. The readable fallback sensor is
`amdgpu:PPT`, so the manifests label its scope as
`unattributed:amdgpu:PPT` and list the exact sysfs sensor. These joule totals are
real readings over each inference interval, but they are not attributable CPU
model energy and must not be used for model-efficiency claims. On a host with a
readable package RAPL counter, the same code records scope `cpu-package`.
