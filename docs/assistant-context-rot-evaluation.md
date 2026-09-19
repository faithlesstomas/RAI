# Assistant context-rot evaluation

`rai assistant benchmark-context-rot` measures how raw model output changes as
irrelevant conversation grows and a relevant exchange moves through the input.
It also runs one compact graph-memory control per scenario and trial. The
control measures a different question—whether the complete RAI response path
can use selected memory—and is not relabelled as if it had a long-context
length or needle position.

## Protocol v2

Protocol artifact `rai-assistant-context-rot-v2` corrects several validity
problems in the historical v1 reports:

- raw model output and the final response delivered after deterministic RAI
  grounding are judged and reported separately;
- backend failures have explicit status and failure codes and are excluded from
  recall and hallucination denominators;
- prompt processing time is reported as end-to-end latency because the current
  non-streaming backend contract does not expose time to first token;
- backend-reported prompt tokens remain separate from the documented heuristic
  used to construct the haystack;
- every measurement records the estimated realized context length and needle
  midpoint, rather than treating a requested length label as exact;
- graph memory is a single constant control instead of being duplicated across
  every raw-context length and position;
- binomial rates include Wilson 95% intervals and the report records model,
  prompt-template, judge, corpus, sampling and configured context-window
  metadata;
- an exploratory effective-context summary uses the NoLiMa convention of
  retaining at least 85% of the shortest-context baseline. It is marked invalid
  if a cell has fewer than three scored needles, its baseline is zero, or the
  backend failure rate exceeds 20%.

The bundled v2 corpus contains Polish and English literal-retrieval and
semantic-retrieval probes. It is a deterministic smoke and regression corpus,
not a publishable estimate of general model capability.

Example:

```bash
uv run rai assistant benchmark-context-rot \
  --backend lemonade \
  --model Qwen3.5-4B-GGUF \
  --server-context-window 65536 \
  --token-steps 512,2048,8192,16384,32768 \
  --trials 3 \
  --output docs/evaluation/assistant-context-rot-qwen4b-v2.json
```

For Lemonade and Ollama, `--server-context-window` is disclosure metadata: RAI
does not currently control or reliably discover the server's actual limit.
Confirm that value in the server configuration and verify that requests are not
silently truncated. For the direct llama backend, RAI records the configured
`context_window` automatically.

## Existing benchmark families

The internal corpus should be complemented, not replaced, by externally
maintained suites. Published scores remain comparable only when prompt format,
tokenization, truncation, tools, output budget, sampling and grading match the
upstream protocol.

| Suite | What it measures | Recommended RAI use |
|---|---|---|
| [RULER](https://github.com/NVIDIA/RULER) | Length-controllable synthetic retrieval, multi-hop variable tracing, aggregation and QA | First external adapter for model-only curves at 4K–128K. Apache-2.0 and deterministic grading make it the most practical conformance suite. |
| [MRCR v2](https://github.com/google-deepmind/eval_hub/tree/master/eval_hub/mrcr_v2) | Reproduction of one requested response among many similar multi-round dialogue turns, with 2/4/8 relevant needles | Second adapter and the closest external test of long conversational histories. Use the upstream metric, disclose whether tools are disabled, and begin with bounded 4K–128K subsets. Apache-2.0. |
| [LongBench](https://github.com/THUDM/LongBench) | Bilingual realistic QA, summarization, few-shot learning, synthetic retrieval and code completion; LongBench-E provides length buckets | Use selected English/Chinese and code tasks for downstream validity after synthetic failures are understood. MIT licensed, but individual source-dataset terms still require review. |
| [LongBench v2](https://github.com/THUDM/LongBench) | 503 difficult multiple-choice cases covering documents, dialogue, repositories, structured data and in-context learning | Optional expensive reasoning suite. It is useful for whole-task quality, not for precisely locating context-rot onset because length and task difficulty vary together. |
| [AA-LCR](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR) | 100 questions requiring synthesis across roughly 100K-token real-world document sets | Optional release comparison for models that can complete 100K inputs. Preserve the official prompt and judge; do not mix its scores with the internal deterministic judge. Apache-2.0 dataset. |
| [NoLiMa](https://github.com/adobe-research/NoLiMa) | Needle retrieval with deliberately low lexical overlap and an 85%-of-baseline effective-length definition | Reproduce the design principle with RAI-owned fixtures. The upstream Adobe Research License is non-commercial, so its data should remain an explicit research-only opt-in rather than a bundled default. |
| [HELMET](https://github.com/princeton-nlp/HELMET) | Seven application-oriented categories including recall, RAG, reranking, citations, long QA, summarization and in-context learning | Later holistic cross-check. Its breadth and dependencies are less suitable for the fast local regression command. |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | Long-term assistant memory: extraction, multi-session reasoning, updates, temporal reasoning and abstention | Adapt selected cases for `benchmark-memory`, not for pure model context rot. Retrieval and answer utilization must retain separate budgets and metrics. |

## Planned integration order

1. Keep the RAI-owned v2 corpus as the offline CI and regression floor.
2. Add import adapters for generated RULER cases and bounded MRCR v2 CSV
   subsets without making either framework a runtime dependency.
3. Run the same raw-model protocol at 4K, 8K, 16K, 32K, 64K and 128K where
   the configured backend can fit those inputs.
4. Add selected LongBench/LongBench-E cases to check whether synthetic curves
   predict realistic QA and code behavior.
5. Run AA-LCR or LongBench v2 only as an opt-in release evaluation because the
   compute cost is unsuitable for routine local development.
6. Adapt LongMemEval separately to compare RAI retrieval and graph-memory
   strategies under equal returned-context budgets.

The project must retain upstream dataset identifiers, versions, licences,
prompts and metrics in every imported run manifest. External corpus downloads
remain an explicit evaluation setup step; normal RAI installation and tests do
not fetch them.
