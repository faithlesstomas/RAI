# Rich Assistant live-model acceptance

## Verified run

The Issue #34 live gate was run on 2026-09-14 with no network model provider and
with a fresh `RAI_DATA_DIR` profile.

| Item | Recorded value |
|---|---|
| Backend | `llama-cpp-python 0.3.16`, CPU |
| Model | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` |
| Artifact | `sha256:9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0` |
| Model size | 638 MiB |
| Python | 3.12.11 |
| Host envelope | x86_64, AMD Ryzen 7 PRO 8840HS, 16 logical CPUs, 53 GiB RAM |
| Strategy | `DIRECT`, plain-text prompt v2, temperature 0.2, output ceiling 128 tokens |

The run used separate CLI processes and new assistant session IDs. The initial
Guile preference was admitted as a separate `MemoryRecord`. A later request had
an empty recent-turn window and selected that memory by ID. A correction created
a Python record that superseded Guile. After another process restart, the
manifest selected only the Python memory and the delivered answer named Python.

The raw TinyLlama output in the correction run ignored the selected preference.
This was useful negative evidence: the deterministic preference-grounding gate
caught the contradiction, returned the graph value and recorded a grounding
override in the run audit. The test therefore demonstrates both model execution
and a concrete hallucination-resistance boundary; it does not claim that a
1.1-billion-parameter model is generally reliable.

The deterministic acceptance suite additionally verifies source deletion,
non-reactivation of superseded memory, cancellation, timeout, invalid output,
poisoned/expired retrieval, privacy non-downgrade and concurrent duplicate
delivery.

## Reproduce it

Choose a disposable profile so existing assistant memory cannot affect the
result:

```bash
export RAI_DATA_DIR=/tmp/rai-assistant-experiment
uv run rai assistant ask \
  --backend llama --model /path/to/tinyllama.gguf --show-context \
  "Zapamiętaj, że w przykładach kodu preferuję Guile zamiast Pythona."

uv run rai assistant ask \
  --backend llama --model /path/to/tinyllama.gguf --show-context \
  "W jakim języku powinieneś pokazywać mi przykłady kodu?"
```

The second manifest must contain one durable-memory ID and no recent-turn IDs.
Repeat with `Zmień tę preferencję: używaj Pythona w przykładach kodu.` and then
ask again from a new process. The final manifest must select a different,
current memory ID and the answer must name Python.

For context-rot experiments, use one fixed `--session-id`, insert more than ten
distractor turns, then ask about the preference. Compare that with a new session
using the same `RAI_DATA_DIR`. `recent_turn_ids` is capped independently from
`durable_memory_ids`; `exclusions` shows entries removed by privacy, validity or
character budgets.

## Current limits

- Durable admission is limited to explicit Guile/Python code-example
  preferences; this is not general autobiographical or factual memory.
- TinyLlama frequently produces weak or repetitive prose. A stronger local
  instruct model should be used for qualitative hallucination studies.
- Token streaming currently emits a validated terminal response as one SSE/CLI
  chunk so that memory proposals cannot bypass the single commit path.
- Rich History references, voice, tools/actions, proactive turns, vector
  retrieval and remote model APIs are outside this slice.
- SQLite and JSONL files are access-restricted but not separately encrypted.
