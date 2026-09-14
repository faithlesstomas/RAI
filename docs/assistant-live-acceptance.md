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
| Strategy | `DIRECT`, plain-text prompt v3, temperature 0.2, output ceiling 256 tokens |

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

## Functional-chat follow-up

The expanded MVP was verified on 2026-09-14 with both the GGUF model above and
the active local Ollama profile using `qwen3.5:2b`. In a fresh data directory,
`Jestem Tomek, a Ty?` admitted a `user.identity.name` fact. A new process and a
new session then answered `Jak mam na imię?` with `Masz na imię Tomek.` while
the manifest contained the durable-memory ID and an empty recent-turn list.

The same stored value was recalled through all three standalone surfaces:
`rai assistant ask`, interactive `rai assistant chat`, and `rai -p`. The
interactive chat also listed the active record through `/memories`. Ollama
reasoning output is disabled for this plain assistant path because
`qwen3.5:2b` otherwise consumed the complete output budget in its hidden
`thinking` field and returned an empty visible response.

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

The personal-fact path can be reproduced independently:

```bash
export RAI_DATA_DIR=/tmp/rai-assistant-name-experiment
uv run rai assistant ask --backend ollama --model qwen3.5:2b \
  "Jestem Tomek, a Ty?"
uv run rai assistant ask --backend ollama --model qwen3.5:2b \
  --show-context "Jak mam na imię?"
uv run rai assistant chat --backend ollama --model qwen3.5:2b
```

Existing turns created by an older build are not retroactively promoted into
memory. Restate the fact or use a fresh `RAI_DATA_DIR` when repeating an older
experiment.

For context-rot experiments, use one fixed `--session-id`, insert more than ten
distractor turns, then ask about the preference. Compare that with a new session
using the same `RAI_DATA_DIR`. `recent_turn_ids` is capped independently from
`durable_memory_ids`; `exclusions` shows entries removed by privacy, validity or
character budgets.

## Current limits

- Durable admission covers name, age, home location, explicit
  `remember`/`zapamiętaj` facts and Guile/Python code-example preferences. It is
  still a bounded rule set, not unrestricted autobiographical extraction.
- TinyLlama frequently produces weak or repetitive prose. A stronger local
  instruct model should be used for qualitative hallucination studies.
- Token streaming currently emits a validated terminal response as one SSE/CLI
  chunk so that memory proposals cannot bypass the single commit path.
- Rich History references, voice, tools/actions, proactive turns, vector
  retrieval and remote model APIs are outside this slice.
- SQLite and JSONL files are access-restricted but not separately encrypted.
