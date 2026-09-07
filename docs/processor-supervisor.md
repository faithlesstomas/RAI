# Bounded Local Intelligence & Processor Supervisor

Stage 4 introduces bounded local intelligence to Rich AI (RAI). The processor supervisor coordinates local language models and lightweight neural engines to extract structured, verifiable claims from desktop activity episodes while enforcing strict resource, concurrency, and privacy limits.

## Processor Supervisor Architecture

The `ProcessorSupervisor` implements the kernel's `LocalProcessor` protocol. It isolates the rest of the system from backend-specific inference quirks, memory leaks, and blocking operations.

```
+--------------------------------------------------------------------+
|                       ProcessorSupervisor                          |
|                                                                    |
|  +---------------------+  +-----------------+  +----------------+  |
|  | Concurrency Gate    |  | Idle Reaper     |  | Episode-to-    |  |
|  | (asyncio.Semaphore) |  | (TTL Unloader)  |  | Claim Engine   |  |
|  +----------+----------+  +--------+--------+  +--------+-------+  |
+-------------|----------------------|--------------------|----------+
              |                      |                    |
              v                      v                    v
+--------------------------------------------------------------------+
|                    LocalTextEngine (Protocol)                      |
|                                                                    |
|   +-------------------+  +-------------------+  +---------------+  |
|   |   OllamaEngine    |  |  LlamaCppEngine   |  |  IreeEngine   |  |
|   |  (AsyncClient)    |  |   (Thread-pool)   |  | (Frozen Stub) |  |
|   +-------------------+  +-------------------+  +---------------+  |
+--------------------------------------------------------------------+
```

### Key Guarantees

1. **Deterministic Concurrency Bounding:**
   Requests are gated by an `asyncio.Semaphore`. When the concurrency threshold is saturated, requests do not queue indefinitely; they fail immediately and gracefully with `ActionFailure(code="CAPACITY_EXCEEDED")`.
2. **Non-Blocking Operation:**
   Underlying C-extension libraries and synchronous generation routines (such as `llama-cpp-python`) are offloaded to dedicated worker threads via `asyncio.to_thread`. The main event loop remains fully responsive for perceptual collectors and kernel dispatch.
3. **Automated Idle Memory Eviction:**
   Local weights occupy significant host RAM and VRAM. A background idle reaper monitors the last request timestamp and invokes `engine.unload()` once the engine remains inactive beyond `idle_unload_seconds`.
4. **Cooperative Cancellation:**
   If a task is cancelled or exceeds its `InferenceBudget` deadline, running generation is cleanly halted and returns `ActionFailure(code="CANCELLED")`.
5. **Provenance & Claim Synthesis:**
   Transformations from Stage 3 `Episode` packages to `Claim` objects include cryptographic or identifier-based `ProvenanceReference` records linking claims directly to the evidence episodes.

---

## Supported Local Text Engines

Engines conform to the `LocalTextEngine` protocol, exposing uniform `load()`, `generate()`, `stream()`, and `unload()` asynchronous lifecycles.

### Ollama (`OllamaEngine`)
- Connects asynchronously via `ollama.AsyncClient`.
- Pulls models on demand if missing from the local daemon.
- Supports explicit eviction from GPU memory by issuing generation requests with `keep_alive=0` on unload.

### Llama.cpp (`LlamaCppEngine`)
- Runs GGUF quantized models directly on Linux CPU/Vulkan/ROCm/CUDA.
- Employs lazy importing to ensure that environments lacking C++ toolchains or `llama-cpp-python` can start the RAI kernel without dependency errors.
- Dispatches model initialization and token generation loops onto thread-pool workers.

### IREE (`IreeEngine`)
- Frozen stub for compiled MLIR/Vulkan neural workloads, guarded by `is_iree_available() -> False`.

---

## Container Integration

The processor supervisor is exposed on `ApplicationContainer`:

```python
from datetime import datetime, timedelta, timezone
from returns.result import Success
from rai.container import ApplicationContainer
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ContextManifest,
    ContextManifestItem,
    ContextPackage,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
    Task,
)

container = ApplicationContainer(config={})
supervisor = container.processor_supervisor
await supervisor.start()

# Check operational metrics
health = supervisor.health()
print(health.state, health.active_requests)

# Process a task with context package and budget
producer = ProducerIdentity(producer_id="client", kind="user", version="1.0.0")
task = Task(producer=producer, objective="Summarize episode")
context = ContextPackage(
    producer=producer,
    task_id=task.record_id,
    manifest=ContextManifest(
        producer=producer,
        destination="local-processor",
        items=(
            ContextManifestItem(
                source_id="item-1",
                source_type="episode",
                data_class=DataClass.LOCAL,
            ),
        ),
    ),
    content={"episode": {"applications": ["gedit"], "activity_types": ["edit"]}},
)
budget = InferenceBudget(
    producer=producer,
    max_input_tokens=1000,
    max_output_tokens=256,
    max_agent_turns=1,
    max_tool_calls=0,
    max_images=0,
    max_audio_seconds=0.0,
    max_latency_seconds=15.0,
    max_provider_cost=0.0,
    max_ram_bytes=1024 * 1024 * 1024,
    max_vram_bytes=0,
    cancellation_deadline=datetime.now(timezone.utc) + timedelta(seconds=30),
)
cancellation = CancellationToken()

result = await supervisor.process(
    task=task,
    context=context,
    budget=budget,
    cancellation=cancellation,
)
if isinstance(result, Success):
    claim = result.unwrap()
    print("Synthesized claim:", claim.statement, claim.data_class)
```

Supervisor teardown and model unloads are coordinated automatically during `container.close()`.
