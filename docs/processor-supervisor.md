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
from rai.container import ApplicationContainer

container = ApplicationContainer()
supervisor = container.processor_supervisor

# Check operational metrics
health = supervisor.health()
print(health.state, health.active_requests)

# Process an episode within budget
result = await supervisor.process(episode=episode, budget=budget)
if isinstance(result, Success):
    claim = result.unwrap()
    print("Synthesized claim:", claim.subject, claim.predicate)
```

Supervisor teardown and model unloads are coordinated automatically during `container.close()`.
