
## Local Inference Integration Plan

This roadmap outlines the plan of implementing a **Local Inference** module(s) into RAI,
enabling high-performance execution of models via IREE, llama.cpp, and ONNX, 
while maintaining seamless compatibility with **Agno** and **Pydantic AI**.


---

### Development Roadmap: Functional Local Inference

#### Phase 1: Protocol Definition & Functional Core
* [ ] **Engine Protocol:** Define `InferenceEngine` protocol in `rai.protocols` for structural subtyping.
* [ ] **Monadic Error Handling:** Integrate `returns.result.Result` for all engine operations to ensure robust error propagation.
* [ ] **IREE Implementation:** Create a state-pure (where possible) IREE adapter using the defined protocol.

#### Phase 2: Pipeline Composition
* [ ] **Stream Composition:** Implement async generators for token streaming using functional wrappers.
* [ ] **Model Factory (Functional):** A function returning `Result[InferenceEngine, Error]` based on file signature analysis.

#### Phase 3: Framework Integration
* [ ] **Bridge Protocol:** Map RAI protocols to `pydantic_ai.models.Model` and `agno.models.Model` without breaking functional purity.

---

### Phase 1: Engine Abstraction Layer

**TODO: Revise this idea vs current code - take into account functional approach.**

*Goal: Decouple the inference logic from the agent logic.*

- [ ] **Unified `BaseEngine` Interface:** Define a core abstraction in `rai.engines` (or `rai.adapters.local`) with standard methods: `load()`, `generate()`, and `stream()`.
- [ ] **Llama.cpp Engine:** Implement a wrapper for `llama-cpp-python` to support **GGUF** models.
- [ ] **IREE (MLIR) Engine:** Integrate `iree-runtime` to execute compiled `.vmfb` modules. Must include support for HAL drivers (Vulkan, CUDA, CPU).
- [ ] **ONNX Runtime Engine:** Integrate `onnxruntime` for classic NLP tasks and embedding models.

### Phase 2: Framework Bridging

*Goal: Make local engines "look and feel" like OpenAI/Anthropic providers for high-level libraries.*

- [ ] **Agno (Phidata) Bridge:** Implement a custom `Model` class that maps Agno's internal request format to RAI’s local engines.
- [ ] **Pydantic AI Bridge:** Implement the `Model` protocol for Pydantic AI, ensuring full support for type-safe validation and response handling.
- [ ] **Local Model Factory:** A dispatcher that automatically initializes the correct engine based on file extensions (`.gguf` -> Llama.cpp, `.vmfb` -> IREE, `.onnx` -> ONNX).

### Phase 3: Structured Output & Tool Calling

*Goal: Enable local models to use RAI tools (Function Calling) reliably.*

- [ ] **GBNF Grammar Support:** Leverage llama.cpp's grammar feature to force local models to output valid JSON matching Pydantic schemas.
- [ ] **Tool Prompt Injection:** Create a standardized system prompt generator that injects tool definitions (JSON Schema) into the context for engines without native tool-calling support.
- [ ] **Output Parser:** A robust regex-based or JSON-repairing parser to extract tool calls from raw model streams.

### Phase 4: Rich UI & Telemetry (?)

*Goal: Enhance the "Rich" experience with local performance metrics.*

* [ ] **Performance Monitor:** Use `Rich.Live` to display real-time inference stats:
* Tokens per second (T/s).
* Active HAL Driver (e.g., "Vulkan GPU").
* Memory footprint.


* [ ] **Resource Manager:** A CLI utility to manage local model files and cache.

### Phase 5: Distribution & Packaging

*Goal: Keep RAI lightweight while offering heavy-duty local support.*

* [ ] **Optional Dependencies:** Update `pyproject.toml` to use extras (e.g., `pip install rai[iree]` or `pip install rai[llama]`) to avoid forcing heavy binaries on all users.
* [ ] **Hardware Auto-detection:** Logic to suggest the best local backend based on the user's hardware (e.g., suggesting IREE/Vulkan for Linux/AMD users).

---

### Implementation Strategy

To maintain the modularity of your recent refactor, the integration should follow this flow:

| Component | Responsibility |
| --- | --- |
| **`rai.engines`** | Low-level interaction with libraries (IREE, ONNX). |
| **`rai.models`** | High-level wrappers that implement Agno/Pydantic AI interfaces. |
| **`rai.cli`** | User-facing commands to download and run local models. |

> **Note:** Priority should be given to **Llama.cpp** (GGUF) as it is the most common format for local LLMs, followed by **IREE** for users seeking maximum performance on non-NVIDIA hardware.
