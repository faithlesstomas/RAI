## Local Inference Integration Plan

This roadmap outlines the architecture for implementing a **Local Inference** module in RAI. It aligns with the project's functional paradigm (`returns`, `Protocols`) and aims to expose high-performance local runtimes (IREE, Llama.cpp) through a unified, type-safe interface compatible with **Agno** and **Pydantic AI**.

Based on architectural discussions (Gemini conversation), this implementation prioritizes **dependency injection**, **structural subtyping**, and **monadic error handling**.

---

### Core Architecture: "Pure Bridges"

The design separates the *execution* of a model from the *interface* required by high-level agents.

1.  **Inference Engine (`rai.inference.core`)**
    *   Defined by the `InferenceEngine` **Protocol**.
    *   Responsible solely for raw text generation and token streaming.
    *   **MUST** return `returns.result.Result` for all operations.
    *   **MUST** be stateless where possible (configuration frozen in dataclasses).

2.  **Functional Bridges (`rai.inference.bridges`)**
    *   "Parsers/Transformers" that adapt the `InferenceEngine` to external frameworks.
    *   **Agno Bridge:** A lightweight `Model` subclass that translates Agno's `Message` objects into a prompt string for the engine, and parses the engine's text output back into Agno's format.
    *   **Pydantic AI Bridge:** A similar adapter satisfying `pydantic_ai.models.Model`.

3.  **Functional Composition for Tools**
    *   Instead of heavy inheritance, we use functional composition (e.g., a `with_tools` higher-order function or decorator) to inject tool schemas (JSON Schema) into the system prompt and attach an output parser.

---

### Detailed Development Roadmap

#### Phase 1: Engine Abstraction & Protocols
*Goal: Define the "shape" of a local model and implement the low-level execution logic.*

* [ ] **Define Protocol:** Create `InferenceEngine` in `rai.inference.protocols`.
    ```python
    class InferenceEngine(Protocol):
        def generate(self, prompt: str, stop: List[str] = ...) -> Result[str, Exception]: ...
        def stream(self, prompt: str, stop: List[str] = ...) -> AsyncIterator[Result[str, Exception]]: ...
    ```
* [ ] **Model Factory:** Implement `load_local_model(path: str) -> Result[InferenceEngine, Exception]` in `rai.inference.factory`.
    *   Dispatch based on file signature/extension:
        *   `.vmfb` -> IREE Engine
        *   `.gguf` -> Llama.cpp Engine
        *   `.onnx` -> ONNX Engine
* [ ] **IREE Engine:** Implement `IreeEngine` adapter using `iree-runtime`.
* [ ] **Llama.cpp Engine:** Implement `LlamaCppEngine` wrapper for `llama-cpp-python`.

#### Phase 2: Framework Bridges
*Goal: Make the engines usable by the existing Agents in `rai.adapters`.*

* [ ] **Agno Bridge:** Implement `LocalAgnoModel` in `rai.inference.bridges.agno`.
    *   It should accept an `InferenceEngine` instance in its constructor.
    *   Implementation of `arun` and `astream` delegates to the engine's `generate/stream`.
* [ ] **Pydantic AI Bridge:** Implement `LocalPydanticModel` in `rai.inference.bridges.pydantic_ai`.

#### Phase 3: Tools & Structured Output
*Goal: Enable "Agentic" capabilities on local models that don't support function calling natively.*

* [ ] **Prompt Injector:** Create a functional utility that converts a list of `Tool` definitions into a system prompt snippet (e.g., "You have access to these tools...").
* [ ] **Output Parser:** Implement a robust regex/grammar-based parser to detect when the model attempts to call a tool (e.g., `[TOOL_CALL: ...]`).
* [ ] **Grammar Enforcement (GGUF):** For Llama.cpp, expose `gbnf` grammar support to strictly force valid JSON output for tool calls.

#### Phase 4: User Interface & Telemetry
*Goal: Visualize the performance of local inference.*

* [ ] **Live Metrics:** Integrate `Rich.Live` to show:
    *   Tokens/sec (Generation speed).
    *   Time to First Token (TTFT).
    *   Memory/VRAM usage (if accessible).
* [ ] **Model Manager CLI:** `rai models list`, `rai models pull` (wrapping `huggingface-cli` or similar).

#### Phase 5: Distribution
* [ ] **Optional Dependencies:** Configure `pyproject.toml` with `extras` arrays (e.g., `rai[iree]`, `rai[llama]`).

---

### Implementation Strategy & File Structure

The new modules will be placed in `src/rai/inference/`:

```
src/rai/inference/
├── __init__.py
├── protocols.py       # InferenceEngine Protocol
├── factory.py         # load_local_model logic
├── engines/
│   ├── iree.py        # IREE implementation
│   ├── llama.py       # Llama.cpp implementation
│   └── onnx.py        # ONNX implementation
└── bridges/
    ├── agno.py        # Agno Model adapter
    └── pydantic.py    # PydanticAI Model adapter
```

> **Note:** The `rai.adapters.agno.AgnoAdapter` will utilize `rai.inference.bridges.agno.LocalAgnoModel` when a local model is configured, preserving the high-level Agent API.
