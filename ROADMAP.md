# RAI: Agentic OS Daemon & Secure Linux Gateway
## Project Roadmap & Evolution Plan

This document establishes the single source of truth for the strategic vision, architectural pivot, and tactical implementation phases for **RAI**. It rationalizes and merges the goals in `DEVELOPMENT.md`, `local_inference_plan.md`, `guile_hoot_webui_demo_plan.md`, and the **Antigravity SDK Pivot proposal**.

---

## 1. The Vision: From Orchestrator to Agentic OS Daemon

Initially, RAI was conceived as a framework-agnostic LLM orchestrator. However, trying to unify distinct asynchronous pydantic loops and execution flows (such as Agno/Phidata and Pydantic AI) under a custom unified `Processor` interface (`src/rai/engine.py`) led to continuous refactoring and high technical debt.

### The Pivot
Instead of recreating existing orchestration frameworks (like `liteLLM` or custom multi-framework wrappers), **RAI is pivoting to become an Agentic OS Daemon & Secure Gateway for Linux system environments, powered by the Google Antigravity SDK.**

```
┌────────────────────────────────────────────────────────────────────────┐
│                          CLIENT APPLICATION LAYER                      │
│      [ Emacs Client ]    [ GNU Guile Dashboard (WASM) ]    [ CLI ]     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / WebSockets (JSON API)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        RAI SYSTEM DAEMON (`rai serve`)                 │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │               Antigravity SDK AgentRuntime                     │   │
│   │    - Session management, routing, & history delegated to SDK   │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │ Injects secure context & tools
│                                   ▼
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                  RAI Secure Context & Tool Layer               │   │
│   │    - Custom system tools registered via MCP or SDK decorators │   │
│   └───────────────┬───────────────────────────────┬────────────────┘   │
└───────────────────┼───────────────────────────────┼────────────────────┘
                    │                               │
                    ▼ DBus / IPC                    ▼ Bubblewrap / Guix Container
┌──────────────────────────────────────┐ ┌───────────────────────────────┐
│        DESKTOP ADAPTER LAYER         │ │      SECURE SHELL RUNTIME     │
│ [ GNOME (pydbus) ] [ COSMIC (Rust) ] │ │ (Isolated Command Execution)  │
└──────────────────────────────────────┘ └───────────────────────────────┘
```

By delegating session routing, memory, and model communication to the **Google Antigravity SDK (`AgentRuntime`)**, RAI frees itself from maintaining core AI execution code. Instead, RAI focuses entirely on providing:
1. **Deep Desktop & D-Bus Integrations** (GNOME, COSMIC).
2. **Secure System Sandboxing** (Bubblewrap, Guix containers for arbitrary command execution).
3. **Multi-client Ecosystems** (CLI, Emacs, GNU Guile/Hoot WASM).

---

## 2. Core Architectural Changes

### 2.1 Deprecation of `engine.py` & Custom Adapters
*   **Remove:** `src/rai/engine.py` and the `src/rai/adapters/` directories.
*   **Replace:** Core agent loops will use the native `AgentRuntime` from the `google-antigravity` package.
*   **Impact:** Massive code reduction and elimination of complex `returns` monadic pipelines for LLM wrappers.

### 2.2 Reorganization of System Tools (MCP)
*   **Standardization:** Migrate from the hardcoded `TOOL_REGISTRY` in `core.py` to:
    *   Functions decorated with `@google.antigravity.tool` for direct runtime usage.
    *   An independent **Model Context Protocol (MCP)** server setup, allowing any external agent (including Antigravity CLI) to query system information.
*   **Desktop Modularization:** Move from a flat `gnome.py` to a structured desktop adapter package:
    *   `src/rai/tools/desktop/base.py`: Abstract class for system capabilities (e.g. notifications, screenshotting, window queries).
    *   `src/rai/tools/desktop/gnome.py`: GNOME-specific DBus calls using `pydbus`.
    *   `src/rai/tools/desktop/cosmic.py`: Wayland/DBus integration for the COSMIC Desktop.

### 2.3 Separate Agents and Sessions
*   **Agents (Templates):** Define "who" the agent is. Stored in `~/.config/rai/agents.yaml`.
*   **Sessions (Histories):** Store active interactions. Stored in a local database (SQLite/AioSQLite) or `~/.config/rai/sessions/*.json`.

---

## 3. Milestones & Phases

### Phase 1: Architectural Pivot & SDK Integration (Current Focus)
*   [ ] **Dependency Clean-up:** Update `pyproject.toml` to deprecate `agno` and `pydantic-ai` from the main dependency tree; add `google-antigravity`.
*   [ ] **Deprecate `engine.py`:** Delete the custom multi-adapter abstraction.
*   [ ] **Server Adaptation:** Rewrite `src/rai/server.py` and `routers/execution.py` to use `google-antigravity` runtime for execution and streaming.
*   [ ] **Implement Agent/Session Split:** 
    *   Write config migration logic to separate templates (`agents.yaml`) from histories (`sessions/`).
    *   Update CLI commands (`rai agents` and `rai sessions`) to reflect this change.
*   [ ] **Documentation Overhaul:** Update the main `README.md` to reflect the new Agentic OS Daemon & Secure Linux Gateway vision, new setup requirements, and Antigravity SDK integration.
*   [ ] **Repository Markdown Consolidation:** Clean up, consolidate, and archive/delete obsolete or redundant `.md` files (such as `DEVELOPMENT.md` and previous planning files) to prevent developer confusion.

### Phase 2: Linux Desktop & D-Bus Abstraction
*   [ ] **Create Desktop Abstraction:** Implement `src/rai/tools/desktop/base.py` defining standard signatures for notifications, focus tracking, and UI interactions.
*   [ ] **Refactor GNOME Tool:** Move `src/rai/tools/gnome.py` to `src/rai/tools/desktop/gnome.py` inheriting from the new base class.
*   [ ] **COSMIC Desktop Support:** Implement `src/rai/tools/desktop/cosmic.py` talking to COSMIC applets and its Rust/Wayland-based IPC.
*   [ ] **Environment Auto-detection:** Add desktop detection using the `XDG_CURRENT_DESKTOP` env variable, loading the correct subclass dynamically at startup.

### Phase 3: The Security Layer (Isolated Commands)
*   [ ] **Sandbox Runtime:** Implement a secure runner inside `src/rai/tools/security/`.
*   [ ] **Bubblewrap Integration:** Wrap shell command execution within a read-only bubblewrap (`bwrap`) container, allowing write access only to designated temporary folders.
*   [ ] **Guix Alternative:** Provide a fallback option for `guix shell --container` environments to guarantee reproducible, isolated tool executions.
*   [ ] **Safety Guardrails:** Implement high-level system prompt checks and verification constraints on the server before executing potentially destructive command outputs.

### Phase 4: Multi-Client Ecosystem
*   [ ] **CLI Stabilization:** Standardize CLI output with beautiful options (`--table` / `--details` / `--json`) and clean interactive prompts.
*   [ ] **GNU Guile Dashboard (WASM):** Complete the dashboard inside `guile-dashboard/` using Guile Hoot. Connect the Scheme frontend to the FastAPI server endpoints.
*   [ ] **Emacs Integration:** Develop a lightweight Emacs package (`rai.el`) interacting with the `rai serve` daemon over HTTP/WebSockets.
*   [ ] **NiceGUI Playground:** Redesign the `webui/` playground to serve as a visualization tool for active Antigravity sessions and active agent configurations.

### Phase 5: High-Performance Local Inference (Backlog)
*   [ ] **Local Inference Protocol:** Expose high-performance local runtimes (llama.cpp, ONNX, and IREE/MLIR) for fully offline operation.
*   [ ] **Engine Factories:** Build `src/rai/inference/` loading `.vmfb`, `.gguf`, or `.onnx` files, using functional adapters to wrap them for the Antigravity runner.
*   [ ] **Grammar Enforcement:** Integrate GBNF grammar constraints to force structured JSON output from raw local LLMs.

---

## 4. Technical Debt & Outstanding Bugs

### 4.1 High Priority Fixes
*   [ ] **Session ID Leak in Routing:** In `src/rai/routers/execution.py`, fix `execute_chain` (or the new runtime execution endpoint) to respect and propagate `session_id`, rather than creating new sessions per call.
*   [ ] **Resource Cleanup:** Fix the `ResourceWarning: unclosed database/transport` warnings that occur during daemon exit.
*   [ ] **Config Discrepancy:** Unify `rai config` CLI outputs so that active runtime config and stored `config.json` remain in sync.

### 4.2 CI/CD and Linting
*   [ ] **CI Linter Environment:** Fix `pylint` failures in GitLab CI by ensuring optional dependencies are installed via `pip install -e .[test,dev,lint]` in the `.gitlab-ci.yml`.

---

## 5. Development Guidelines for Agents

When contributing to RAI, all developer agents MUST adhere to these rules:
1.  **Commit Messages:** Always write git commit messages in **English**.
2.  **Environment:** Always use `uv` and ensure `.venv` is activated.
3.  **Verification:** Always run `pytest` and check test coverage on completion.
4.  **Linting:** Verify code sanity before submission using `ruff check src` and `pylint -E src`.
5.  **Functional Consistency:** Respect the functional programming elements (such as `returns` or explicit protocols/interfaces) where they form the codebase's core design.
