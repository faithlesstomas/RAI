# RAI Development Plan

This document serves as the single source of truth for the project's vision, roadmap, and active tasks, consolidating the previous `ROADMAP.md` and `TODO.md`.

## 1. Vision & Architecture

### Agents vs. Sessions
Currently, `rai` uses "sessions" to store both the configuration of an AI assistant (model, system prompt, tools) and the state of the interaction. This conflates the identity of the assistant ("Agent") with the interaction instance ("Session"). The goal is to separate these concepts.

#### Agents (Templates)
*   **Definition:** A named configuration defining *who* the AI is.
*   **Storage:** `~/.config/rai/agents.yaml`
*   **Attributes:** `name`, `model`, `backend`, `system_prompt`, `tools`, `description`, `avatar`.

#### Sessions (Instances)
*   **Definition:** A stateful interaction history with a specific Agent.
*   **Storage:** `~/.config/rai/sessions/*.json`
*   **Attributes:** `id`, `agent_ref`, `history`, `overrides`, `created_at`.

### CLI Consistency & UX
*   **Output Formats:** Minimal (default), `--table`/`--details` (human), `--json` (machine).
*   **Command Structure:** Symmetry between `agents` and `sessions` subcommands.

### WebUI Alignment
*   **Agents Page:** Editor for Agent templates.
*   **Chat/Playground:** Interface for active Sessions.

---

## 2. Milestones

### v0.2.0: Stable CLI & Adapter Protocol (Current Focus)
*   Stabilize the core CLI loop and error handling.
*   Standardize the `Processor` protocol for all adapters.
*   Ensure dependencies are correctly managed for development and testing.

### v0.3.0: Agents & Sessions Separation
*   Implement the configuration migration from `sessions` to `agents`.
*   Update CLI commands (`rai agents`, `rai sessions`).
*   Update Server API to handle the separated logic.

---

## 3. Active Sprint / Immediate Tasks

### Stabilization & Hotfixes (Completed)
- [x] **Processor Definition**: Introduce `rai.adapters.base.Processor` protocol (`arun`, `astream`, `reload`, `close`).
- [x] **Logging Fix**: Fix formatting error in `core.py`.
- [x] **CLI Error Handling**: Add try-except blocks in interactive loop (partially addressed by protocol fix, further testing needed).
- [x] **WebSocket Reload**: Implement `reload()` in `WebSocketAdapter`.
- [x] **Dependencies**: Organize `pyproject.toml` (move `tools` to `dev`, cleanup `test`).

### CLI Refactoring & Features
- [ ] **`rai connect` UDS connection:** Implement fallback to Unix Domain Socket if HTTP fails (or prioritized).
- [ ] **Tests:** Add tests for standalone mode, client mode, and slash commands (`/history`, `/save`, `/clear`).
- [ ] **`/clear` command:** Implement client-side calls in `WebSocketAdapter` to trigger history clear on server. (Server-side is [x]).

### Architecture Impl (Next Steps)
- [ ] Migrate `app_config["sessions"]` to new `agents` structure.
- [ ] Update `rai agents list` to read from `agents.yaml`.
- [ ] Unify `/config set`:
    - Standalone: Update local Agent config.
    - Connect: PATCH request to update Session.

---

## 4. Technical Debt & Bugs

### Bugs
- [ ] **Critical:** Fix `execute_chain` in `routers/execution.py` to pass `session_id`. Currently ignores it, creating new sessions per request.
- [ ] Investigate `ResourceWarning: unclosed database/transport` on exit.
- [ ] Verify `--debug` flag effectiveness with new logger integration.

### Code Cleanup
- [x] Remove `TEST COMMENT` from `core.py`.
- [ ] **CI Linter Env:** CI fails `pylint` due to missing deps.
    - [ ] Update `.gitlab-ci.yml` to install `.[dev]` (instead of relying on a separate requirements file).

### Guile Dashboard
- [ ] Investigate missing IP addresses in logs (showing as `-`).
- [ ] Implement numeric `LOG_LEVEL` support.

---

## 5. Backlog / Future Ideas

- [ ] **LangChain/LangGraph Adapter**: Support for LC agents.
- [ ] **Pydantic AI Adapter**: Finish implementation (streaming, tools).
- [ ] **Network Stack**: Migrate client networking to `aiohttp` to unify HTTP/WS handling.
