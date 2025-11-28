# RAI Project Roadmap

## 1. Architecture Evolution: Agents vs. Sessions

Currently, `rai` uses "sessions" to store both the configuration of an AI assistant (model, system prompt, tools) and the state of the interaction (though chat history is currently stored separately in `history.txt` or memory). This conflates the identity of the assistant ("Agent") with the interaction instance ("Session").

### Goal
Separate **Agents** (Templates/Definitions) from **Sessions** (Instances).

### Proposed Structure

#### Agents (Templates)
*   **Definition:** A named configuration defining *who* the AI is.
*   **Storage:** `~/.config/rai/agents.yaml` (or json)
*   **Attributes:**
    *   `name` (ID)
    *   `model`
    *   `backend`
    *   `system_prompt`
    *   `tools` (list of enabled tools)
    *   `description`
    *   `avatar` (optional, for UI)

#### Sessions (Instances)
*   **Definition:** A stateful interaction history with a specific Agent.
*   **Storage:** `~/.config/rai/sessions/` (one file per session, e.g., `session_id.json`)
*   **Attributes:**
    *   `id` (UUID or named)
    *   `agent_ref` (name of the Agent this session is based on)
    *   `created_at`
    *   `updated_at`
    *   `history` (list of messages)
    *   `overrides` (optional runtime overrides of agent config)

### CLI Implications
*   `rai agents list`: Lists available agent templates.
*   `rai session list`: Lists active/archived conversation sessions.
*   `rai session new --agent <agent_name>`: Starts a new session based on an agent.

## 2. CLI Consistency & UX

### Output Formats
*   **Default:** Minimal, grep-friendly output (simple lists).
*   **`--table` / `--details`:** Rich, formatted tables for human consumption.
*   **`--json`:** Machine-readable JSON for scripting.

### Command Structure
*   Ensure symmetry between `agents` and `sessions` subcommands (`list`, `show`, `create`, `delete`).

## 3. WebUI Alignment
*   The WebUI should reflect this separation:
    *   **Agents Page:** Editor for Agent templates.
    *   **Chat/Playground:** Interface for Sessions. A dropdown allows selecting which Agent to chat with (creating a new session).

## 4. Immediate Steps (Refactoring)
1.  [ ] Standardize CLI output for `rai agents list` to match `rai session list` (simple list by default).
2.  [ ] Add `--table` flag to `rai agents list` for the detailed view.
3.  [ ] Update `rai session list` to support `--table` and `--json` for consistency.
4.  [ ] Begin design of `Agent` vs `Session` separation in `config_manager.py`.
