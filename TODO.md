# TODO List

- [ ] Investigate and fix `ResourceWarning: unclosed database` and `ResourceWarning: unclosed transport` that appear on application exit. This might be related to `prompt_toolkit` history or other network clients not being closed properly.
- [ ] Verify that the `--debug` flag correctly controls the verbosity of logs in the CLI and that the overall CLI logging system is functioning as expected, especially with the new `config_manager` and `agno` logger integration.
- [x] Implement `rai` as standalone by default and `rai connect` for client mode. (This is mostly done, but with some caveats)

## Refactor CLI

- [ ] **`rai connect` UDS connection:** The `connect` command currently only connects via HTTP. The original plan was to first try a Unix Domain Socket (UDS) and fall back to HTTP. This needs to be implemented or the plan updated.
- [ ] **Testing:** The current test suite is insufficient after the CLI refactoring. New tests need to be added for:
    - [ ] `rai` standalone mode (one-shot and interactive).
    - [ ] `rai connect` client mode (one-shot REST and interactive WebSocket).
    - [ ] New slash commands: `/history`, `/save`, `/clear`, `/model`.
    - [ ] `rai config edit` command.
- [ ] Fix `/clear` slash command:
    - [x] Implement server-side history management for `WebSocketAdapter`.
    - [ ] Implement client-side calls in `WebSocketAdapter` to clear history on the server.

## Code Cleanup

- [x] Remove `TEST COMMENT` from `src/rai/core.py`.
