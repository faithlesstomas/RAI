# TODO List

- [ ] Investigate and fix `ResourceWarning: unclosed database` and `ResourceWarning: unclosed transport` that appear on application exit.
      This might be related to `prompt_toolkit` history or other network clients not being closed properly.
- [ ] Verify that the `--debug` flag correctly controls the verbosity of logs in the CLI and that the overall CLI logging system
      is functioning as expected, especially with the new `config_manager` and `agno` logger integration.
- [x] Implement `rai` as standalone by default and `rai connect` for client mode. (This is mostly done, but with some caveats)

## Refactor CLI

- [ ] **`rai connect` UDS connection:** The `connect` command currently only connects via HTTP.
      The original plan was to first try a Unix Domain Socket (UDS) and fall back to HTTP. This needs to be implemented or the plan updated.
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
-   **Fix CI Linter Environment:** The CI pipeline is failing during the `pylint` stage due to missing
    dependencies (`agno.storage.sqlite`, `pydantic_ai`, `pydbus`, `gi.repository`).
    This indicates a discrepancy between the local development environment and the CI environment.

    **Proposed Solution:**

    1.  **Create a `requirements-dev.txt` file:** Create a `requirements-dev.txt` file that lists all the dependencies needed for development and testing,
        including `pylint`, `pytest`, `ruff`, and any other packages needed to run the full test suite and linters.
    2.  **Update `pyproject.toml`:** Remove the `[project.optional-dependencies]` for `test` and `lint` from `pyproject.toml`.
        The `[project.dependencies]` should only contain the packages needed to run the application, not for development.
    3.  **Update `.gitlab-ci.yml`:** In the `before_script` of the CI jobs, install the dependencies from `requirements-dev.txt` using `pip install -r requirements-dev.txt`.
    4.  **Update Local Environment:** Your local development environment should also be updated to use the `requirements-dev.txt` file.

    This approach ensures that the local and CI environments are using the exact same dependencies, which should resolve the `pylint` errors.
    It also separates the application's dependencies from the development dependencies, which is a good practice.
