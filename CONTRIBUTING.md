# Contributing to Rich AI

RAI is evolving into a local-first agent runtime for Linux. Contributions should
strengthen a complete runtime boundary or vertical slice instead of introducing
another provider-specific orchestration layer.

Read [ROADMAP.md](ROADMAP.md), [docs/architecture.md](docs/architecture.md) and
[SECURITY.md](SECURITY.md) before changing runtime or execution code.

## Development setup

Use `uv` and the project virtual environment:

```bash
git clone https://gitlab.com/tk-lab1/ai/rai.git
cd rai
uv sync --extra test --extra lint
```

Install optional platform or inference dependencies only when needed:

```bash
uv sync --extra gnome-tools
uv sync --extra inference-llama
```

Do not commit model weights, API keys, access tokens, local databases or local
harness state. If a credential reaches a workspace file, rotate it; adding the
file to `.gitignore` is not a substitute for rotation.

## Development rules

- Prefer `Protocol` boundaries, immutable records and explicit `Result` values
  at I/O/backend boundaries.
- Keep operating-system state and durable memory outside LLM contexts.
- Treat model and remote-agent output as untrusted input.
- Security controls must fail closed.
- Avoid package-import side effects, global monkeypatches and hidden singleton
  construction.
- Add one backend through a conformance contract rather than branching core
  logic by provider name.
- Use English commit messages and documentation/code identifiers.

## Verification

Before opening a merge request:

```bash
uv run pytest --timeout=30 --cov=src/rai --cov-report=term
uv run ruff check src tests --select E9,F63,F7,F82
uv run pylint -E src/rai
```

Run full style diagnostics as well and avoid adding new violations:

```bash
uv run ruff check src tests
```

Tests must not use the developer's real XDG directories, network credentials or
desktop session. Integration tests should either prove a capability is usable or
skip with a precise platform reason.

## GitLab workflow

Use `glab` for issues, merge requests and CI inspection. An issue should state:

- the user-visible or architectural outcome,
- the boundary/contract affected,
- failure and security behavior,
- acceptance tests,
- documentation that must change.

Keep merge requests narrow enough to review, but complete enough to move the
selected vertical slice end-to-end.
