# Contributing to RAI

Thank you for considering contributing to RAI! It’s people like you who make Open Source a great place to learn, inspire, and create.

RAI is an expert-grade AI assistant for Linux, and we welcome contributions from developers and researchers of all skill levels.
How Can I Contribute?

## Reporting Bugs
  * Check the Issues to see if the bug has already been reported.
  * If not, open a new issue. Include a clear title, a description of the problem, and steps to reproduce it.

## Suggesting Enhancements
  * We are always looking for new Adapters (e.g., LangChain, AutoGPT) and Tools (system integrations).
  * Open an issue with the "enhancement" label to discuss your idea.

## Pull Requests / Merge Requests
  1. Fork the repository.
  1. Create a new branch for your feature or bugfix.
  1. Write tests for your changes.
  1. Ensure the CI pipeline passes (linting and existing tests).
  1. Submit your request!

## Development Setup

### To start developing on RAI, follow these steps:
  1. Clone and Install:
     ```bash
     git clone https://gitlab.com/tk-lab1/ai/rai.git
     cd rai
     pip install -e .[test,dev,lint]
     ```
  1. Environment Variables: Copy .env.example to .env and add your development API keys.
  1. Code Style: We use ruff and pylint to maintain code quality. Please run them before submitting your code:
     ```bash
     ruff check .
     pylint src/rai
     ```
  1. Testing: We use pytest and pytest-asyncio. Run the full suite with:
     ```bash
     pytest
     ```

## Technical Focus Areas (What we need help with)

If you're looking for a place to start, check our [ROADMAP.md](ROADMAP.md). Current priorities include:

  * Agent & Session separation implementation.
  * Desktop Integrations: GNOME and COSMIC modules in `src/rai/tools/desktop/`.
  * Security Sandboxing: Isolated bubblewrap/guix command execution.
  * Multi-client frontends (Emacs, GNU Guile WASM dashboard).

### Architecture Guidance
  * Tools & Environment: System-level tools and adapters are built as standalone, secure utilities. Desktop-specific tools belong in `src/rai/tools/desktop/` and inherit from the base desktop class to ensure dynamic compatibility across Linux desktop environments.

*** Thank you for helping us build the best AI assistant for the Linux ecosystem! ***