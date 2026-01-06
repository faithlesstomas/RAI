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

If you're looking for a place to start, check our TODO.md. Current priorities include:

  * Session Management: Fixing the session_id propagation in execution.py.
  * Resource Cleanup: Investigating and fixing unclosed database/transport warnings on exit.
  * New Adapters: Implementing adapters for LangChain or LangGraph.
  * WebUI: Enhancing the current nicegui implementation.

### Architecture Guidance
  * Adapters: If you are adding a new AI framework, look at src/rai/adapters/base.py for the required interface and src/rai/engine.py for how they are discovered.
  * Tools: System-level tools should be added to src/rai/tools/ and registered in src/rai/core.py.

*** Thank you for helping us build the best AI assistant for the Linux ecosystem! ***