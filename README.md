# RAI - Rich AI Assistant

**The Expert-Grade AI Agent Orchestrator for the Linux Ecosystem.**

RAI (Rich AI) is a powerful, modular AI assistant designed specifically for Linux power users, researchers, and developers. It bridges the gap between sophisticated LLMs and your local system environment through a robust client-server architecture and a pluggable adapter system.

---

## Key Features

* **Expert Orchestration:** Unlike simple chat interfaces, RAI uses an "Engine" to route tasks to specialized framework adapters (currently supporting **Agno** and **Pydantic AI**).
* **Linux-First Integration:** Deep native support for the Linux desktop, including GNOME notifications, system screenshots, and shell execution.
* **Client-Server Flexibility:** Run RAI as a standalone CLI or as a background server accessible via REST or WebSockets from any client (examples in Python, GNU Guile and Emacs ELisp).
* **Agnostic Backends:** Seamlessly switch between local models via **Ollama** and cloud-based providers like **Gemini**, **OpenAI**, or **Anthropic**.

---

## Architecture

RAI is built on a modular foundation:
* **Server (`rai serve`):** A FastAPI-based core handling AI logic, session state, and history.
* **Engine (`src/rai/engine.py`):** The brain that discovers and manages pluggable framework adapters.
* **Adapters:** Wrappers for specific AI frameworks. Currently implemented: `Agno` and `Pydantic AI`.
* **CLI Client:** A `click`-based terminal interface that supports both interactive chat and one-shot inference.

---

## Quick Start

### 1. Installation
Clone the repository and install with development dependencies:

```bash
git clone [https://gitlab.com/tk-lab1/ai/rai.git](https://gitlab.com/tk-lab1/ai/rai.git)
cd rai
pip install -e .[test,dev,lint]
```

*For GNOME tools integration:* `pip install -e .[gnome-tools]`.

### 2. Configuration

Create your configuration file in `~/.config/rai/config.json` or use environment variables via a `.env` file.
*TODO: rething configuration file*

### 3. Usage

* **Standalone Chat:** `rai`
* **One-shot Inference:** `rai -p "Analyze this log file: error.log"`
* **Run as Server:** `rai serve`
* **Connect Client to Server:** `rai --connect`.

---

## Documentation / API Reference

[Documentation](https://tk-lab1.gitlab.io/ai/rai/) is build with sphinx.

## Roadmap & Evolution

We are currently transitioning from a session-based configuration to a distinct **Agent vs. Session** model:

* **Agents:** Defined templates for specific expertise (e.g., Scientist, Coder, Researcher).
* **Sessions:** Stateful instances of interactions based on an Agent template.
* **Next Up:** LangChain/LangGraph adapters and an enhanced WebUI.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code style, development setup, and how to submit pull requests.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.