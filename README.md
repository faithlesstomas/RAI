# Rich AI (RAI)
> **Pronunciation:** /raɪ/ - rhymes with **"sky"** or **"rye"** (or "paradise" in polish :) )

### The expert-grade AI agent orchestrator and private assistant - your local agentic paradise!

**RAI** is a powerful, modular AI assistant and agent manager designed specifically for power users, researchers, and developers who needs privacy (or just want to have local AI assistant on their server/desktop OS). It bridges the gap between sophisticated LLMs and your local system environment through a robust client-server architecture and a pluggable adapter system.


## Key Features

* **Expert Orchestration:** Unlike simple chat interfaces, RAI uses an "Engine" to route tasks to specialized framework adapters (currently supporting **Agno** and **Pydantic AI**).
* **Linux-First Integration:** Deep native support for the Linux desktop, including GNOME notifications, system screenshots, and shell execution (but there's plan for more).
* **Client-Server Flexibility:** Run RAI as a standalone CLI or as a background server accessible via REST and/or WebSockets from any client (examples in Python, GNU Guile and Emacs ELisp).
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

### **❤️ Support this project**

RAI is an independent open-source project developed with passion for the Linux and AI communities. If you find RAI useful and would like to support its further development, scientific research, and my work as a creator, there are several ways you can help:

#### **Direct Support**

* **GitHub Sponsors:** https://github.com/sponsors/faithlesstomas (For developers)
* **Ko-fi:** https://ko-fi.com/faithlesstomas (Buy me a coffee for a quick "thank you")
* **Patreon:** [TBD] (For long-term monthly support and community perks)

#### **Cryptocurrency**

For those who prefer decentralized support or wish to contribute directly without intermediaries, you can use the following addresses:

* **Bitcoin (BTC):** `bc1quwm8746cunecarqdhsqm2fx8cu2mw3sggfdugu`
* **Ethereum (ETH):** `0xAD78139a15D98c55D8f1a2d48091C727B32CF147`
* **Solana (SOL):** `FDZ6J3NGB4FQGrpBYWNihfAc9m2QZ7AznupDfYqeSJ5N`

#### **Other Ways to Help**

Not in a position to donate? You can still help RAI grow:

* **Star the repository** on GitLab/GitHub.
* **Contribute code** or fix bugs (see [CONTRIBUTING.md](https://www.google.com/search?q=CONTRIBUTING.md)).
* **Share the project** with your fellow Linux power users.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.