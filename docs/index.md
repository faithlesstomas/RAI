# Welcome to Rich AI – The Intelligent Agent Ecosystem for Linux

**Rich AI (RAI)** is an open-source project designed to bridge the gap between powerful LLMs and the everyday POSIX workflow. Far more than a simple API wrapper, RAI is a modular client-server platform built for users who demand more than just text generation from their AI.

## Core Pillars of the Project:

*   **Agentic Architecture**: Using a pluggable adapter system, RAI allows you to "hire" different frameworks for specific tasks. Need a web-searching researcher? Use the Agno adapter. Building precise, type-safe business logic? Deploy Pydantic AI.

*   **Deep OS Integration**: RAI is a first-class citizen of the Linux world. It can trigger desktop notifications, capture screenshots for visual analysis, and manage files—all under your explicit permission and control.

*   **Built for Researchers & Devs**: The project supports advanced "Chains," enabling you to link multiple specialized agents into powerful information-processing pipelines.

*   **True Interoperability**: The RAI server communicates via an asynchronous API, allowing you to build custom clients in any language.

## The Future of Expert AI:

We are currently in a high-velocity development phase, moving toward a complete separation of Agent definitions and Session instances to provide a truly stateful AI experience. Join us in defining what a professional AI assistant looks like in 2026.

---

## TL;DR;

### **Quick Start Guide**

Get RAI up and running on your Linux machine in a few simple steps.

#### **1. Prerequisites**

* **Python:** version 3.10 or higher.
* **Local LLM (Optional):** Install [Ollama](https://ollama.com/) if you want to run models locally.

#### **2. Installation**

Clone the repository and install the package in editable mode:

```bash
git clone https://gitlab.com/tk-lab1/ai/rai.git
cd rai
pip install -e .

```

*Note: For GNOME integration (notifications, screenshots), use:* `pip install -e .[gnome-tools]`.

#### **3. Configuration**

Set up your environment variables. Create a `.env` file in the project root:

```bash
cp .env.example .env

```

Edit `.env` and add your API keys (e.g., `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`) or set your `OLLAMA_HOST`.

#### **4. Basic Usage**

**A. Standalone Mode (Direct CLI)**
Interact with the AI directly from your terminal:

```bash
rai -p "Explain the Linux kernel in one sentence"

```

**B. Client-Server Mode**

1. **Start the server:**
```bash
rai serve

```


2. **Connect with the client:**
```bash
rai --connect

```



#### **5. Running Tests**

Ensure everything is set up correctly by running the test suite:

```bash
pip install -e .[test]
pytest

```

---

```{toctree}
:maxdepth: 2
:caption: Contents:

reference/modules
```
