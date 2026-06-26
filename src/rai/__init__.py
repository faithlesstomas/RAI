"""
Rich AI (RAI) - assistant and expert agent orchestrator for the Linux ecosystem.
"""
# pylint: disable=no-member
import os
import logging
import subprocess
from typing import Any
from google.antigravity.connections.local import localharness_pb2
from google.antigravity.connections.local import local_connection

original_build_harness_config = local_connection.LocalConnectionStrategy._build_harness_config

def patched_build_harness_config(self) -> Any:
    harness_config = original_build_harness_config(self)
    model_name = self._gemini_config.models.default.name if self._gemini_config else None

    if model_name and not model_name.startswith("gemini"):
        logging.info(f"[RAI Patch] Routing model {model_name} to GemmaConfig")
        harness_config.ClearField("gemini_config")

        # Determine host dynamically from configuration or environment
        from rai.config_manager import load_config
        app_config = load_config()
        active_agent = app_config.get("active_agent") or "default"
        agent_cfg = app_config.get("agents", {}).get(active_agent, {})
        host = os.environ.get("OLLAMA_HOST") or agent_cfg.get("ollama_host") or "http://127.0.0.1:11434"

        logging.info(f"[RAI Patch] Using Ollama host: {host}")
        harness_config.gemma_config.CopyFrom(
            localharness_pb2.GemmaConfig(
                base_url=host,
                model_name=model_name
            )
        )
    return harness_config

local_connection.LocalConnectionStrategy._build_harness_config = patched_build_harness_config

# Set a fallback dummy GEMINI_API_KEY to satisfy __aenter__ check if not present in environment
if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = "dummy-key-for-local-model-execution"

# Monkeypatch subprocess.Popen to run localharness with an isolated HOME
original_popen = subprocess.Popen

def patched_popen(*args, **kwargs):
    if args and isinstance(args[0], list) and any("localharness" in str(x) for x in args[0]):
        env = kwargs.get("env") or os.environ.copy()
        # Create a isolated temp home directory inside user cache
        temp_home = os.path.expanduser("~/.cache/rai/temp_home")
        os.makedirs(temp_home, exist_ok=True)
        env["HOME"] = temp_home
        kwargs["env"] = env
    return original_popen(*args, **kwargs)

subprocess.Popen = patched_popen

__version__ = "0.2.0"
