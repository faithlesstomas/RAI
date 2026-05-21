"""
Service for managing and discovering LLM models from various backends.
"""
import logging
import os
from typing import Dict, List, Any, Optional

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore

class ModelRegistry:
    """
    Registry for managing connections to LLM backends and retrieving available models.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._ollama_client: Optional[Any] = None

    def _get_ollama_client(self) -> Any: # noqa: ANN401
        """Lazy initialization of the Ollama client."""
        if self._ollama_client:
            return self._ollama_client

        if not ollama:
            logging.warning("Ollama package not installed.")
            return None

        # Prioritize environment variable, then session config, then default
        host = os.getenv("OLLAMA_HOST")
        if not host:
            active_session_name = self.config.get("active_session", "default")
            session_config = self.config.get("sessions", {}).get(active_session_name, {})
            host = session_config.get("ollama_host", "http://127.0.0.1:11434")

        logging.debug("Initializing Ollama client with host: %s", host)
        self._ollama_client = ollama.AsyncClient(host=host)
        return self._ollama_client

    async def get_models(self, backend: str) -> List[str]:
        """
        Retrieves a list of available models for a specific backend.
        """
        if backend == "ollama":
            return await self._get_ollama_models()
        if backend == "gemini":
            return await self._get_gemini_models()

        # Placeholder for other backends
        logging.warning("Backend '%s' not yet supported for model listing.", backend)
        return []

    async def _get_ollama_models(self) -> List[str]:
        """Fetches models from the Ollama backend."""
        client = self._get_ollama_client()
        if not client:
            return []

        try:
            response = await client.list()
            # Handle both object-style and dict-style responses from the library
            models = response.get("models", [])
            return [m.get("model") for m in models]
        except Exception as e: # pylint: disable=broad-exception-caught
            logging.error("Failed to fetch models from Ollama: %s", e)
            return []

    async def _get_gemini_models(self) -> List[str]:
        """Fetches models from the Google Gemini backend."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logging.warning("GOOGLE_API_KEY or GEMINI_API_KEY not set. Cannot fetch Gemini models.")
            return []

        try:
            if not genai:
                raise ImportError("google-genai package not installed.")
            
            client = genai.Client(api_key=api_key)
            models = []
            for m in client.models.list():
                name = m.name or ""
                if name.startswith("models/"):
                    name = name.replace("models/", "")
                if "gemini" in name.lower() or "text" in name.lower():
                    models.append(name)
            return models
        except ImportError:
            logging.warning("google-genai package not installed.")
            return []
        except Exception as e: # pylint: disable=broad-exception-caught
            logging.error("Failed to fetch models from Gemini: %s", e)
            return []

    async def get_all_models(self) -> Dict[str, List[str]]:
        """
        Retrieves models from all supported backends.
        """
        backends = ["ollama", "gemini"] # Add "openai", "anthropic" etc. as they are implemented
        results = {}
        for backend in backends:
            models = await self.get_models(backend)
            if models:
                results[backend] = models
        return results
