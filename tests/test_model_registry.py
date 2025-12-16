"""
Tests for the ModelRegistry service.
"""
import os
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from rai.services.model_registry import ModelRegistry

@pytest.fixture
def mock_config():
    return {
        "active_session": "default",
        "sessions": {
            "default": {
                "ollama_host": "http://test-host:11434"
            }
        }
    }

@pytest.mark.asyncio
async def test_get_ollama_client_env_var(mock_config):
    """Test that OLLAMA_HOST env var takes precedence."""
    with patch.dict(os.environ, {"OLLAMA_HOST": "http://env-host:11434"}):
        with patch("rai.services.model_registry.ollama") as mock_ollama:
            registry = ModelRegistry(mock_config)
            # Trigger client creation
            await registry.get_models("ollama")
            
            mock_ollama.AsyncClient.assert_called_with(host="http://env-host:11434")

@pytest.mark.asyncio
async def test_get_ollama_client_config_fallback(mock_config):
    """Test fallback to config when OLLAMA_HOST is not set."""
    # Ensure env var is not set
    with patch.dict(os.environ, {}, clear=True):
        with patch("rai.services.model_registry.ollama") as mock_ollama:
            registry = ModelRegistry(mock_config)
            await registry.get_models("ollama")
            
            mock_ollama.AsyncClient.assert_called_with(host="http://test-host:11434")

@pytest.mark.asyncio
async def test_get_ollama_models_success(mock_config):
    """Test successful retrieval of Ollama models."""
    with patch("rai.services.model_registry.ollama") as mock_ollama:
        mock_client = AsyncMock()
        mock_client.list.return_value = {
            "models": [
                {"model": "llama2"},
                {"model": "mistral"}
            ]
        }
        mock_ollama.AsyncClient.return_value = mock_client
        
        registry = ModelRegistry(mock_config)
        models = await registry.get_models("ollama")
        
        assert models == ["llama2", "mistral"]

@pytest.mark.asyncio
async def test_get_gemini_models_success(mock_config):
    """Test successful retrieval of Gemini models."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}):
        # Patch rai.services.model_registry.genai instead of sys.modules
        with patch("rai.services.model_registry.genai") as mock_genai:
            # Mock the list_models function
            model1 = MagicMock()
            model1.name = "models/gemini-pro"
            model1.supported_generation_methods = ["generateContent"]
    
            model2 = MagicMock()
            model2.name = "models/embedding-001"
            model2.supported_generation_methods = ["embedContent"]
    
            mock_genai.list_models.return_value = [model1, model2]

            registry = ModelRegistry(mock_config)
            models = await registry.get_models("gemini")
             
            assert models == ["gemini-pro"]

@pytest.mark.asyncio
async def test_get_all_models(mock_config):
    """Test aggregation of models from all backends."""
    with patch.object(ModelRegistry, "get_models") as mock_get_models:
        mock_get_models.side_effect = lambda backend: {
            "ollama": ["llama2"],
            "gemini": ["gemini-pro"]
        }.get(backend, [])
        
        registry = ModelRegistry(mock_config)
        all_models = await registry.get_all_models()
        
        assert all_models == {
            "ollama": ["llama2"],
            "gemini": ["gemini-pro"]
        }
