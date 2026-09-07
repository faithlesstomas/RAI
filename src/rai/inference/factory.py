from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from returns.result import Failure, Result, Success, safe

from .engines.iree import is_iree_available
from .engines.llama import is_llama_cpp_available


def is_backend_available(backend: str) -> bool:
    """Check if a named local inference backend is operational."""
    b = backend.lower()
    if b == "ollama":
        try:
            import ollama  # noqa: PLC0415
            return True
        except ImportError:
            return False
    elif b == "llama":
        return is_llama_cpp_available()
    elif b == "iree":
        return is_iree_available()
    elif b == "onnx":
        return False
    return False


def get_available_backends() -> tuple[str, ...]:
    """Return all currently operational local inference backends."""
    backends: list[str] = []
    for candidate in ("ollama", "llama"):
        if is_backend_available(candidate):
            backends.append(candidate)
    return tuple(backends)


@safe
def _validate_path(model_path: str) -> Path:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not path.is_file():
        raise IsADirectoryError(f"Path is not a file: {model_path}")
    return path


def load_local_model(
    model_path: str,
    backend: Optional[str] = None,
) -> Result[Any, Exception]:
    """
    Loads a local model and returns an engine instance.
    """
    # If backend is explicitly ollama, we don't require model_path to be a local file path
    if backend and backend.lower() == "ollama":
        from .engines.ollama import OllamaEngine  # noqa: PLC0415
        return Success(OllamaEngine(model_name=model_path))

    # 1. Validate Path
    path_result = _validate_path(model_path)
    if isinstance(path_result, Failure):
        return path_result

    path = path_result.unwrap()  # pylint: disable=no-member
    extension = path.suffix.lower()

    # 2. Determine Backend
    target_backend = backend
    if not target_backend:
        if extension == ".gguf":
            target_backend = "llama"
        elif extension == ".vmfb":
            target_backend = "iree"
        elif extension == ".onnx":
            target_backend = "onnx"
        else:
            return Failure(ValueError(f"Could not infer backend for extension: {extension}"))

    return _load_local_model_cached(str(path), target_backend.lower())


@lru_cache(maxsize=4)
def _load_local_model_cached(
    model_path_str: str,
    backend: str,
) -> Result[Any, Exception]:
    """Cached worker for loading local models."""
    try:
        if backend == "llama":
            if not is_llama_cpp_available():
                return Failure(
                    ImportError(
                        "llama-cpp-python is not installed. Install with: uv sync --extra inference-llama"
                    )
                )
            from .engines.llama import LlamaCppEngine  # noqa: PLC0415
            return Success(LlamaCppEngine(model_path_str))

        elif backend == "ollama":
            from .engines.ollama import OllamaEngine  # noqa: PLC0415
            return Success(OllamaEngine(model_name=model_path_str))

        elif backend == "iree":
            return Failure(
                NotImplementedError(
                    "IREE backend is frozen in Stage 4 pending conformance tests and runtime availability"
                )
            )

        elif backend == "onnx":
            return Failure(NotImplementedError("ONNX backend is frozen pending conformance tests"))

        else:
            return Failure(ValueError(f"Unsupported backend: {backend}"))

    except ImportError as e:
        return Failure(ImportError(f"Missing dependency for {backend}: {e}"))
    except Exception as e:
        return Failure(e)

