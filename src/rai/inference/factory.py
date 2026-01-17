"""
Factory for creating InferenceEngine instances based on model files.
"""

from pathlib import Path
from typing import Optional

from returns.result import Result, Failure, Success, safe

from .protocols import InferenceEngine
# Note: Actual engine imports will be lazy to avoid hard dependencies


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
    backend: Optional[str] = None
) -> Result[InferenceEngine, Exception]:  # pylint: disable=too-many-return-statements
    """
    Loads a local model and returns an InferenceEngine.
    
    Dispatches to the correct engine based on:
    1. Explicit 'backend' argument (if provided).
    2. File extension (.gguf -> llama, .vmfb -> iree).
    
    Args:
        model_path: Path to the model file.
        backend: Optional explicit backend name ('llama', 'iree', 'onnx').
        
    Returns:
        Result[InferenceEngine, Exception]: The loaded engine or a Failure.
    """
    
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

    # 3. Dispatch (Lazy Loading)
    try:
        if target_backend == "llama":
            from .engines.llama import LlamaCppEngine  # noqa: PLC0415
            return Success(LlamaCppEngine(str(path)))
        
        elif target_backend == "iree":
            from .engines.iree import IreeEngine  # noqa: PLC0415
            return Success(IreeEngine(str(path)))
            
        elif target_backend == "onnx":
            # from .engines.onnx import OnnxEngine
            # return Success(OnnxEngine(str(path)))
            return Failure(NotImplementedError("ONNX backend not yet implemented"))
            
        else:
            return Failure(ValueError(f"Unsupported backend: {target_backend}"))
            
    except ImportError as e:
        return Failure(ImportError(f"Missing dependency for {target_backend}: {e}"))
    except Exception as e:
        return Failure(e)
