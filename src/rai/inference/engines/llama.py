"""
Llama.cpp implementation of InferenceEngine.
"""
from typing import List, Optional, AsyncIterator
import functools

from returns.result import Result, Success, Failure, safe
from llama_cpp import Llama

from ..protocols import InferenceEngine, InferenceResult, GenerationStats

class LlamaCppEngine(InferenceEngine):
    """
    Adapter for llama-cpp-python.
    """
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        # Initialize Llama model
        # verbose=False to keep stdout clean
        # n_ctx=0 means use model's default context (or set 2048/4096)
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048, # Safe default
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Llama model from {model_path}: {e}")

    @safe
    def generate(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> InferenceResult:
        """
        Synchronous generation.
        """
        output = self.llm.create_completion(
            prompt=prompt,
            stop=stop or [],
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False
        )
        
        # Parse output
        text = output['choices'][0]['text']
        usage = output.get('usage', {})
        
        # Basic stats if available
        stats = GenerationStats(
            input_tokens=usage.get('prompt_tokens', 0),
            output_tokens=usage.get('completion_tokens', 0),
            total_time_sec=0.0, # Llama output doesn't give time directly easily without calculating
            tokens_per_sec=0.0 
        )
        
        return InferenceResult(text=text, stats=stats, finish_reason=output['choices'][0]['finish_reason'])

    async def stream(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> AsyncIterator[Result[str, Exception]]:
        """
        Asynchronous stream wrapper.
        Since Llama.cpp is sync/blocking (unless we use Llama.generate which is a generator),
        we iterate the generator. For async, we technically block the loop between tokens 
        unless we run in thread, but here we just yield.
        """
        try:
            stream = self.llm.create_completion(
                prompt=prompt,
                stop=stop or [],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            for chunk in stream:
                 text = chunk['choices'][0]['text']
                 yield Success(text)
        
        except Exception as e:
            yield Failure(e)

    def unload(self) -> None:
        if hasattr(self, 'llm'):
            del self.llm
        pass
