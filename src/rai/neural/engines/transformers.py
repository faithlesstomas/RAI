"""Lazy, cancellable Hugging Face Transformers generation engine."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator

from ..artifacts import LensRegistry, ModelIdentity
from ..contracts import GenerationRequest, NeuralObservation, NcsiErrorCode, NcsiRuntimeError
from ..inspectors.jlens import JLensInspector
from .base import EngineDelta, ModelDescription


class TransformersEngine:
    """Own one causal language model for the lifetime of the neural sidecar."""

    def __init__(  # noqa: PLR0913
        self,
        model_id: str,
        model_revision: str,
        tokenizer_revision: str | None,
        lens_registry: LensRegistry,
        *,
        device: str = "auto",
        dtype: str = "auto",
        quantization: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision or model_revision
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        self.lens_registry = lens_registry
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_lock = asyncio.Lock()
        self._resolved_model_revision = model_revision
        self._resolved_tokenizer_revision = self.tokenizer_revision
        self.load_seconds: float | None = None
        self.last_peak_accelerator_bytes: int | None = None

    @property
    def description(self) -> ModelDescription:
        actual_device = str(getattr(self._model, "device", self.device))
        actual_dtype = str(getattr(self._model, "dtype", self.dtype)).removeprefix("torch.")
        return ModelDescription(
            model_id=self.model_id,
            model_revision=self._resolved_model_revision,
            tokenizer_revision=self._resolved_tokenizer_revision,
            loaded=self._model is not None,
            device=actual_device,
            dtype=actual_dtype,
        )

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            started = time.monotonic()
            try:
                model, tokenizer = await asyncio.to_thread(self._load_sync)
            except ImportError as exc:
                raise NcsiRuntimeError(
                    NcsiErrorCode.MODEL_LOAD_FAILED,
                    "install rai[inference-transformers] to use the neural sidecar",
                ) from exc
            except Exception as exc:
                if _is_out_of_memory(exc):
                    raise NcsiRuntimeError(
                        NcsiErrorCode.ACCELERATOR_OOM, "accelerator ran out of memory loading model"
                    ) from exc
                raise NcsiRuntimeError(
                    NcsiErrorCode.MODEL_LOAD_FAILED, f"unable to load configured model: {exc}"
                ) from exc
            self._model = model
            self._tokenizer = tokenizer
            self._resolved_model_revision = (
                getattr(model.config, "_commit_hash", None) or self.model_revision
            )
            self._resolved_tokenizer_revision = (
                getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
                or self.tokenizer_revision
            )
            self.load_seconds = time.monotonic() - started

    def _load_sync(self) -> tuple[Any, Any]:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        dtype: str | Any = self.dtype
        if self.dtype != "auto":
            dtype = getattr(torch, self.dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.tokenizer_revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            dtype=dtype,
            device_map=self.device,
        )
        model.eval()
        return model, tokenizer

    def _identity(self) -> ModelIdentity:
        config = getattr(self._model, "config", None)
        architectures = getattr(config, "architectures", None) or []
        architecture = architectures[0] if architectures else type(self._model).__name__
        return ModelIdentity(
            model_id=self.model_id,
            model_revision=self._resolved_model_revision,
            tokenizer_revision=self._resolved_tokenizer_revision,
            architecture=architecture,
            dtype=self.description.dtype,
            quantization=self.quantization,
        )

    def _inspector(self, lens_id: str | None) -> JLensInspector | None:
        if lens_id is None:
            return None
        manifest, payload = self.lens_registry.get(lens_id)
        return JLensInspector(
            manifest,
            payload,
            self._identity(),
            lambda token_id: self._tokenizer.decode([token_id]),
            self._unembed,
        )

    def _unembed(self, residual: object) -> object:
        """Apply the HF model's final norm and output embedding like upstream J-lens."""
        import torch  # noqa: PLC0415

        decoder = self._text_decoder()
        norm = getattr(decoder, "norm", None) or getattr(decoder, "final_layernorm", None)
        output = self._model.get_output_embeddings()
        if norm is None or output is None:
            raise NcsiRuntimeError(
                NcsiErrorCode.MODEL_INCOMPATIBLE,
                "unable to locate final normalization or output embedding",
            )
        target_dtype = output.weight.dtype
        target_device = output.weight.device
        normalized = norm.forward(residual.to(dtype=target_dtype, device=target_device))
        logits = output(normalized)
        text_config = self._model.config.get_text_config()
        softcap = getattr(text_config, "final_logit_softcapping", None)
        return softcap * torch.tanh(logits / softcap) if softcap is not None else logits

    def _text_decoder(self) -> object:
        candidates = (
            getattr(self._model, "model", None),
            getattr(getattr(self._model, "model", None), "language_model", None),
            getattr(self._model, "language_model", None),
            getattr(self._model, "transformer", None),
            getattr(self._model, "gpt_neox", None),
        )
        for candidate in candidates:
            if candidate is not None and (
                hasattr(candidate, "norm") or hasattr(candidate, "final_layernorm")
            ):
                return candidate
        raise NcsiRuntimeError(
            NcsiErrorCode.MODEL_INCOMPATIBLE, "unable to locate the model text decoder"
        )

    async def stream(
        self,
        request: GenerationRequest,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[EngineDelta]:
        await self._ensure_loaded()
        inspector = self._inspector(request.lens_id)
        selected_layers = request.layers or (() if inspector is None else inspector.manifest.layers)
        if inspector is not None and any(layer not in inspector.manifest.layers for layer in selected_layers):
            raise NcsiRuntimeError(
                NcsiErrorCode.LENS_INCOMPATIBLE, "request includes a layer absent from the lens"
            )
        try:
            encoded = self._tokenizer(request.prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self._model.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._model.device)
            prompt_length = int(input_ids.shape[-1])
            past_key_values = None
            current_ids = input_ids
            for index in range(request.max_new_tokens):
                if cancel_event.is_set():
                    raise NcsiRuntimeError(NcsiErrorCode.CANCELLED, "generation was cancelled")
                forward_pass_id = str(uuid.uuid4())
                outputs = await self._forward_cancellable(
                    current_ids, attention_mask, past_key_values, bool(selected_layers)
                )
                if cancel_event.is_set():
                    raise NcsiRuntimeError(NcsiErrorCode.CANCELLED, "generation was cancelled")
                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                token_id = int(next_token.item())
                token_text = self._tokenizer.decode([token_id])
                timestamp = time.time()
                observations = self._observations(
                    request,
                    inspector,
                    selected_layers,
                    outputs.hidden_states,
                    forward_pass_id,
                    prompt_length - 1 + index,
                    prompt_length + index,
                    timestamp,
                )
                yield EngineDelta(token_id, token_text, observations)
                if token_id == self._tokenizer.eos_token_id:
                    break
                past_key_values = outputs.past_key_values
                current_ids = next_token.unsqueeze(0)
                if attention_mask is not None:
                    import torch  # noqa: PLC0415

                    attention_mask = torch.cat(
                        (attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))), dim=-1
                    )
            self._record_peak_memory()
        except NcsiRuntimeError:
            raise
        except Exception as exc:
            self._record_peak_memory()
            if _is_out_of_memory(exc):
                raise NcsiRuntimeError(
                    NcsiErrorCode.ACCELERATOR_OOM, "accelerator ran out of memory during generation"
                ) from exc
            raise NcsiRuntimeError(
                NcsiErrorCode.INFERENCE_FAILED, f"model generation failed: {exc}"
            ) from exc

    async def _forward_cancellable(
        self,
        input_ids: object,
        attention_mask: object,
        past_key_values: object,
        hidden_states: bool,
    ) -> object:
        """Do not release the accelerator slot before a cancelled thread exits."""
        task = asyncio.create_task(
            asyncio.to_thread(
                self._forward_sync,
                input_ids,
                attention_mask,
                past_key_values,
                hidden_states,
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    def _observations(  # noqa: PLR0913
        self,
        request: GenerationRequest,
        inspector: JLensInspector | None,
        selected_layers: tuple[int, ...],
        hidden_states: object,
        forward_pass_id: str,
        position: int,
        predicts_position: int,
        timestamp: float,
    ) -> tuple[NeuralObservation, ...]:
        if inspector is None:
            return ()
        observations = []
        for layer in selected_layers:
            readout = inspector.read(layer, hidden_states[layer + 1][0, -1], request.top_k)
            observations.append(
                NeuralObservation(
                    request_id=request.request_id,
                    forward_pass_id=forward_pass_id,
                    model_id=self.model_id,
                    model_revision=self._resolved_model_revision,
                    tokenizer_revision=self._resolved_tokenizer_revision,
                    lens_id=inspector.manifest.lens_id,
                    lens_revision=inspector.manifest.lens_revision,
                    layer=layer,
                    position=position,
                    concepts=readout.concepts,
                    readout_method=readout.method,
                    parameters={
                        "top-k": request.top_k,
                        "predicts-position": predicts_position,
                    },
                    reconstruction_error=readout.reconstruction_error,
                    timestamp=timestamp,
                )
            )
        return tuple(observations)

    def _forward_sync(
        self, input_ids: object, attention_mask: object, past_key_values: object, hidden_states: bool
    ) -> object:
        import torch  # noqa: PLC0415

        with torch.inference_mode():
            return self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=hidden_states,
            )

    def _record_peak_memory(self) -> None:
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                self.last_peak_accelerator_bytes = int(torch.cuda.max_memory_allocated())
        except ImportError:
            return

    async def unload(self) -> None:
        async with self._load_lock:
            self._model = None
            self._tokenizer = None
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


def _is_out_of_memory(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()
