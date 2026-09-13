"""
Processor supervisor for local inference execution.

Manages model lifecycle, health, concurrency limits, thread offloading,
idle unloading, and task-to-claim conversion with provenance.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import time
from typing import Any, Optional, Union, cast

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken, LifecycleState, LocalProcessor
from rai.kernel.records import (
    ActionFailure,
    Claim,
    ContextManifestItem,
    ContextPackage,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
    ProvenanceReference,
    Task,
)

from .factory import get_available_backends, is_backend_available, load_local_model
from .protocols import (
    AsyncEngineAdapter,
    InferenceEngine,
    InferenceResult,
    LocalTextEngine,
    ProcessorHealth,
    is_async_local_engine,
)
from .tasks import BoundedTaskContract, BoundedTaskKind, get_bounded_task_contract

logger = logging.getLogger(__name__)

SUPERVISOR_PRODUCER = ProducerIdentity(
    producer_id="rai.processor.supervisor",
    kind="processor",
    version="1.0.0",
)


@dataclass(frozen=True)
class _RetainedContext:
    """Privacy-filtered context metadata safe to use for local inference."""

    content: dict[str, Any]
    data_class: DataClass
    source_ids: frozenset[str]


@dataclass(frozen=True)
class _PreparedRequest:
    """Validated request state at the future cache-lookup boundary."""

    contract: BoundedTaskContract | None
    budget: InferenceBudget
    operation_deadline: float
    prompt: str
    retained_context: _RetainedContext


@dataclass(frozen=True)
class _GuardedOperationFailure:
    """Engine failure plus ownership of the reserved capacity lease."""

    failure: ActionFailure
    capacity_release_deferred: bool = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProcessorSupervisor(LocalProcessor):
    """
    Supervisor for local inference engines.
    Ensures safe concurrency, non-blocking execution, idle unloading,
    and schema-constrained Claim generation from ContextPackages.
    """

    def __init__(  # noqa: PLR0913
        self,
        engine: Optional[Union[LocalTextEngine, InferenceEngine]] = None,
        model_name: str = "default",
        backend: str = "ollama",
        max_concurrency: int = 2,
        idle_unload_seconds: float = 300.0,
        name: str = "local-processor",
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.backend = backend.lower()
        self.engine: Optional[LocalTextEngine] = (
            self._coerce_engine(engine) if engine is not None else None
        )
        self.max_concurrency = max_concurrency
        self.idle_unload_seconds = idle_unload_seconds

        self._state = LifecycleState.CREATED
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active_requests = 0
        self._total_requests = 0
        self._error_count = 0
        self._last_active_timestamp: Optional[float] = None
        self._reaper_task: Optional[asyncio.Task[None]] = None
        self._drain_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _coerce_engine(
        engine: Union[LocalTextEngine, InferenceEngine],
    ) -> LocalTextEngine:
        """Normalize synchronous engines without relying on structural isinstance."""
        if is_async_local_engine(engine):
            return engine
        return AsyncEngineAdapter(cast(InferenceEngine, engine))

    @property
    def state(self) -> LifecycleState:
        return self._state

    def _get_engine(self) -> Result[LocalTextEngine, ActionFailure]:
        """Resolve or lazily initialize the underlying local text engine."""
        if self.engine is not None:
            return Success(self.engine)

        if not is_backend_available(self.backend):
            return Failure(
                self._failure(
                    "BACKEND_UNAVAILABLE",
                    f"Inference backend '{self.backend}' is not operational or missing dependencies.",
                )
            )

        load_res = load_local_model(self.model_name, backend=self.backend)
        if isinstance(load_res, Failure):
            return Failure(
                self._failure(
                    "MODEL_INIT_FAILED",
                    f"Failed to initialize engine for '{self.model_name}': {load_res.failure()}",
                )
            )

        raw_engine = load_res.unwrap()
        self.engine = self._coerce_engine(raw_engine)
        return Success(self.engine)

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        """Starts the supervisor and its idle-unloading reaper."""
        if self._state == LifecycleState.RUNNING:
            return Success(self._state)

        self._state = LifecycleState.RUNNING
        if self.idle_unload_seconds > 0:
            self._reaper_task = asyncio.create_task(self._idle_reaper_loop())

        logger.info("ProcessorSupervisor started (backend: %s, idle: %ss)", self.backend, self.idle_unload_seconds)
        return Success(self._state)

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        """Stops the supervisor, cancels the idle reaper, and unloads model memory."""
        if self._state == LifecycleState.STOPPED:
            return Success(self._state)

        self._state = LifecycleState.STOPPING
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None

        if self._drain_tasks:
            await asyncio.gather(*tuple(self._drain_tasks), return_exceptions=True)

        if self.engine is not None and self.engine.is_loaded:
            try:
                await self.engine.unload()
            except Exception as exc:
                logger.warning("Error unloading engine during stop: %s", exc)

        self._state = LifecycleState.STOPPED
        logger.info("ProcessorSupervisor stopped")
        return Success(self._state)

    def health(self) -> ProcessorHealth:
        """Report live health and memory residency status."""
        loaded_model = None
        if self.engine is not None and self.engine.is_loaded:
            loaded_model = self.engine.model_name

        return ProcessorHealth(
            name=self.name,
            state=self._state,
            loaded_model=loaded_model,
            active_requests=self._active_requests,
            total_requests=self._total_requests,
            error_count=self._error_count,
            last_active_timestamp=self._last_active_timestamp,
            available_backends=get_available_backends(),
            required_ram_bytes=(
                getattr(self.engine, "required_ram_bytes", None)
                if self.engine is not None
                else None
            ),
            required_vram_bytes=(
                getattr(self.engine, "required_vram_bytes", None)
                if self.engine is not None
                else None
            ),
        )

    def _release_capacity(self) -> None:
        """Release one processor slot after its engine operation really finishes."""
        self._semaphore.release()
        self._active_requests -= 1
        self._last_active_timestamp = time.monotonic()

    def _defer_capacity_release(self, operation: asyncio.Task[Any]) -> None:
        """Keep capacity reserved while an uninterruptible backend drains."""

        async def drain() -> None:
            try:
                await operation
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("Deferred inference operation failed while draining: %s", exc)
            finally:
                self._release_capacity()

        drain_task = asyncio.create_task(drain())
        self._drain_tasks.add(drain_task)
        drain_task.add_done_callback(self._drain_tasks.discard)

    async def _await_operation(
        self,
        operation: asyncio.Task[Any],
        *,
        cancellation: CancellationToken,
        deadline: float,
        request_id: str,
        operation_name: str,
    ) -> tuple[Any | None, ActionFailure | None, bool]:
        """Await an engine operation while preserving capacity on early return."""
        try:
            while not operation.done():
                if cancellation.cancelled:
                    self._defer_capacity_release(operation)
                    return None, self._cancelled(request_id), True

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._defer_capacity_release(operation)
                    return (
                        None,
                        self._failure(
                            "DEADLINE_EXCEEDED",
                            f"Inference {operation_name} exceeded the budget deadline",
                            request_id=request_id,
                        ),
                        True,
                    )
                await asyncio.sleep(min(0.05, max(0.005, remaining)))

            return await operation, None, False
        except asyncio.CancelledError:
            if not operation.done():
                self._defer_capacity_release(operation)
                return None, self._cancelled(request_id), True
            return None, self._cancelled(request_id), False

    @classmethod
    def _contains_content_key(cls, value: object, target: str) -> bool:
        """Return whether a nested JSON-like value contains a mapping key."""
        if isinstance(value, dict):
            return target in value or any(
                cls._contains_content_key(item, target) for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(cls._contains_content_key(item, target) for item in value)
        return False

    @classmethod
    def _drop_forbidden_content(cls, value: object, forbidden_keys: set[str]) -> object:
        """Recursively remove explicitly forbidden source and field keys."""
        if isinstance(value, dict):
            return {
                key: cls._drop_forbidden_content(item, forbidden_keys)
                for key, item in value.items()
                if key not in forbidden_keys
            }
        if isinstance(value, (list, tuple)):
            return tuple(cls._drop_forbidden_content(item, forbidden_keys) for item in value)
        return value

    async def _idle_reaper_loop(self) -> None:
        """Periodically checks if the model has exceeded idle duration and unloads it."""
        while self._state == LifecycleState.RUNNING:
            try:
                await asyncio.sleep(min(5.0, max(0.05, self.idle_unload_seconds / 4)))
                if (
                    self.engine is not None
                    and self.engine.is_loaded
                    and self._active_requests == 0
                    and self._last_active_timestamp is not None
                ):
                    idle_time = time.monotonic() - self._last_active_timestamp
                    if idle_time >= self.idle_unload_seconds:
                        logger.info(
                            "Idle duration %.1fs reached threshold %.1fs. Unloading model '%s'",
                            idle_time,
                            self.idle_unload_seconds,
                            self.engine.model_name,
                        )
                        await self.engine.unload()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Unexpected error in idle reaper loop: %s", exc)

    def _guard_request_start(
        self,
        task: Task,
        cancellation: CancellationToken,
    ) -> Result[None, ActionFailure]:
        """Reject requests which cannot enter the preparation pipeline."""
        if self._state != LifecycleState.RUNNING:
            return Failure(
                self._failure(
                    "PROCESSOR_NOT_RUNNING",
                    f"Processor is in state {self._state.value}, not RUNNING",
                    request_id=task.record_id,
                )
            )
        if cancellation.cancelled:
            return Failure(self._cancelled(task.record_id))
        return Success(None)

    def _resolve_contract(
        self,
        bounded_task: BoundedTaskKind | str | None,
        budget: InferenceBudget,
        request_id: str,
    ) -> Result[
        tuple[BoundedTaskContract | None, InferenceBudget], ActionFailure
    ]:
        """Resolve one bounded contract and apply its non-expandable limits."""
        if bounded_task is None:
            return Success((None, budget))

        contract_result = get_bounded_task_contract(bounded_task)
        if isinstance(contract_result, Failure):
            contract_failure = contract_result.failure()
            return Failure(
                self._failure(
                    contract_failure.code,
                    contract_failure.message,
                    request_id=request_id,
                    retryable=contract_failure.retryable,
                )
            )

        contract = contract_result.unwrap()
        return Success((contract, contract.limits.constrain(budget)))

    def _operation_deadline(
        self,
        budget: InferenceBudget,
        request_id: str,
    ) -> Result[float, ActionFailure]:
        """Calculate the monotonic deadline shared by capacity, load and generation."""
        deadline_remaining = (
            budget.cancellation_deadline - _utc_now()
        ).total_seconds()
        timeout = min(budget.max_latency_seconds, deadline_remaining)
        if timeout <= 0:
            return Failure(
                self._failure(
                    "DEADLINE_EXCEEDED",
                    "Inference budget deadline exceeded before execution "
                    f"({timeout:.2f}s remaining)",
                    request_id=request_id,
                )
            )
        return Success(time.monotonic() + timeout)

    def _retain_allowed_context(
        self,
        context: ContextPackage,
        request_id: str,
    ) -> Result[_RetainedContext, ActionFailure]:
        """Drop mapped forbidden fields and fail closed on ambiguous content."""
        forbidden_classes = {
            DataClass.SECRET,
            DataClass.BLOCKED,
            "SECRET",
            "BLOCKED",
        }
        manifest_items = context.manifest.items
        forbidden_items = tuple(
            item for item in manifest_items if item.data_class in forbidden_classes
        )
        allowed_items = tuple(
            item for item in manifest_items if item.data_class not in forbidden_classes
        )

        if manifest_items and not allowed_items:
            return Failure(
                self._failure(
                    "DATA_CLASS_REJECTED",
                    "Context package contains only SECRET or BLOCKED data classes "
                    "which cannot be sent to local model",
                    request_id=request_id,
                )
            )

        unmapped_forbidden = tuple(
            item.source_id
            for item in forbidden_items
            if not item.fields
            and not self._contains_content_key(context.content, item.source_id)
        )
        if unmapped_forbidden:
            return Failure(
                self._failure(
                    "DATA_CLASS_REJECTED",
                    "SECRET/BLOCKED manifest sources cannot be mapped safely to "
                    "context content: " + ", ".join(unmapped_forbidden),
                    request_id=request_id,
                )
            )

        forbidden_keys = {item.source_id for item in forbidden_items}
        forbidden_keys.update(
            field.split(".")[-1]
            for item in forbidden_items
            for field in item.fields
        )
        sanitized_content = cast(
            dict[str, Any],
            self._drop_forbidden_content(context.content, forbidden_keys),
        )
        if context.content and not sanitized_content:
            return Failure(
                self._failure(
                    "DATA_CLASS_REJECTED",
                    "All context package content was stripped due to "
                    "SECRET/BLOCKED policy",
                    request_id=request_id,
                )
            )

        return Success(
            _RetainedContext(
                content=sanitized_content,
                data_class=self._retained_data_class(allowed_items),
                source_ids=frozenset(item.source_id for item in allowed_items),
            )
        )

    @staticmethod
    def _retained_data_class(
        manifest_items: tuple[ContextManifestItem, ...],
    ) -> DataClass:
        """Return the strongest retained classification for derived output."""
        if not manifest_items:
            return DataClass.LOCAL

        classes = {item.data_class for item in manifest_items}
        if DataClass.PRIVATE in classes or "PRIVATE" in classes:
            return DataClass.PRIVATE
        if DataClass.LOCAL in classes or "LOCAL" in classes:
            return DataClass.LOCAL
        return DataClass.PUBLIC

    def _prepare_prompt(
        self,
        task: Task,
        retained: _RetainedContext,
        contract: BoundedTaskContract | None,
        budget: InferenceBudget,
    ) -> Result[str, ActionFailure]:
        """Build the deterministic prompt and enforce its input allowance."""
        prompt = (
            contract.build_prompt(
                task.objective,
                retained.content,
                allowed_source_ids=retained.source_ids,
            )
            if contract is not None
            else self._build_prompt(task, retained.content)
        )
        estimated_tokens = max(len(prompt.split()), len(prompt) // 4)
        if budget.max_input_tokens <= 0:
            return Failure(
                self._failure(
                    "BUDGET_EXCEEDED",
                    "max_input_tokens is 0; model input is forbidden by budget",
                    request_id=task.record_id,
                )
            )
        if estimated_tokens > budget.max_input_tokens:
            return Failure(
                self._failure(
                    "BUDGET_EXCEEDED",
                    f"Input token count ({estimated_tokens}) exceeds "
                    f"max_input_tokens ({budget.max_input_tokens})",
                    request_id=task.record_id,
                )
            )
        return Success(prompt)

    def _prepare_request(
        self,
        task: Task,
        context: ContextPackage,
        budget: InferenceBudget,
        bounded_task: BoundedTaskKind | str | None,
    ) -> Result[_PreparedRequest, ActionFailure]:
        """Prepare all policy-visible state before capacity or model access."""
        contract_result = self._resolve_contract(
            bounded_task, budget, task.record_id
        )
        if isinstance(contract_result, Failure):
            return contract_result
        contract, constrained_budget = contract_result.unwrap()

        if constrained_budget.max_output_tokens <= 0:
            return Failure(
                self._failure(
                    "BUDGET_EXCEEDED",
                    "max_output_tokens is 0; generation forbidden by budget",
                    request_id=task.record_id,
                )
            )

        deadline_result = self._operation_deadline(
            constrained_budget, task.record_id
        )
        if isinstance(deadline_result, Failure):
            return deadline_result

        retained_result = self._retain_allowed_context(context, task.record_id)
        if isinstance(retained_result, Failure):
            return retained_result
        retained = retained_result.unwrap()

        prompt_result = self._prepare_prompt(
            task, retained, contract, constrained_budget
        )
        if isinstance(prompt_result, Failure):
            return prompt_result

        return Success(
            _PreparedRequest(
                contract=contract,
                budget=constrained_budget,
                operation_deadline=deadline_result.unwrap(),
                prompt=prompt_result.unwrap(),
                retained_context=retained,
            )
        )

    async def _acquire_capacity(
        self,
        operation_deadline: float,
        request_id: str,
    ) -> Result[None, ActionFailure]:
        """Reserve one capacity lease or return a typed bounded failure."""
        if self._semaphore.locked():
            return Failure(self._capacity_failure(request_id))

        remaining = operation_deadline - time.monotonic()
        if remaining <= 0:
            return Failure(
                self._failure(
                    "DEADLINE_EXCEEDED",
                    "Inference budget deadline exceeded before capacity acquisition",
                    request_id=request_id,
                )
            )
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=min(0.05, remaining)
            )
        except asyncio.TimeoutError:
            if time.monotonic() >= operation_deadline:
                return Failure(
                    self._failure(
                        "DEADLINE_EXCEEDED",
                        "Inference budget deadline exceeded during capacity acquisition",
                        request_id=request_id,
                    )
                )
            return Failure(self._capacity_failure(request_id))

        self._active_requests += 1
        self._total_requests += 1
        return Success(None)

    def _capacity_failure(self, request_id: str) -> ActionFailure:
        """Build the common immediate-capacity failure."""
        return self._failure(
            "CAPACITY_EXCEEDED",
            f"Concurrency limit ({self.max_concurrency}) reached for local processor",
            request_id=request_id,
            retryable=True,
        )

    def _resolve_eligible_engine(
        self,
        prepared: _PreparedRequest,
        request_id: str,
    ) -> Result[LocalTextEngine, ActionFailure]:
        """Resolve an engine and enforce provider and resource ceilings."""
        engine_result = self._get_engine()
        if isinstance(engine_result, Failure):
            self._error_count += 1
            return engine_result
        engine = engine_result.unwrap()

        allowed_providers = prepared.budget.allowed_providers
        if allowed_providers:
            allowed_set = set(allowed_providers)
            if engine.model_name not in allowed_set and self.backend not in allowed_set:
                return Failure(
                    self._failure(
                        "PROVIDER_DISALLOWED",
                        f"Engine '{engine.model_name}' ({self.backend}) is not in "
                        f"allowed_providers: {allowed_providers}",
                        request_id=request_id,
                    )
                )

        resource_result = self._check_engine_resources(
            engine, prepared.budget, request_id
        )
        if isinstance(resource_result, Failure):
            return resource_result
        return Success(engine)

    def _check_engine_resources(
        self,
        engine: LocalTextEngine,
        budget: InferenceBudget,
        request_id: str,
    ) -> Result[None, ActionFailure]:
        """Reject engines with known RAM or VRAM requirements above budget."""
        resources = (
            ("RAM", getattr(engine, "required_ram_bytes", None), budget.max_ram_bytes),
            (
                "VRAM",
                getattr(engine, "required_vram_bytes", None),
                budget.max_vram_bytes,
            ),
        )
        for resource_name, required, limit in resources:
            if required is not None and required > limit:
                return Failure(
                    self._failure(
                        "RESOURCE_CAPACITY_EXCEEDED",
                        f"Engine requires {required} bytes of {resource_name}, "
                        f"budget allows {limit}",
                        request_id=request_id,
                    )
                )
        return Success(None)

    async def _ensure_engine_loaded(
        self,
        engine: LocalTextEngine,
        prepared: _PreparedRequest,
        cancellation: CancellationToken,
        request_id: str,
    ) -> Result[None, _GuardedOperationFailure]:
        """Load an engine within the shared deadline and capacity lease."""
        if engine.is_loaded:
            return Success(None)

        load_task = asyncio.create_task(engine.load())
        load_result, operation_failure, deferred = await self._await_operation(
            load_task,
            cancellation=cancellation,
            deadline=prepared.operation_deadline,
            request_id=request_id,
            operation_name="model loading",
        )
        if operation_failure is not None:
            return Failure(
                _GuardedOperationFailure(operation_failure, deferred)
            )
        if isinstance(load_result, Failure):
            self._error_count += 1
            return Failure(
                _GuardedOperationFailure(
                    self._failure(
                        "MODEL_LOAD_FAILED",
                        f"Failed to load model '{engine.model_name}': "
                        f"{load_result.failure()}",
                        request_id=request_id,
                    )
                )
            )
        return Success(None)

    async def _generate(
        self,
        engine: LocalTextEngine,
        prepared: _PreparedRequest,
        cancellation: CancellationToken,
        request_id: str,
    ) -> Result[InferenceResult, _GuardedOperationFailure]:
        """Generate one bounded result with live cancellation monitoring."""
        generation_task = asyncio.create_task(
            engine.generate(
                prompt=prepared.prompt,
                max_tokens=prepared.budget.max_output_tokens,
                temperature=(
                    prepared.contract.temperature
                    if prepared.contract is not None
                    else 0.7
                ),
            )
        )
        generation_result, operation_failure, deferred = await self._await_operation(
            generation_task,
            cancellation=cancellation,
            deadline=prepared.operation_deadline,
            request_id=request_id,
            operation_name="generation",
        )
        if operation_failure is not None:
            return Failure(
                _GuardedOperationFailure(operation_failure, deferred)
            )
        if isinstance(generation_result, Failure):
            self._error_count += 1
            return Failure(
                _GuardedOperationFailure(
                    self._failure(
                        "INFERENCE_FAILED",
                        f"Inference execution failed: {generation_result.failure()}",
                        request_id=request_id,
                        retryable=True,
                    )
                )
            )
        return Success(generation_result.unwrap())

    def _build_claim(
        self,
        task: Task,
        context: ContextPackage,
        prepared: _PreparedRequest,
        inference_result: InferenceResult,
    ) -> Result[Claim, ActionFailure]:
        """Validate model output before the future cache-store boundary."""
        raw_statement = inference_result.text.strip()
        statement = raw_statement or "No summary or classification generated."
        confidence = 0.9
        epistemic_status = "inferred"
        output_data_class = prepared.retained_context.data_class

        if prepared.contract is not None:
            validation_result = prepared.contract.validate_output(
                raw_statement,
                input_data_class=output_data_class,
                allowed_source_ids=prepared.retained_context.source_ids,
            )
            if isinstance(validation_result, Failure):
                validation_failure = validation_result.failure()
                self._error_count += 1
                return Failure(
                    self._failure(
                        validation_failure.code,
                        validation_failure.message,
                        request_id=task.record_id,
                        retryable=validation_failure.retryable,
                    )
                )
            validated_output = validation_result.unwrap()
            statement = validated_output.claim_statement()
            confidence = validated_output.confidence
            output_data_class = validated_output.result_data_class(
                output_data_class
            )
            epistemic_status = (
                f"inferred:{prepared.contract.kind.value}@"
                f"{prepared.contract.version}"
            )

        source_reference = ProvenanceReference(
            source_id=context.record_id,
            source_type=context.record_type,
            source_version=context.schema_version,
            relation="derived-from",
            producer=context.producer,
        )
        return Success(
            Claim(
                producer=SUPERVISOR_PRODUCER,
                correlation_id=task.correlation_id,
                statement=statement,
                confidence=confidence,
                epistemic_status=epistemic_status,
                data_class=output_data_class,
                provenance=(source_reference,),
            )
        )

    async def _execute_prepared_request(
        self,
        task: Task,
        context: ContextPackage,
        prepared: _PreparedRequest,
        cancellation: CancellationToken,
    ) -> tuple[Result[Claim, ActionFailure], bool]:
        """Execute with capacity reserved; return whether release is deferred."""
        if cancellation.cancelled:
            return Failure(self._cancelled(task.record_id)), False

        engine_result = self._resolve_eligible_engine(prepared, task.record_id)
        if isinstance(engine_result, Failure):
            return engine_result, False
        engine = engine_result.unwrap()

        load_result = await self._ensure_engine_loaded(
            engine, prepared, cancellation, task.record_id
        )
        if isinstance(load_result, Failure):
            guarded_failure = load_result.failure()
            return (
                Failure(guarded_failure.failure),
                guarded_failure.capacity_release_deferred,
            )
        if cancellation.cancelled:
            return Failure(self._cancelled(task.record_id)), False

        generation_result = await self._generate(
            engine, prepared, cancellation, task.record_id
        )
        if isinstance(generation_result, Failure):
            guarded_failure = generation_result.failure()
            return (
                Failure(guarded_failure.failure),
                guarded_failure.capacity_release_deferred,
            )

        return (
            self._build_claim(
                task, context, prepared, generation_result.unwrap()
            ),
            False,
        )

    async def _execute_with_capacity(
        self,
        task: Task,
        context: ContextPackage,
        prepared: _PreparedRequest,
        cancellation: CancellationToken,
    ) -> Result[Claim, ActionFailure]:
        """Release the capacity lease exactly once unless an operation drains."""
        release_capacity = True
        try:
            result, release_deferred = await self._execute_prepared_request(
                task, context, prepared, cancellation
            )
            release_capacity = not release_deferred
            return result
        finally:
            if release_capacity:
                self._release_capacity()

    async def process(
        self,
        task: Task,
        context: ContextPackage,
        budget: InferenceBudget,
        cancellation: CancellationToken,
        *,
        bounded_task: BoundedTaskKind | str | None = None,
    ) -> Result[Claim, ActionFailure]:
        """
        Executes bounded local inference over context packages.
        Outputs a schema-validated Claim with provenance.
        """
        guard_result = self._guard_request_start(task, cancellation)
        if isinstance(guard_result, Failure):
            return guard_result

        prepared_result = self._prepare_request(
            task, context, budget, bounded_task
        )
        if isinstance(prepared_result, Failure):
            return prepared_result
        prepared = prepared_result.unwrap()

        # Future cache lookup boundary: policy, privacy, contract, context and
        # caller budget have been checked, but no capacity or model was touched.
        # A cache implementation must key policy/model versions and retained
        # classification/source identity, then rebind current provenance.
        capacity_result = await self._acquire_capacity(
            prepared.operation_deadline, task.record_id
        )
        if isinstance(capacity_result, Failure):
            return capacity_result

        result = await self._execute_with_capacity(
            task, context, prepared, cancellation
        )
        # Future cache store boundary: only a schema-validated Claim can succeed.
        return result

    async def process_bounded(
        self,
        kind: BoundedTaskKind | str,
        task: Task,
        context: ContextPackage,
        budget: InferenceBudget,
        cancellation: CancellationToken,
    ) -> Result[Claim, ActionFailure]:
        """Run one registered bounded task through the standard supervisor guards."""
        return await self.process(
            task,
            context,
            budget,
            cancellation,
            bounded_task=kind,
        )

    def _build_prompt(self, task: Task, content: dict[str, Any]) -> str:
        """Constructs a deterministic schema-constrained prompt from task and sanitized content."""
        objective = task.objective
        content_items: list[str] = []

        episode = content.get("episode")
        if isinstance(episode, dict):
            apps = episode.get("applications", ())
            resources = episode.get("resources", ())
            projects = episode.get("projects", ())
            activity = episode.get("activity_types", ())
            content_items.append(f"Episode applications: {', '.join(apps) if apps else 'none'}")
            content_items.append(f"Episode projects: {', '.join(projects) if projects else 'none'}")
            content_items.append(f"Episode resources: {', '.join(resources) if resources else 'none'}")
            content_items.append(f"Episode activity: {', '.join(activity) if activity else 'none'}")
        else:
            for k, v in content.items():
                content_items.append(f"{k}: {v}")

        joined_context = "\n".join(content_items) if content_items else "No detailed context."
        return (
            f"Objective: {objective}\n"
            f"Context:\n{joined_context}\n"
            f"Provide a concise, factual summary or classification."
        )

    def _failure(
        self,
        code: str,
        message: str,
        request_id: str = "processor.supervisor",
        retryable: bool = False,
    ) -> ActionFailure:
        return ActionFailure(
            producer=SUPERVISOR_PRODUCER,
            request_id=request_id,
            capability=f"processor.{self.name}",
            code=code,
            message=message,
            retryable=retryable,
        )

    def _cancelled(self, request_id: str) -> ActionFailure:
        return ActionFailure(
            producer=SUPERVISOR_PRODUCER,
            request_id=request_id,
            capability=f"processor.{self.name}",
            code="CANCELLED",
            message="Local processor execution was cancelled.",
            retryable=False,
        )
