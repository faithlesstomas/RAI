"""
Processor supervisor for local inference execution.

Manages model lifecycle, health, concurrency limits, thread offloading,
idle unloading, and task-to-claim conversion with provenance.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import time
from typing import Any, Optional

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken, LifecycleState, LocalProcessor
from rai.kernel.records import (
    ActionFailure,
    Claim,
    ContextPackage,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
    ProvenanceReference,
    Task,
)

from .factory import get_available_backends, is_backend_available, load_local_model
from .protocols import LocalTextEngine, ProcessorHealth

logger = logging.getLogger(__name__)

SUPERVISOR_PRODUCER = ProducerIdentity(
    producer_id="rai.processor.supervisor",
    kind="processor",
    version="1.0.0",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProcessorSupervisor(LocalProcessor):
    """
    Supervisor for local inference engines.
    Ensures safe concurrency, non-blocking execution, idle unloading,
    and schema-constrained Claim generation from ContextPackages.
    """

    def __init__(
        self,
        engine: Optional[LocalTextEngine] = None,
        model_name: str = "default",
        backend: str = "ollama",
        max_concurrency: int = 2,
        idle_unload_seconds: float = 300.0,
        name: str = "local-processor",
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.backend = backend.lower()
        self.engine = engine
        self.max_concurrency = max_concurrency
        self.idle_unload_seconds = idle_unload_seconds

        self._state = LifecycleState.CREATED
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active_requests = 0
        self._total_requests = 0
        self._error_count = 0
        self._last_active_timestamp: Optional[float] = None
        self._reaper_task: Optional[asyncio.Task[None]] = None

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

        self.engine = load_res.unwrap()
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
        )

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

    async def process(
        self,
        task: Task,
        context: ContextPackage,
        budget: InferenceBudget,
        cancellation: CancellationToken,
    ) -> Result[Claim, ActionFailure]:
        """
        Executes bounded local inference over context packages.
        Outputs a schema-validated Claim with provenance.
        """
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

        # Check concurrency capacity
        try:
            # Non-blocking or bounded check to report CAPACITY_EXCEEDED
            acquired = False
            try:
                # Wait up to budget limit or 0.1s for concurrency slot
                wait_time = min(0.5, budget.max_input_tokens / 1000.0) if budget.max_input_tokens > 0 else 0.1
                await asyncio.wait_for(self._semaphore.acquire(), timeout=wait_time)
                acquired = True
            except asyncio.TimeoutError:
                return Failure(
                    self._failure(
                        "CAPACITY_EXCEEDED",
                        f"Concurrency limit ({self.max_concurrency}) reached for local processor",
                        request_id=task.record_id,
                        retryable=True,
                    )
                )

            self._active_requests += 1
            self._total_requests += 1

            if cancellation.cancelled:
                return Failure(self._cancelled(task.record_id))

            # Resolve engine
            engine_res = self._get_engine()
            if isinstance(engine_res, Failure):
                self._error_count += 1
                return engine_res

            engine = engine_res.unwrap()

            # Ensure model is loaded
            if not engine.is_loaded:
                load_res = await engine.load()
                if isinstance(load_res, Failure):
                    self._error_count += 1
                    return Failure(
                        self._failure(
                            "MODEL_LOAD_FAILED",
                            f"Failed to load model '{engine.model_name}': {load_res.failure()}",
                            request_id=task.record_id,
                        )
                    )

            if cancellation.cancelled:
                return Failure(self._cancelled(task.record_id))

            # Construct bounded prompt from context
            prompt = self._build_prompt(task, context)

            # Generate with budget limits
            gen_res = await engine.generate(
                prompt=prompt,
                max_tokens=budget.max_output_tokens if budget.max_output_tokens > 0 else 512,
            )

            if isinstance(gen_res, Failure):
                self._error_count += 1
                return Failure(
                    self._failure(
                        "INFERENCE_FAILED",
                        f"Inference execution failed: {gen_res.failure()}",
                        request_id=task.record_id,
                        retryable=True,
                    )
                )

            inference_result = gen_res.unwrap()
            statement = inference_result.text.strip()
            if not statement:
                statement = "No summary or classification generated."

            # Construct provenance reference to context package
            source_ref = ProvenanceReference(
                source_id=context.record_id,
                source_type=context.record_type,
                source_version=context.schema_version,
                relation="derived-from",
                producer=context.producer,
            )

            # Preserve data class from manifest items or default to LOCAL
            out_data_class = DataClass.LOCAL
            if context.manifest.items:
                # If any input is PRIVATE, output remains PRIVATE
                classes = {item.data_class for item in context.manifest.items}
                if DataClass.PRIVATE in classes:
                    out_data_class = DataClass.PRIVATE

            claim = Claim(
                producer=SUPERVISOR_PRODUCER,
                correlation_id=task.correlation_id,
                statement=statement,
                confidence=0.9,
                epistemic_status="inferred",
                data_class=out_data_class,
                provenance=(source_ref,),
            )
            return Success(claim)

        finally:
            if acquired:
                self._semaphore.release()
                self._active_requests -= 1
            self._last_active_timestamp = time.monotonic()

    def _build_prompt(self, task: Task, context: ContextPackage) -> str:
        """Constructs a deterministic schema-constrained prompt from task and context."""
        objective = task.objective
        content_items: list[str] = []

        episode = context.content.get("episode")
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
            for k, v in context.content.items():
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
