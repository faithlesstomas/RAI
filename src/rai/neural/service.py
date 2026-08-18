"""NCSI request lifecycle, limits, cancellation, and operational telemetry."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from typing import AsyncIterator

from .contracts import (
    EventType,
    GenerationRequest,
    NcsiErrorCode,
    NcsiEvent,
    NcsiRuntimeError,
)
from .engines.base import GenerationEngine

LOGGER = logging.getLogger(__name__)


class NeuralService:
    """Coordinate one long-lived engine without leaking framework objects."""

    def __init__(self, engine: GenerationEngine, *, max_concurrency: int = 1) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self.engine = engine
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._requests: dict[str, asyncio.Event] = {}
        self._request_lock = asyncio.Lock()
        self._counters: Counter[str] = Counter()
        self._total_latency = 0.0

    async def reserve(self, request_id: str) -> bool:
        """Atomically reserve an ID before an HTTP streaming response starts."""
        async with self._request_lock:
            if request_id in self._requests:
                return False
            self._requests[request_id] = asyncio.Event()
            return True

    async def generate(
        self, request: GenerationRequest, *, reserved: bool = False
    ) -> AsyncIterator[NcsiEvent]:
        """Emit exactly one started-to-terminal lifecycle for a request."""
        started_at = time.monotonic()
        cancel_event = await self._claim_request(request.request_id, reserved=reserved)
        registered = False
        acquired = False
        try:
            registered = True
            yield NcsiEvent(
                EventType.GENERATION_STARTED,
                request.request_id,
                time.time(),
                {"model-id": self.engine.description.model_id},
            )
            if request.model_id not in (None, self.engine.description.model_id):
                raise NcsiRuntimeError(
                    NcsiErrorCode.MODEL_NOT_FOUND, "requested model is not hosted by this sidecar"
                )
            if self._semaphore.locked():
                raise NcsiRuntimeError(
                    NcsiErrorCode.CONCURRENCY_LIMIT, "sidecar concurrency limit is reached"
                )
            await self._semaphore.acquire()
            acquired = True
            async for event in self._stream_active(request, cancel_event):
                yield event
        except NcsiRuntimeError as exc:
            self._counters[f"failed:{exc.code.value}"] += 1
            yield self._failure(request.request_id, exc.code, str(exc))
        except asyncio.CancelledError:
            cancel_event.set()
            self._counters["disconnected"] += 1
            LOGGER.info("NCSI client disconnected", extra={"request_id": request.request_id})
            raise
        except Exception:
            self._counters[f"failed:{NcsiErrorCode.INTERNAL_ERROR.value}"] += 1
            LOGGER.exception("Unhandled neural sidecar error", extra={"request_id": request.request_id})
            yield self._failure(
                request.request_id, NcsiErrorCode.INTERNAL_ERROR, "internal sidecar error"
            )
        finally:
            if acquired:
                self._semaphore.release()
            if registered:
                async with self._request_lock:
                    self._requests.pop(request.request_id, None)
            self._counters["requests"] += 1
            self._total_latency += time.monotonic() - started_at

    async def _claim_request(self, request_id: str, *, reserved: bool) -> asyncio.Event:
        async with self._request_lock:
            existing = self._requests.get(request_id)
            if reserved and existing is not None:
                return existing
            if existing is not None:
                raise NcsiRuntimeError(
                    NcsiErrorCode.INVALID_REQUEST, "request-id is already active"
                )
            cancel_event = asyncio.Event()
            self._requests[request_id] = cancel_event
            return cancel_event

    async def _stream_active(
        self, request: GenerationRequest, cancel_event: asyncio.Event
    ) -> AsyncIterator[NcsiEvent]:
        tokens: list[str] = []
        deadline = time.monotonic() + request.timeout_seconds
        iterator = self.engine.stream(request, cancel_event).__aiter__()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                cancel_event.set()
                raise NcsiRuntimeError(NcsiErrorCode.TIMEOUT, "generation timed out")
            try:
                delta = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as exc:
                cancel_event.set()
                raise NcsiRuntimeError(NcsiErrorCode.TIMEOUT, "generation timed out") from exc
            tokens.append(delta.token_text)
            yield NcsiEvent(
                EventType.TOKEN_DELTA,
                request.request_id,
                time.time(),
                {"token-id": delta.token_id, "token-text": delta.token_text},
            )
            for observation in delta.observations:
                yield NcsiEvent(
                    EventType.NEURAL_STATE_OBSERVED,
                    request.request_id,
                    observation.timestamp,
                    observation.to_wire(),
                )
        self._counters["completed"] += 1
        yield NcsiEvent(
            EventType.GENERATION_COMPLETED,
            request.request_id,
            time.time(),
            {"final-text": "".join(tokens), "token-count": len(tokens)},
        )

    async def cancel(self, request_id: str) -> bool:
        """Cooperatively cancel one active request."""
        async with self._request_lock:
            cancel_event = self._requests.get(request_id)
            if cancel_event is None:
                return False
            cancel_event.set()
            return True

    async def unload(self) -> bool:
        """Unload only while idle so an active request retains model ownership."""
        async with self._request_lock:
            if self._requests:
                return False
            await self.engine.unload()
            return True

    def telemetry(self) -> dict[str, object]:
        requests = self._counters["requests"]
        return {
            "active-requests": len(self._requests),
            "max-concurrency": self.max_concurrency,
            "request-count": requests,
            "completed-count": self._counters["completed"],
            "failure-counts": {
                key.removeprefix("failed:"): value
                for key, value in self._counters.items()
                if key.startswith("failed:")
            },
            "mean-latency-seconds": self._total_latency / requests if requests else 0.0,
            "model-load-seconds": getattr(self.engine, "load_seconds", None),
            "peak-accelerator-bytes": getattr(
                self.engine, "last_peak_accelerator_bytes", None
            ),
        }

    @staticmethod
    def _failure(request_id: str, code: NcsiErrorCode, message: str) -> NcsiEvent:
        return NcsiEvent(
            EventType.GENERATION_FAILED,
            request_id,
            time.time(),
            {"error-code": code.value, "error-message": message},
        )
