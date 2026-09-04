"""Small durable stores used by the Stage 1 conformance kernel."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from returns.result import Failure, Result, Success

from rai.paths import data_dir

from .records import (
    ActionFailure,
    Episode,
    KernelRecord,
    Observation,
    ProducerIdentity,
)

RecordT = TypeVar("RecordT", Observation, Episode)
STORE_PRODUCER = ProducerIdentity(
    producer_id="rai.record-store", kind="storage", version="1.0.0"
)


class _JsonRecordStore:
    """One-record-per-file store with atomic replacement and explicit failures."""

    record_model: type[KernelRecord]
    capability: str

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = asyncio.Lock()

    async def _append(self, record: RecordT) -> Result[RecordT, ActionFailure]:
        try:
            async with self._lock:
                self._write_atomic(record)
            return Success(record)
        except OSError as exc:
            return Failure(self._failure(record.record_id, "STORE_WRITE_FAILED", str(exc)))

    async def _get(self, record_id: str) -> Result[RecordT, ActionFailure]:
        try:
            data = json.loads(
                (self.root / f"{record_id}.json").read_text(encoding="utf-8")
            )
            return Success(self.record_model.model_validate(data))  # type: ignore[arg-type,return-value]
        except FileNotFoundError:
            return Failure(self._failure(record_id, "NOT_FOUND", "record was not found"))
        except (OSError, ValueError) as exc:
            return Failure(self._failure(record_id, "STORE_READ_FAILED", str(exc)))

    def _write_atomic(self, record: RecordT) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=self.root, prefix=".record-", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(record.model_dump(mode="json"), stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / f"{record.record_id}.json")
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def _failure(self, record_id: str, code: str, message: str) -> ActionFailure:
        return ActionFailure(
            producer=STORE_PRODUCER,
            correlation_id=record_id,
            request_id=record_id,
            capability=self.capability,
            code=code,
            message=message,
        )


class JsonObservationStore(_JsonRecordStore):
    record_model = Observation
    capability = "observation.store"

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(root or data_dir() / "kernel" / "observations")

    async def append(
        self, observation: Observation
    ) -> Result[Observation, ActionFailure]:
        return await self._append(observation)

    async def get(self, record_id: str) -> Result[Observation, ActionFailure]:
        return await self._get(record_id)


class JsonEpisodeStore(_JsonRecordStore):
    record_model = Episode
    capability = "episode.store"

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(root or data_dir() / "kernel" / "episodes")

    async def append(self, episode: Episode) -> Result[Episode, ActionFailure]:
        return await self._append(episode)

    async def get(self, record_id: str) -> Result[Episode, ActionFailure]:
        return await self._get(record_id)
