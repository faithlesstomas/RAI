"""Incremental history agrees with full replay without rewriting old sources."""
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import threading
from unittest.mock import patch
from types import SimpleNamespace

from returns.result import Success
from rai.container import ApplicationContainer

import pytest

from test_rich_history import service
from rai.history.models import SourceEvent
from rai.history.fusion import DeterministicEpisodeBuilder, DeterministicFusion
from rai.history.sidecars.filesystem import watch
from rai.history.storage import EncryptedHistoryStore, StaticKeyProvider
from rai.kernel.records import DataClass, Episode, Observation, ProducerIdentity

PRODUCER = ProducerIdentity(producer_id="history-perf", kind="test", version="1.0.0")
START = datetime(2026, 9, 1, tzinfo=timezone.utc)


def observation(index: int, seconds: int) -> Observation:
    return Observation(
        record_id=f"source:{index}", timestamp=START + timedelta(seconds=seconds),
        producer=PRODUCER, kind="idle" if index % 7 == 0 else "focus",
        payload={"application_id": f"app-{index % 3}", "source": "gnome", "kind": "idle" if index % 7 == 0 else "focus"},
        data_class=DataClass.PRIVATE if index % 2 else DataClass.LOCAL,
    )


def store_at(path: Path) -> EncryptedHistoryStore:
    return EncryptedHistoryStore(path, key_provider=StaticKeyProvider(b"x" * 32))


def replay(observations: tuple[Observation, ...]) -> tuple[Episode, ...]:
    return DeterministicEpisodeBuilder().build(DeterministicFusion().fuse(observations))


def comparable(episodes: tuple[Episode, ...]) -> list[dict[str, object]]:
    return sorted((ep.model_dump(exclude={"timestamp"}) for ep in episodes), key=lambda ep: ep["record_id"])


@pytest.mark.parametrize("shuffled", [False, True])
def test_incremental_matches_replay_after_every_event(tmp_path: Path, shuffled: bool) -> None:
    store = store_at(tmp_path / "history.sqlite3")
    values = [observation(i, (i // 15) * 600 + (i % 15) // 2) for i in range(75)]
    if shuffled:
        random.Random(42).shuffle(values)
    accumulated = []
    for value in values:
        accumulated.append(value)
        store.append_observation(value, replay, timedelta(seconds=2))
        assert comparable(store.query_episodes()) == comparable(replay(tuple(accumulated)))
    # Restart/retry is idempotent and doesn't change stored provenance.
    store = store_at(tmp_path / "history.sqlite3")
    store.append_observation(values[-1], replay, timedelta(seconds=2))
    assert len(store.observations()) == len(values)


def test_new_event_does_not_read_or_encrypt_closed_history(tmp_path: Path) -> None:
    store = store_at(tmp_path / "history.sqlite3")
    values = tuple(observation(i, i * 600) for i in range(100))
    store.replace(values, replay(values))
    with patch.object(store, "_decrypt", wraps=store._decrypt) as decrypt, patch.object(
        store, "_encrypt", wraps=store._encrypt,
    ) as encrypt:
        store.append_observation(observation(100, 60000), replay, timedelta(seconds=2))
    assert decrypt.call_count == 1
    assert encrypt.call_count <= 3  # noqa: PLR2004 - one source plus two episodes


@pytest.mark.asyncio
async def test_filesystem_startup_is_a_baseline_not_created_events(tmp_path: Path) -> None:
    (tmp_path / "already-present").write_text("fixture")
    with patch("rai.history.sidecars.filesystem._emit") as emit, patch(
        "rai.history.sidecars.filesystem.asyncio.sleep", side_effect=asyncio.CancelledError,
    ), pytest.raises(asyncio.CancelledError):
        await watch((tmp_path,), 1)
    emit.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_write_finishes_before_deletion(tmp_path: Path) -> None:

    history = service(tmp_path)
    entered, release = threading.Event(), threading.Event()
    append = history.store.append_observation

    def delayed_append(*args: object) -> None:
        entered.set()
        assert release.wait(5)
        append(*args)

    with patch.object(history.store, "append_observation", side_effect=delayed_append):
        pending = asyncio.create_task(history.ingest(SourceEvent(
            source="gnome", kind="focus", application_id="synthetic", timestamp=START,
        )))
        try:
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(0.005)
            assert entered.is_set()  # the event loop remained responsive
            pending.cancel()
            deletion = asyncio.create_task(history.delete_range(START, START))
            await asyncio.sleep(0.02)
            assert not deletion.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        result = await deletion
    assert result["verified"] is True
    assert history.store.observations() == ()
    await history.close()


@pytest.mark.asyncio
async def test_dispatch_polling_backs_off_when_idle_and_resets_on_work() -> None:

    idle_polls = 6

    class Dispatcher:
        last_batch_empty = True
        calls = 0

        async def process_next(self) -> Success[None]:
            self.calls += 1
            self.last_batch_empty = self.calls <= idle_polls
            return Success(None)

    delays = []

    async def sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == idle_polls + 1:
            raise asyncio.CancelledError

    with patch("rai.container.asyncio.sleep", sleep), pytest.raises(asyncio.CancelledError):
        await ApplicationContainer._dispatch_events(SimpleNamespace(event_dispatcher=Dispatcher()))
    assert delays == [0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 0.05]
