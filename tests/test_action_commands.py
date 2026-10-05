"""Streaming output is complete and bounded even across partial pipe reads."""
import asyncio
import sys

import pytest
from returns.result import Failure, Success

from rai.actions.commands import MAX_COMMAND_BYTES, read_bounded_output, run_command
from rai.kernel.ports import CancellationToken



@pytest.mark.asyncio
async def test_partial_reads_are_drained_until_eof():
    reader = asyncio.StreamReader()
    reader.feed_data(b'first ')
    pending = asyncio.create_task(read_bounded_output(reader))
    await asyncio.sleep(0)
    assert not pending.done()
    reader.feed_data(b'second')
    reader.feed_eof()
    assert await pending == b'first second'


@pytest.mark.asyncio
async def test_output_limit_does_not_wait_for_eof():
    reader = asyncio.StreamReader()
    reader.feed_data(b'x' * (MAX_COMMAND_BYTES + 1))
    with pytest.raises(ValueError, match='OUTPUT_LIMIT'):
        await read_bounded_output(reader)


@pytest.mark.asyncio
async def test_exact_output_limit_is_accepted():
    reader = asyncio.StreamReader()
    reader.feed_data(b'x' * MAX_COMMAND_BYTES)
    reader.feed_eof()
    assert len(await read_bounded_output(reader)) == MAX_COMMAND_BYTES


async def run(*code: str, token: CancellationToken | None = None):
    return await run_command(sys.executable, ("-c", *code), token or CancellationToken())


@pytest.mark.asyncio
async def test_run_command_returns_decoded_output_and_filters_environment(monkeypatch):
    monkeypatch.setenv('RAI_SECRET_TOKEN', 'must-not-leak')
    script = 'import os; print(os.environ.get("RAI_SECRET_TOKEN", "absent"), end="")'
    assert await run(script) == Success('absent')


@pytest.mark.asyncio
async def test_run_command_without_program_or_after_cancel_does_not_spawn():
    assert await run_command('definitely-not-installed-program', (), CancellationToken()) == Failure('BACKEND_UNAVAILABLE')
    token = CancellationToken()
    token.cancel()
    assert await run('print(1)', token=token) == Failure('CANCELLED')


@pytest.mark.asyncio
async def test_run_command_maps_nonzero_exit_invalid_text_and_oversized_output():
    assert await run('raise SystemExit(3)') == Failure('COMMAND_FAILED')
    assert await run('import sys; sys.stdout.buffer.write(b"\\xff\\xfe")') == Failure('COMMAND_FAILED')
    oversized = f'import sys; sys.stdout.write("x" * {MAX_COMMAND_BYTES + 1})'
    assert await run(oversized) == Failure('OUTPUT_LIMIT')


@pytest.mark.asyncio
async def test_cancelled_command_is_killed_and_reaped():
    token = CancellationToken()
    pending = asyncio.create_task(run('import time; time.sleep(30)', token=token))
    await asyncio.sleep(0.2)
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 5)

