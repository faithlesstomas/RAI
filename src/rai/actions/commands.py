"""Bounded fixed-program invocation, without a shell or model-owned executable."""
from __future__ import annotations

import asyncio
import os
import shutil

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken
from .documents import DESKTOP_ENVIRONMENT
from .service import bounded

MAX_COMMAND_BYTES = 256 * 1024


async def read_bounded_output(reader: asyncio.StreamReader) -> bytes:
    """Drain chunks to EOF while retaining at most the configured byte budget."""
    chunks = bytearray()
    while True:
        chunk = await reader.read(min(65536, MAX_COMMAND_BYTES + 1 - len(chunks)))
        if not chunk:
            return bytes(chunks)
        chunks.extend(chunk)
        if len(chunks) > MAX_COMMAND_BYTES:
            raise ValueError("OUTPUT_LIMIT")


async def run_command(program: str, arguments: tuple[str, ...], token: CancellationToken) -> Result[str, str]:  # noqa: PLR0911
    executable = shutil.which(program)
    if executable is None:
        return Failure("BACKEND_UNAVAILABLE")
    if token.cancelled:
        return Failure("CANCELLED")
    process = await asyncio.create_subprocess_exec(executable, *arguments, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.DEVNULL,
                                                 env={key: value for key, value in os.environ.items() if key in DESKTOP_ENVIRONMENT})
    try:
        if process.stdout is None:
            return Failure("COMMAND_FAILED")
        raw = await bounded(read_bounded_output(process.stdout), token, 5.0)
        await bounded(process.wait(), token, 5.0)
        if process.returncode != 0:
            return Failure("COMMAND_FAILED")
        return Success(raw.decode("utf-8"))
    except (UnicodeError, OSError):
        return Failure("COMMAND_FAILED")
    except ValueError:
        return Failure("OUTPUT_LIMIT")
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()
