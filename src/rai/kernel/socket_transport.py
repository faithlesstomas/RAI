"""Protected newline-delimited JSON transport over a local Unix socket."""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from typing import Any

from returns.result import Failure

from rai.paths import runtime_dir
from rai.tools.security.auth import is_authorized

from .event_service import EventService
from .events import EventCursor
from .ports import EventJournal
from .records import DataClass


class EventSocketServer:
    """Expose the event service on a mode-0600 Unix-domain socket only."""

    def __init__(
        self,
        service: EventService,
        journal: EventJournal,
        path: Path | None = None,
        *,
        max_request_bytes: int = 256 * 1024,
    ) -> None:
        self.service = service
        self.journal = journal
        self.path = path or runtime_dir() / "events-v1.sock"
        self.max_request_bytes = max_request_bytes
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        if self._server is not None:
            return
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            existing_mode = self.path.lstat().st_mode
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISSOCK(existing_mode):
                raise RuntimeError(
                    f"refusing to replace non-socket event path: {self.path}"
                )
            self.path.unlink()
        self._server = await asyncio.start_unix_server(
            self._handle, path=self.path, limit=self.max_request_bytes + 1
        )
        self.path.chmod(0o600)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        try:
            current_mode = self.path.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(current_mode):
            self.path.unlink()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            try:
                request = await reader.readline()
            except ValueError:
                await self._write(writer, {"error": {"code": "OVERSIZED_EVENT", "message": "request exceeds configured size limit"}})
                return
            if len(request) > self.max_request_bytes or not request.endswith(b"\n"):
                await self._write(writer, {"error": {"code": "OVERSIZED_EVENT", "message": "request exceeds configured size limit"}})
                return
            try:
                command: Any = json.loads(request)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                await self._write(writer, {"error": {"code": "INVALID_EVENT", "message": str(exc)}})
                return
            if not isinstance(command, dict) or not is_authorized({"x-rai-token": str(command.get("token", ""))}):
                await self._write(writer, {"error": {"code": "UNAUTHORIZED", "message": "authentication required"}})
                return
            await self._dispatch(command, writer)
        finally:
            writer.close()
            await writer.wait_closed()

    async def _dispatch(self, command: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        operation = command.get("operation")
        if operation == "ingest":
            payload = json.dumps(command.get("event"), separators=(",", ":")).encode()
            supplied_class = command.get("data_class")
            try:
                data_class = DataClass(supplied_class) if supplied_class else None
            except ValueError as exc:
                await self._write(writer, {"error": {"code": "INVALID_EVENT", "message": str(exc)}})
                return
            result = await self.service.ingest(
                payload,
                data_class=data_class,
                clock_source=command.get("clock_source", "system-utc"),
                clock_uncertainty_ms=command.get("clock_uncertainty_ms", 0),
            )
        elif operation == "replay":
            try:
                result = await self.service.replay(command["cursor"], command.get("limit", 50))
            except (KeyError, TypeError) as exc:
                await self._write(writer, {"error": {"code": "INVALID_EVENT", "message": str(exc)}})
                return
        elif operation == "ack":
            try:
                cursor = EventCursor(value=command["cursor"])
                consumer_id = command["consumer_id"]
                if not isinstance(consumer_id, str):
                    raise TypeError("consumer_id must be a string")
                result = await self.journal.acknowledge(consumer_id, cursor)
            except (KeyError, TypeError, ValueError) as exc:
                await self._write(writer, {"error": {"code": "INVALID_EVENT", "message": str(exc)}})
                return
        else:
            await self._write(writer, {"error": {"code": "INVALID_EVENT", "message": "unsupported operation"}})
            return
        if isinstance(result, Failure):
            await self._write(writer, {"error": result.failure().model_dump(mode="json")})
        else:
            await self._write(writer, {"result": result.unwrap().model_dump(mode="json")})

    @staticmethod
    async def _write(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        writer.write(json.dumps(payload, sort_keys=True).encode() + b"\n")
        await writer.drain()
