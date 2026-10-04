"""Allowed-root file discovery and scoped, revalidated document activation."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Protocol

from returns.result import Failure, Result, Success

from rai.kernel.capabilities import CapabilityDescriptor, CapabilityRegistry, RegisteredCapability
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, ActionResult, CapabilityRequest, RiskClass, _utc_now

from .capabilities import HANDLE_TTL, MAX_RESULTS, PRODUCER, failure, result
from .handles import SQLiteHandleStore
from .records import ResourceHandle

MAX_VISITED = 10000
MAX_QUERY = 256
SAFE_DOCUMENT_SUFFIXES = frozenset({".txt", ".md", ".pdf", ".odt", ".ods", ".odp", ".docx", ".xlsx", ".pptx", ".csv", ".rtf"})


@dataclass(frozen=True)
class Document:
    path: Path
    fingerprint: str


def inspect_document(path: Path, roots: tuple[Path, ...]) -> Document | None:
    """Reject symlinks, executable files and targets outside approved roots."""
    try:
        absolute = path.absolute()
        if not any(absolute.is_relative_to(root) for root in roots):
            return None
        if any(part.is_symlink() for part in (absolute, *absolute.parents)):
            return None
        value = absolute.stat()
        if not stat.S_ISREG(value.st_mode) or value.st_mode & 0o111:
            return None
        if absolute.suffix.lower() not in SAFE_DOCUMENT_SUFFIXES:
            return None
        identity = f"{value.st_dev}:{value.st_ino}:{value.st_size}:{value.st_mtime_ns}"
        return Document(absolute, hashlib.sha256(identity.encode()).hexdigest())
    except OSError:
        return None


def search_documents(roots: tuple[Path, ...], query: str) -> tuple[Document, ...]:
    """Bound work by visited directory entries; never read document contents."""
    found: list[Document] = []
    pending = list(roots)
    visited = 0
    while pending and visited < MAX_VISITED and len(found) < MAX_RESULTS:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    visited += 1
                    if visited > MAX_VISITED:
                        break
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif query.casefold() in entry.name.casefold():
                        document = inspect_document(Path(entry.path), roots)
                        if document:
                            found.append(document)
                            if len(found) >= MAX_RESULTS:
                                break
        except OSError:
            continue
    return tuple(sorted(found, key=lambda item: str(item.path)))


class DocumentBackend(Protocol):
    async def open(self, document: Document, cancellation: CancellationToken) -> Result[dict, str]: ...


class FileSearch:
    name = "file.search"

    def __init__(self, roots: tuple[Path, ...], handles: SQLiteHandleStore) -> None:
        self.roots = roots
        self.handles = handles

    async def invoke(self, request: CapabilityRequest, cancellation: CancellationToken) -> Result[ActionResult, ActionFailure]:
        query = request.arguments["query"]
        task_id = request.arguments["task_id"]
        if not self.roots:
            return Failure(failure(request, "FILE_ROOTS_NOT_CONFIGURED"))
        if not query.strip() or len(query) > MAX_QUERY or not task_id:
            return Failure(failure(request, "INVALID_ARGUMENT"))
        documents = await asyncio.to_thread(search_documents, self.roots, query)
        entries = []
        for document in documents:
            if cancellation.cancelled:
                return Failure(failure(request, "CANCELLED"))
            now = _utc_now()
            handle = ResourceHandle(
                producer=PRODUCER, timestamp=now, actor_id=request.actor.producer_id,
                task_id=task_id, kind="file", target=str(document.path), fingerprint=document.fingerprint,
                operations=("document.open",), expires_at=now + HANDLE_TTL, data_class=request.data_class,
            )
            issued = self.handles.issue(handle)
            if isinstance(issued, Failure):
                return Failure(failure(request, issued.failure()))
            entries.append({"name": document.path.name, "path": str(document.path), "handle": issued.unwrap(),
                            "expires_at": handle.expires_at.isoformat()})
        return Success(result(request, {"documents": entries, "limit": MAX_RESULTS}, {"allowed_roots_checked": True}))


class DocumentOpen:
    name = "document.open"

    def __init__(self, roots: tuple[Path, ...], handles: SQLiteHandleStore, backend: DocumentBackend) -> None:
        self.roots, self.handles, self.backend = roots, handles, backend

    async def invoke(self, request: CapabilityRequest, cancellation: CancellationToken) -> Result[ActionResult, ActionFailure]:
        resolved = self.handles.resolve(request.arguments["handle"], actor_id=request.actor.producer_id,
                                        task_id=request.arguments["task_id"], operation=self.name)
        if isinstance(resolved, Failure):
            return Failure(failure(request, resolved.failure()))
        handle = resolved.unwrap()
        document = inspect_document(Path(handle.target), self.roots)
        if handle.kind != "file" or document is None or document.fingerprint != handle.fingerprint:
            return Failure(failure(request, "STALE_RESOURCE"))
        if handle.data_class != request.data_class:
            return Failure(failure(request, "RESOURCE_CLASS_MISMATCH"))
        opened = await self.backend.open(document, cancellation)
        if isinstance(opened, Failure):
            return Failure(failure(request, opened.failure()))
        return Success(result(request, {"status": "SUCCEEDED", "handle": handle.record_id}, opened.unwrap()))


def register_file_capabilities(registry: CapabilityRegistry, roots: tuple[Path, ...], handles: SQLiteHandleStore,
                               backend: DocumentBackend) -> None:
    for implementation, fields, risk, effects, checks in (
        (FileSearch(roots, handles), ("query", "task_id"), RiskClass.LOW, (), ("allowed-root-metadata",)),
        (DocumentOpen(roots, handles, backend), ("handle", "task_id"), RiskClass.MODERATE,
         ("document-open",), ("document-application-evidence",)),
    ):
        registry.register(RegisteredCapability(CapabilityDescriptor(
            name=implementation.name, description=f"Bounded {implementation.name} (v1).",
            input_schema={"type": "object", "properties": {name: {"type": "string"} for name in fields},
                          "required": list(fields), "additionalProperties": False},
            risk_class=risk, side_effects=effects, isolation="host-api", verification_plan=checks,
        ), implementation=implementation))
