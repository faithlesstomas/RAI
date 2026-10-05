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
from .applications import ApplicationBackend

MAX_VISITED = 10000
MAX_QUERY = 256
SAFE_DOCUMENT_SUFFIXES = frozenset({".txt", ".md", ".pdf", ".odt", ".ods", ".odp", ".docx", ".xlsx", ".pptx", ".csv", ".rtf"})


@dataclass(frozen=True)
class Document:
    path: Path
    fingerprint: str


def inspect_document(path: Path, roots: tuple[Path, ...], *, allow_directory: bool = False) -> Document | None:  # noqa: PLR0911
    """Reject symlinks, executable files and targets outside approved roots."""
    try:
        absolute = Path(os.path.abspath(path))
        if not any(absolute.is_relative_to(root) for root in roots):
            return None
        if any(part.is_symlink() for part in (absolute, *absolute.parents)):
            return None
        value = absolute.stat()
        directory = stat.S_ISDIR(value.st_mode)
        if directory and not allow_directory:
            return None
        if not directory and (not stat.S_ISREG(value.st_mode) or value.st_mode & 0o111):
            return None
        if not directory and absolute.suffix.lower() not in SAFE_DOCUMENT_SUFFIXES:
            return None
        identity = f"{value.st_dev}:{value.st_ino}:{value.st_size}:{value.st_mtime_ns}"
        return Document(absolute, hashlib.sha256(identity.encode()).hexdigest())
    except OSError:
        return None


def search_documents(roots: tuple[Path, ...], query: str) -> tuple[Document, ...]:
    """Bound work by visited directory entries; never read document contents."""
    if Path(query).is_absolute():
        document = inspect_document(Path(query), roots, allow_directory=True)
        return (document,) if document else ()
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


class FileAccess:
    """Describe the configured document boundary, not ambient OS permissions."""

    name = "file.access"

    def __init__(self, roots: tuple[Path, ...]) -> None:
        self.roots = roots

    async def invoke(self, request: CapabilityRequest, cancellation: CancellationToken) -> Result[ActionResult, ActionFailure]:
        if cancellation.cancelled:
            return Failure(failure(request, "CANCELLED"))
        return Success(result(request, {"allowed_roots": [str(root) for root in self.roots],
                                        "editing_supported": False, "scope": "document-actions"},
                              {"source": "runtime-configuration"}))


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


class FileList:
    """List one allowed directory without reading contents or following links."""

    name = "file.list"

    def __init__(self, roots: tuple[Path, ...]) -> None:
        self.roots = roots

    async def invoke(self, request: CapabilityRequest, cancellation: CancellationToken) -> Result[ActionResult, ActionFailure]:
        if cancellation.cancelled:
            return Failure(failure(request, "CANCELLED"))
        path = Path(request.arguments["query"])
        if not self.roots:
            return Failure(failure(request, "FILE_ROOTS_NOT_CONFIGURED"))
        if not path.is_absolute() or inspect_document(path, self.roots, allow_directory=True) is None:
            return Failure(failure(request, "PATH_NOT_ALLOWED"))

        def listing() -> dict:
            # Traverse via descriptors so swapping a parent for a symlink cannot
            # redirect scandir outside the configured boundary after validation.
            descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
            try:
                for component in path.parts[1:]:
                    child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                    os.close(descriptor)
                    descriptor = child
                entries = []
                truncated = False
                with os.scandir(descriptor) as children:
                    for index, entry in enumerate(children):
                        if index >= MAX_RESULTS:
                            truncated = True
                            break
                        kind = "symlink" if entry.is_symlink() else "directory" if entry.is_dir(follow_symlinks=False) else "file"
                        entries.append({"name": entry.name, "path": str(path / entry.name), "kind": kind})
                return {"entries": sorted(entries, key=lambda item: item["name"]), "truncated": truncated}
            finally:
                os.close(descriptor)

        try:
            output = await asyncio.to_thread(listing)
        except OSError:
            return Failure(failure(request, "DIRECTORY_UNAVAILABLE"))
        return Success(result(request, output, {"allowed_roots_checked": True}))


class DocumentOpen:
    name = "document.open"

    def __init__(self, roots: tuple[Path, ...], handles: SQLiteHandleStore, backend: DocumentBackend,
                 applications: ApplicationBackend | None = None) -> None:
        self.roots, self.handles, self.backend = roots, handles, backend
        self.applications = applications

    async def invoke(self, request: CapabilityRequest, cancellation: CancellationToken) -> Result[ActionResult, ActionFailure]:  # noqa: PLR0911
        resolved = self.handles.resolve(request.arguments["handle"], actor_id=request.actor.producer_id,
                                        task_id=request.arguments["task_id"], operation=self.name)
        if isinstance(resolved, Failure):
            return Failure(failure(request, resolved.failure()))
        handle = resolved.unwrap()
        document = inspect_document(Path(handle.target), self.roots, allow_directory=True)
        if handle.kind != "file" or document is None or document.fingerprint != handle.fingerprint:
            return Failure(failure(request, "STALE_RESOURCE"))
        if handle.data_class != request.data_class:
            return Failure(failure(request, "RESOURCE_CLASS_MISMATCH"))
        if "application_handle" in request.arguments:
            app_handle = self.handles.resolve(request.arguments["application_handle"],
                actor_id=request.actor.producer_id, task_id=request.arguments["task_id"], operation=self.name)
            if isinstance(app_handle, Failure):
                return Failure(failure(request, app_handle.failure()))
            authority = app_handle.unwrap()
            if authority.kind != "application" or authority.data_class != request.data_class:
                return Failure(failure(request, "RESOURCE_CLASS_MISMATCH"))
            if self.applications is None or not hasattr(self.backend, "open_with_application"):
                return Failure(failure(request, "APPLICATION_OPEN_UNAVAILABLE"))
            discovered = await self.applications.discover()
            if isinstance(discovered, Failure):
                return Failure(failure(request, discovered.failure()))
            matches = [app for app in discovered.unwrap()
                       if app.desktop_id == authority.target and app.fingerprint == authority.fingerprint]
            if len(matches) != 1:
                return Failure(failure(request, "STALE_RESOURCE"))
            opened = await self.backend.open_with_application(document, matches[0], cancellation)
        else:
            opened = await self.backend.open(document, cancellation)
        if isinstance(opened, Failure):
            return Failure(failure(request, opened.failure()))
        return Success(result(request, {"status": "SUCCEEDED", "handle": handle.record_id}, opened.unwrap()))


def register_file_capabilities(registry: CapabilityRegistry, roots: tuple[Path, ...], handles: SQLiteHandleStore,
                               backend: DocumentBackend, applications: ApplicationBackend | None = None) -> None:
    for implementation, fields, risk, effects, checks in (
        (FileAccess(roots), ("task_id",), RiskClass.LOW, (), ("runtime-configuration",)),
        (FileList(roots), ("query", "task_id"), RiskClass.LOW, (), ("allowed-root-metadata",)),
        (FileSearch(roots, handles), ("query", "task_id"), RiskClass.LOW, (), ("allowed-root-metadata",)),
        (DocumentOpen(roots, handles, backend, applications), ("handle", "task_id"), RiskClass.MODERATE,
         ("document-open",), ("document-application-evidence",)),
    ):
        properties = {name: {"type": "string"} for name in fields}
        if implementation.name == "document.open":
            properties["application_handle"] = {"type": "string"}
        registry.register(RegisteredCapability(CapabilityDescriptor(
            name=implementation.name, description=f"Bounded {implementation.name} (v1).",
            input_schema={"type": "object", "properties": properties,
                          "required": list(fields), "additionalProperties": False},
            risk_class=risk, side_effects=effects, isolation="host-api", verification_plan=checks,
        ), implementation=implementation))
