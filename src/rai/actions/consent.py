"""Process-local, request-bound consent; never accepted from serialized arguments."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
from typing import Iterator

from rai.kernel.records import CapabilityRequest

_browser_consent: ContextVar[str | None] = ContextVar("browser_consent", default=None)
BROWSER_ACTIONS = frozenset({"browser.search", "browser.open_result", "browser.read_page"})


def fingerprint(request: CapabilityRequest) -> str:
    return hashlib.sha256(request.model_dump_json().encode()).hexdigest()


def has_browser_consent(request: CapabilityRequest) -> bool:
    return _browser_consent.get() == fingerprint(request)


@contextmanager
def browser_consent(request: CapabilityRequest) -> Iterator[None]:
    token = _browser_consent.set(fingerprint(request))
    try:
        yield
    finally:
        _browser_consent.reset(token)


def review_text(request: CapabilityRequest) -> str:
    """Escape terminal control characters in untrusted query/target text."""
    import json  # noqa: PLC0415

    payload = request.arguments.get("query") if request.capability == "browser.search" else request.target_resource
    return (f"Action: {request.capability}\nClass: {request.data_class}\n"
            f"Outbound query/target: {json.dumps(payload, ensure_ascii=True)}\n"
            "Only this query/target is authorized; conversation and history are not sent.")
