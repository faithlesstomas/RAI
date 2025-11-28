"""
Utility functions for RAI WebUI.
"""
import logging
from typing import Any, Coroutine, TypeVar
from returns.result import Result, Success, Failure
from nicegui import ui

T = TypeVar("T")

logger = logging.getLogger(__name__)

async def safe_api_call(
    coro: Coroutine[Any, Any, T],
    error_msg: str = "Operation failed"
) -> Result[T, Exception]:
    """
    Executes an async API call safely, catching exceptions and returning a Result.
    Logs errors and shows a UI notification on failure.
    """
    try:
        result = await coro
        return Success(result)
    except Exception as e: # pylint: disable=broad-exception-caught
        logger.error("API Call failed: %s", e, exc_info=True)
        ui.notify(f"{error_msg}: {str(e)}", type='negative')
        return Failure(e)
