"""
Custom exceptions for the RAI application.
"""


class RAIError(Exception):
    """Base exception for all RAI application errors."""


class AdapterNotFoundError(RAIError):
    """Raised when a requested AI framework adapter cannot be found."""


class ChainExecutionError(RAIError):
    """Raised when an error occurs during the execution of an agent chain."""


class AgentConfigError(RAIError):
    """Raised when there is an error with the agent configuration."""
