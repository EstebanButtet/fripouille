"""Business repository error definitions."""

from __future__ import annotations


class RepositoryError(RuntimeError):
    """Raised when persisted business data cannot be handled safely."""


class TaskNotFoundError(RepositoryError):
    """Raised when a requested task does not exist."""


class TaskAlreadyCompletedError(RepositoryError):
    """Raised when a completed task is completed again."""


class MemoryNotFoundError(RepositoryError):
    """Raised when a requested memory does not exist."""
