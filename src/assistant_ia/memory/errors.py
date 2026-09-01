"""Exceptions métier communes aux repositories persistants.

Elles permettent aux couches supérieures de traiter un échec connu sans
dépendre des exceptions techniques propres à SQLite.
"""

from __future__ import annotations


class RepositoryError(RuntimeError):
    """Signaler qu'une donnée persistée ne peut pas être manipulée sûrement."""


class TaskNotFoundError(RepositoryError):
    """Signaler que la tâche demandée n'existe pas."""


class TaskAlreadyCompletedError(RepositoryError):
    """Signaler une nouvelle complétion d'une tâche déjà terminée."""


class MemoryNotFoundError(RepositoryError):
    """Signaler que le souvenir demandé n'existe pas."""
