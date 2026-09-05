"""FRP-IA-08: état fonctionnel éphémère, sans émotion ni autorité."""

from dataclasses import dataclass
from time import monotonic
from collections.abc import Callable
from enum import Enum


class StateEvent(str, Enum):
    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"
    ACTION_SUCCEEDED = "action_succeeded"
    ACTION_FAILED = "action_failed"
    CANCELLED = "cancelled"
    INFORMATION_REQUIRED = "information_required"
    CONFIRMATION_REQUIRED = "confirmation_required"
    CORRECTION = "correction"
    GOAL_COMPLETED = "goal_completed"
    PERSON_CHANGED = "person_changed"
    ERROR = "error"
    IDLE = "idle"
    RESET = "reset"


@dataclass(frozen=True, slots=True)
class InternalStateSnapshot:
    activity: str = "available"
    guidance: str | None = None
    reason: str = "reset"
    consecutive_failures: int = 0
    failed_action: str | None = None


class InternalStateService:
    """Transitions applicatives ; expiration paresseuse après 5 min au repos.

    Un début de tour conserve le besoin de changer de stratégie jusqu'au
    résultat suivant. Aucun texte du modèle n'est accepté comme événement.
    Les compteurs sont diagnostiques, jamais des permissions.
    """

    def __init__(self, *, clock: Callable[[], float] = monotonic) -> None:
        self._clock = clock
        self._updated_at = clock()
        self._snapshot = InternalStateSnapshot()

    @property
    def snapshot(self) -> InternalStateSnapshot:
        if (self._snapshot.activity != "engaged"
                and self._clock() - self._updated_at >= 300):
            self.transition(StateEvent.IDLE)
        return self._snapshot

    def transition(self, event: StateEvent, *, action: str | None = None) -> InternalStateSnapshot:
        if not isinstance(event, StateEvent):
            raise TypeError("An application StateEvent is required.")
        old = self.snapshot if event != StateEvent.IDLE else self._snapshot
        activity, guidance, count, failed = "available", None, 0, None
        if event == StateEvent.TURN_STARTED:
            activity, guidance = "engaged", old.guidance
            count, failed = old.consecutive_failures, old.failed_action
        elif event == StateEvent.ACTION_FAILED:
            if not isinstance(action, str) or not action.strip():
                raise ValueError("A failed action requires its application name.")
            count = min(2, old.consecutive_failures + 1) if old.failed_action == action else 1
            failed = action
            guidance = "needs_strategy_change" if count >= 2 else "blocked"
        elif event == StateEvent.ERROR:
            guidance = "blocked"
        elif event == StateEvent.CORRECTION:
            guidance = "needs_strategy_change"
        elif event == StateEvent.INFORMATION_REQUIRED:
            guidance = "missing_information"
        elif event == StateEvent.CONFIRMATION_REQUIRED:
            activity = "waiting"
        self._snapshot = InternalStateSnapshot(activity, guidance, event.value, count, failed)
        self._updated_at = self._clock()
        return self._snapshot
