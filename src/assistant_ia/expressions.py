"""FRP-IA-10 : intentions de présentation, distinctes de toute émotion."""
from dataclasses import dataclass
from enum import Enum
from time import monotonic
from collections.abc import Callable
from typing import Protocol
from assistant_ia.internal_state import InternalStateSnapshot


class Expression(str, Enum):
    NEUTRAL = "neutral"
    FOCUSED = "focused"
    CURIOUS = "curious"
    CONCERNED = "concerned"


@dataclass(frozen=True, slots=True)
class ExpressiveIntent:
    expression: Expression = Expression.NEUTRAL
    reason: str = "default"

    def __post_init__(self):
        if not isinstance(self.expression, Expression):
            raise TypeError("Expression must belong to the application catalogue.")
        if not isinstance(self.reason, str) or not self.reason or len(self.reason) > 80:
            raise ValueError("An expressive reason must contain 1 to 80 characters.")


class ExpressionPresenter(Protocol):
    def present_expression(self, intent: ExpressiveIntent) -> None:
        """Traduire uniquement un symbole validé en présentation."""
        ...


def expression_for_state(state: InternalStateSnapshot) -> ExpressiveIntent:
    if state.guidance in {"blocked", "needs_strategy_change"}:
        return ExpressiveIntent(Expression.CONCERNED, "application_difficulty")
    if state.guidance == "missing_information" or state.activity == "waiting":
        return ExpressiveIntent(Expression.CURIOUS, "awaiting_user")
    if state.activity == "engaged":
        return ExpressiveIntent(Expression.FOCUSED, "processing")
    return ExpressiveIntent()


class ExpressionController:
    """Une intention temporaire expire en cinq secondes ; aucun hardware ici."""
    def __init__(self, *, clock: Callable[[], float] = monotonic):
        self._clock = clock
        self._intent = ExpressiveIntent()
        self._expires = clock()

    @property
    def current(self) -> ExpressiveIntent:
        return self._intent if self._clock() < self._expires else ExpressiveIntent()

    def show(self, intent: ExpressiveIntent) -> None:
        if not isinstance(intent, ExpressiveIntent):
            raise TypeError("A validated expressive intent is required.")
        self._intent, self._expires = intent, self._clock() + 5

    def reset(self) -> None:
        self._intent, self._expires = ExpressiveIntent(), self._clock()
