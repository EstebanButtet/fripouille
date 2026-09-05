"""FRP-IA-12 : rôles fonctionnels applicatifs, sans permissions nouvelles."""
from dataclasses import dataclass
from collections.abc import Callable
from types import MappingProxyType
import re


@dataclass(frozen=True, slots=True)
class FunctionalRole:
    id: str
    name: str
    objective: str
    required_capabilities: frozenset[str]
    priorities: tuple[str, ...]
    constraints: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.id, str) or not re.fullmatch(r"[a-z][a-z_]{0,31}", self.id):
            raise ValueError("A stable role identifier is required.")
        for text in (self.name, self.objective):
            if not isinstance(text, str) or not text.strip() or len(text) > 200:
                raise ValueError("Role text must contain 1 to 200 characters.")
        if not isinstance(self.required_capabilities, frozenset):
            raise TypeError("Required capabilities must be immutable.")
        if any(not isinstance(x, str) or not re.fullmatch(r"[a-z][a-z_]{0,47}", x)
               for x in self.required_capabilities):
            raise ValueError("Invalid capability identifier.")
        for values in (self.priorities, self.constraints):
            if not isinstance(values, tuple) or len(values) > 4:
                raise ValueError("At most four immutable role directives are allowed.")
            if any(not isinstance(x, str) or not x.strip() or len(x) > 120 for x in values):
                raise ValueError("Role directive is invalid or too long.")


DEFAULT_ROLES = (
    FunctionalRole("guide", "Guide", "Accompagner une explication étape par étape.",
                   frozenset({"conversation"}), ("Clarifier le besoin.", "Expliquer progressivement."),
                   ("Ne pas inventer de connaissance du lieu ou du monde physique.",)),
    FunctionalRole("observateur", "Observateur", "Décrire les perceptions disponibles sans identifier les personnes.",
                   frozenset({"social_vision"}), ("Distinguer observation et hypothèse.",),
                   ("Ne pas inférer d'état mental.", "Aucune surveillance en arrière-plan.")),
    FunctionalRole("cameraman", "Cameraman", "Cadrer une prise de vue à la demande.",
                   frozenset({"record_video", "follow_target"}), ("Vérifier le cadrage et la demande.",),
                   ("Passer uniquement par les capacités ROB autorisées.",)),
)


@dataclass(frozen=True, slots=True)
class RoleDecision:
    accepted: bool
    role_id: str | None
    reason: str
    missing_capabilities: tuple[str, ...] = ()


class RoleService:
    def __init__(self, capabilities: Callable[[], frozenset[str]],
                 roles: tuple[FunctionalRole, ...] = DEFAULT_ROLES):
        if any(not isinstance(role, FunctionalRole) for role in roles):
            raise TypeError("Role catalogue requires validated definitions.")
        if len({role.id for role in roles}) != len(roles):
            raise ValueError("Duplicate role identifier.")
        self.catalogue = MappingProxyType({role.id: role for role in roles})
        self._capabilities, self._active = capabilities, None
        self.last_decision = RoleDecision(True, None, "default")

    @property
    def active(self) -> FunctionalRole | None:
        # Une capacité perdue désactive le rôle au lieu de conserver une promesse.
        if self._active is not None and not self._active.required_capabilities <= self._capabilities():
            self._active = None
            self.last_decision = RoleDecision(False, None, "capability_lost")
        return self._active

    @property
    def active_id(self) -> str | None:
        role = self.active
        return role.id if role else None

    def activate(self, role_id: str) -> RoleDecision:
        if not isinstance(role_id, str):
            raise TypeError("Role identifier must be text.")
        role = self.catalogue.get(role_id)
        if role is None:
            result = RoleDecision(False, self.active_id, "unknown_role")
        else:
            missing = tuple(sorted(role.required_capabilities - self._capabilities()))
            if missing:
                result = RoleDecision(False, self.active_id, "missing_capabilities", missing)
            else:
                self._active = role
                result = RoleDecision(True, role.id, "explicit_activation")
        self.last_decision = result
        return result

    def reset(self) -> RoleDecision:
        self._active = None
        self.last_decision = RoleDecision(True, None, "default")
        return self.last_decision

    def handle_request(self, text: str) -> str | None:
        """Grammaire complète de demande utilisateur ; aucune sortie LLM ici."""
        normalized = text.strip().casefold().rstrip(".")
        if normalized in {"/role off", "passe en mode assistant", "désactive le rôle"}:
            self.reset()
            return "Je reprends mon fonctionnement habituel."
        match = re.fullmatch(r"(?:/role |passe en mode )([a-z_]+)", normalized)
        if match is None:
            return None
        decision = self.activate(match[1])
        if decision.accepted:
            return f"Rôle {self.active.name} activé."
        return "Ce rôle n'est pas disponible avec mes capacités actuelles."
