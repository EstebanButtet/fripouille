"""Tests for the user-facing and diagnostic response boundary."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

from assistant_ia.actions.action import Action
from assistant_ia.actions.registry import ActionRegistry
from assistant_ia.core.assistant import AssistantCore
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.interfaces.presentation import build_user_facing_response
from assistant_ia.memory.models import Memory, MemoryCandidate
from assistant_ia.memory.promotion import MemoryPromotionProposal
from assistant_ia.runtime import AssistantRuntime, TurnDiagnostics


class FakeModelClient:
    def generate_response(self, messages: tuple[object, ...]) -> ModelResponse:
        return ModelResponse(
            content="Résultat modèle ignoré.",
            model="fake-model",
            intent=Intent(
                name="save_memory",
                parameters={"content": "Mon animal préféré est le renard."},
            ),
        )


class RecordingDiagnosticReporter:
    def __init__(self) -> None:
        self.turns: list[TurnDiagnostics] = []

    def report(self, diagnostics: TurnDiagnostics) -> None:
        self.turns.append(diagnostics)


class ResponsePresentationTests(unittest.TestCase):
    def _candidate(self, content: str) -> MemoryCandidate:
        return MemoryCandidate(
            content=content,
            source_text=content,
            confidence=0.9,
        )

    def _memory(self, content: str) -> Memory:
        timestamp = datetime(2026, 8, 28, tzinfo=timezone.utc)
        return Memory(
            id=7,
            content=content,
            source="conversation_analysis",
            source_text=content,
            confidence=0.9,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def test_runtime_hides_created_memory_id_and_keeps_diagnostic(
        self,
    ) -> None:
        reporter = RecordingDiagnosticReporter()
        assistant = AssistantCore(
            model_client=FakeModelClient(),
            action_registry=ActionRegistry(
                actions=(
                    Action(
                        name="save_memory",
                        handler=lambda parameters: (
                            "Souvenir enregistré : [#7] "
                            f"{parameters['content']}"
                        ),
                    ),
                )
            ),
        )
        runtime = AssistantRuntime(
            assistant,
            diagnostic_reporter=reporter,
        )

        response = runtime.process_message(
            "Souviens-toi que mon animal préféré est le renard."
        )

        self.assertEqual(response, "D’accord, je garde ça en tête.")
        self.assertNotIn("[#7]", response)
        self.assertEqual(
            reporter.turns[0].raw_response,
            "Souvenir enregistré : [#7] "
            "Mon animal préféré est le renard.",
        )

    def test_creation_proposal_is_natural(self) -> None:
        proposal = MemoryPromotionProposal(
            operation="create",
            candidate=self._candidate("Mon animal préféré est le renard."),
        )

        response = build_user_facing_response(
            "Réponse naturelle.\n\n"
            "Ça ressemble à une information utile à retenir : "
            "« Mon animal préféré est le renard. » Réponds oui ou non.",
            intent=Intent(name="conversation"),
            memory_proposal=proposal,
            awaiting_memory_confirmation=True,
        )

        self.assertIn("Je le garde en tête ?", response)
        self.assertNotIn("Réponse naturelle.", response)
        self.assertNotIn("Memory", response)
        self.assertNotIn("[#", response)

    def test_correction_proposal_hides_related_memory_id(self) -> None:
        proposal = MemoryPromotionProposal(
            operation="update",
            candidate=self._candidate(
                "Je préfère maintenant les pieuvres."
            ),
            related_memory=self._memory(
                "Mon animal préféré est le corbeau."
            ),
        )

        response = build_user_facing_response(
            "Réponse.\n\nCorrection du souvenir [#7].",
            intent=Intent(name="conversation"),
            memory_proposal=proposal,
            awaiting_memory_confirmation=True,
        )

        self.assertIn("les pieuvres", response)
        self.assertNotIn("[#7]", response)
        self.assertNotIn("correction du souvenir", response.casefold())

    def test_duplicate_hides_internal_structure(self) -> None:
        proposal = MemoryPromotionProposal(
            operation="possible_duplicate",
            candidate=self._candidate("Je préfère les renards."),
            related_memory=self._memory("Mon animal préféré est le renard."),
        )

        response = build_user_facing_response(
            "Réponse.\n\npossible_duplicate [#7] MemoryCandidate",
            intent=Intent(name="conversation"),
            memory_proposal=proposal,
            awaiting_memory_confirmation=True,
        )

        self.assertIn("quelque chose que je connais déjà", response)
        self.assertNotIn("possible_duplicate", response)
        self.assertNotIn("MemoryCandidate", response)
        self.assertNotIn("[#7]", response)

    def test_confirmation_and_refusal_remain_natural(self) -> None:
        proposal = MemoryPromotionProposal(
            operation="create",
            candidate=self._candidate("Je préfère les renards."),
        )

        accepted = build_user_facing_response(
            "Souvenir enregistré : [#7] Je préfère les renards.",
            intent=None,
            memory_proposal=proposal,
            awaiting_memory_confirmation=False,
        )
        refused = build_user_facing_response(
            "D'accord, je ne conserverai pas cette information.",
            intent=None,
            memory_proposal=proposal,
            awaiting_memory_confirmation=False,
        )

        self.assertEqual(accepted, "D’accord.")
        self.assertIn("ne conserverai pas", refused)
        self.assertNotIn("[#", accepted + refused)

    def test_memory_search_removes_only_persistence_ids(self) -> None:
        response = build_user_facing_response(
            "Souvenirs trouvés :\n• [#7] Le renard.",
            intent=Intent(
                name="find_memory",
                parameters={"query": "renard"},
            ),
            memory_proposal=None,
            awaiting_memory_confirmation=False,
        )

        self.assertEqual(response, "Souvenirs trouvés :\n• Le renard.")

    def test_failed_memory_action_is_never_presented_as_success(self) -> None:
        error_response = "La mémoire n'a pas pu être modifiée."

        response = build_user_facing_response(
            error_response,
            intent=Intent(
                name="save_memory",
                parameters={"content": "Information."},
            ),
            memory_proposal=None,
            awaiting_memory_confirmation=False,
        )

        self.assertEqual(response, error_response)


if __name__ == "__main__":
    unittest.main()
