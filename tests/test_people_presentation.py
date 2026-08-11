from __future__ import annotations

import unittest

from assistant_ia.core.assistant import AssistantCore
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.models import PersonProfile
from assistant_ia.people.presentation import detect_presented_person


class RecordingModelClient:
    def __init__(
        self,
        person_context: ActivePersonContext,
    ) -> None:
        self._person_context = person_context
        self.observed_person_names: list[str] = []

    def generate_response(
        self,
        messages,
    ) -> ModelResponse:
        self.observed_person_names.append(
            self._person_context.active_person.name
        )

        return ModelResponse(
            content="Bonjour.",
            model="test-model",
            intent=Intent(
                name="conversation",
                parameters={},
            ),
        )


class PersonPresentationTests(unittest.TestCase):
    def test_detects_moi_cest(self) -> None:
        person = detect_presented_person(
            "Salut, moi c'est Lucas."
        )

        self.assertIsNotNone(person)
        self.assertEqual(
            person.name,
            "Lucas",
        )

    def test_detects_je_mappelle(self) -> None:
        person = detect_presented_person(
            "Bonjour, je m'appelle Sarah."
        )

        self.assertIsNotNone(person)
        self.assertEqual(
            person.name,
            "Sarah",
        )

    def test_detects_mon_prenom_est(self) -> None:
        person = detect_presented_person(
            "Mon prénom est Alex."
        )

        self.assertIsNotNone(person)
        self.assertEqual(
            person.name,
            "Alex",
        )

    def test_ignores_person_mention(self) -> None:
        self.assertIsNone(
            detect_presented_person(
                "Mon ami Lucas vient demain."
            )
        )

    def test_ignores_opinion_about_person(self) -> None:
        self.assertIsNone(
            detect_presented_person(
                "Je pense que Lucas a raison."
            )
        )

    def test_core_switches_before_model_call(self) -> None:
        person_context = ActivePersonContext(
            assistant_name="Fripouille",
            default_person=PersonProfile(
                name="Este",
            ),
        )
        model_client = RecordingModelClient(
            person_context
        )
        assistant = AssistantCore(
            model_client=model_client,
            person_context=person_context,
        )

        assistant.process_message(
            "Salut, moi c'est Lucas."
        )

        self.assertEqual(
            person_context.active_person.name,
            "Lucas",
        )
        self.assertEqual(
            model_client.observed_person_names,
            ["Lucas"],
        )

    def test_name_question_is_not_a_presentation(
        self,
    ) -> None:
        person = detect_presented_person(
            "Tu sais comment je m'appelle maintenant ?"
        )

        self.assertIsNone(person)

    def test_name_question_does_not_replace_active_person(
        self,
    ) -> None:
        person_context = ActivePersonContext(
            assistant_name="Fripouille",
            default_person=PersonProfile(
                name="Este",
            ),
        )
        model_client = RecordingModelClient(
            person_context
        )
        assistant = AssistantCore(
            model_client=model_client,
            person_context=person_context,
        )

        assistant.process_message(
            "Salut, moi c'est Lucas."
        )
        assistant.process_message(
            "Tu sais comment je m'appelle maintenant ?"
        )

        self.assertEqual(
            person_context.active_person.name,
            "Lucas",
        )
        self.assertEqual(
            model_client.observed_person_names,
            ["Lucas", "Lucas"],
        )

    def test_reserved_assistant_name_does_not_switch_user(
        self,
    ) -> None:
        person_context = ActivePersonContext(
            assistant_name="Fripouille",
            default_person=PersonProfile(
                name="Este",
            ),
        )
        model_client = RecordingModelClient(
            person_context
        )
        assistant = AssistantCore(
            model_client=model_client,
            person_context=person_context,
        )

        assistant.process_message(
            "Salut, moi c'est Fripouille."
        )

        self.assertEqual(
            person_context.active_person.name,
            "Este",
        )
        self.assertEqual(
            model_client.observed_person_names,
            ["Este"],
        )

    def test_ordinary_message_does_not_switch_user(
        self,
    ) -> None:
        person_context = ActivePersonContext(
            assistant_name="Fripouille",
            default_person=PersonProfile(
                name="Este",
            ),
        )
        model_client = RecordingModelClient(
            person_context
        )
        assistant = AssistantCore(
            model_client=model_client,
            person_context=person_context,
        )

        assistant.process_message(
            "Mon ami Lucas vient demain."
        )

        self.assertEqual(
            person_context.active_person.name,
            "Este",
        )
        self.assertEqual(
            model_client.observed_person_names,
            ["Este"],
        )


if __name__ == "__main__":
    unittest.main()
