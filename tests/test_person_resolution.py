"""Tests for deterministic active-person resolution in FRP-IA-04B."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant_ia.application import build_default_assistant
from assistant_ia.core.assistant import AssistantCore
from assistant_ia.core.context import ConversationMessage
from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.prompt import build_system_prompt
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.defaults import (
    DEFAULT_PERSON_ID,
    DEFAULT_PERSON_NAME,
)
from assistant_ia.people.models import PersonProfile
from assistant_ia.people.person_repository import PersonRepository
from assistant_ia.people.resolution import PersonResolutionService
from assistant_ia.security.confirmation import (
    ConfirmationHandler,
    ConfirmationRequest,
)


class ObservingModelClient:
    """Record the active identity visible when generation starts."""

    def __init__(self) -> None:
        self.person_context: ActivePersonContext | None = None
        self.observed_people: list[tuple[int | None, str]] = []

    def generate_response(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> ModelResponse:
        if self.person_context is None:
            raise AssertionError("Person context was not connected.")

        self.observed_people.append(
            (
                self.person_context.active_person_id,
                self.person_context.active_person.name,
            )
        )

        return ModelResponse(
            content="Bonjour.",
            model="test-model",
            intent=Intent(name="conversation", parameters={}),
        )


class PersonResolutionServiceTests(unittest.TestCase):
    """Validate repository decisions independently from conversation."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name) / "assistant.db"
        )
        self.database.initialize()
        self.repository = PersonRepository(self.database)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_resolves_one_exact_normalized_existing_person(self) -> None:
        existing = self.repository.create_person("Élodie")

        resolution = PersonResolutionService(
            self.repository,
            confirmation_handler=lambda request: self.fail(
                "Existing people must not require confirmation."
            ),
        ).resolve_presentation(
            PersonProfile(name="  e\u0301LODIE  ")
        )

        self.assertEqual(resolution.status, "existing")
        self.assertEqual(resolution.person, existing)

    def test_unknown_name_is_not_created_without_confirmation(self) -> None:
        resolution = PersonResolutionService(
            self.repository
        ).resolve_presentation(PersonProfile(name="Alice"))

        self.assertEqual(resolution.status, "creation_refused")
        self.assertIsNone(resolution.person)
        self.assertEqual(
            tuple(
                person.display_name
                for person in self.repository.list_persons()
            ),
            (DEFAULT_PERSON_NAME,),
        )

    def test_confirmation_is_precise_and_creates_person(self) -> None:
        requests: list[ConfirmationRequest] = []

        def confirm(request: ConfirmationRequest) -> bool:
            requests.append(request)
            return True

        resolution = PersonResolutionService(
            self.repository,
            confirmation_handler=confirm,
        ).resolve_presentation(PersonProfile(name="Alice"))

        self.assertEqual(resolution.status, "created")
        self.assertIsNotNone(resolution.person)
        self.assertEqual(resolution.person.display_name, "Alice")
        self.assertEqual(
            requests,
            [
                ConfirmationRequest(
                    action_name="create_person",
                    description=(
                        "créer la personne « Alice » et la définir "
                        "comme interlocuteur actif"
                    ),
                )
            ],
        )

    def test_repeated_presentation_does_not_duplicate_person(self) -> None:
        confirmation_count = 0

        def confirm(request: ConfirmationRequest) -> bool:
            nonlocal confirmation_count
            confirmation_count += 1
            return True

        service = PersonResolutionService(
            self.repository,
            confirmation_handler=confirm,
        )

        first = service.resolve_presentation(PersonProfile(name="Alice"))
        second = service.resolve_presentation(PersonProfile(name="alice"))

        self.assertEqual(first.status, "created")
        self.assertEqual(second.status, "existing")
        self.assertEqual(second.person, first.person)
        self.assertEqual(confirmation_count, 1)
        self.assertEqual(len(self.repository.list_persons()), 2)

    def test_homonyms_are_reported_without_selection(self) -> None:
        first = self.repository.create_person("Alex")
        second = self.repository.create_person("alex")

        resolution = PersonResolutionService(
            self.repository,
            confirmation_handler=lambda request: self.fail(
                "Homonyms must not request creation confirmation."
            ),
        ).resolve_presentation(PersonProfile(name="ALEX"))

        self.assertEqual(resolution.status, "ambiguous")
        self.assertIsNone(resolution.person)
        self.assertEqual(
            resolution.matching_person_ids,
            (first.id, second.id),
        )

    def test_stale_confirmation_cannot_create_or_select_person(self) -> None:
        def confirm_after_registry_change(
            request: ConfirmationRequest,
        ) -> bool:
            self.repository.create_person("Alice")
            return True

        resolution = PersonResolutionService(
            self.repository,
            confirmation_handler=confirm_after_registry_change,
        ).resolve_presentation(PersonProfile(name="Alice"))

        self.assertEqual(resolution.status, "stale")
        self.assertIsNone(resolution.person)
        self.assertEqual(
            len(self.repository.find_persons_by_display_name("Alice")),
            1,
        )


class PersistentActivePersonPipelineTests(unittest.TestCase):
    """Validate the application-to-core active-person pipeline."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name) / "assistant.db"
        )
        self.database.initialize()
        self.repository = PersonRepository(self.database)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _build_assistant(
        self,
        *,
        confirmation_handler: ConfirmationHandler | None = None,
    ) -> tuple[AssistantCore, ObservingModelClient]:
        model_client = ObservingModelClient()
        assistant = build_default_assistant(
            database=self.database,
            model_client=model_client,
            confirmation_handler=confirmation_handler,
        )
        model_client.person_context = assistant.person_context
        return assistant, model_client

    def test_starts_with_persistent_default_person(self) -> None:
        assistant, model_client = self._build_assistant()

        self.assertEqual(
            assistant.person_context.active_person_id,
            DEFAULT_PERSON_ID,
        )
        self.assertEqual(
            assistant.person_context.active_person.name,
            DEFAULT_PERSON_NAME,
        )
        self.assertEqual(model_client.observed_people, [])

    def test_existing_person_is_active_before_model_call(self) -> None:
        alice = self.repository.create_person("Alice")
        assistant, model_client = self._build_assistant()

        result = assistant.process_message("Je m'appelle alice.")

        self.assertEqual(result, "Bonjour.")
        self.assertEqual(
            model_client.observed_people,
            [(alice.id, "Alice")],
        )
        self.assertEqual(
            assistant.last_person_resolution.status,
            "existing",
        )

    def test_new_name_without_confirmation_skips_model_and_creation(
        self,
    ) -> None:
        assistant, model_client = self._build_assistant()

        result = assistant.process_message("Je m'appelle Alice.")

        self.assertIn("je ne crée pas", result)
        self.assertEqual(model_client.observed_people, [])
        self.assertEqual(
            self.repository.find_persons_by_display_name("Alice"),
            (),
        )
        self.assertEqual(
            assistant.person_context.active_person_id,
            DEFAULT_PERSON_ID,
        )

    def test_confirmed_person_is_created_then_activated(self) -> None:
        requests: list[ConfirmationRequest] = []

        def confirm(request: ConfirmationRequest) -> bool:
            requests.append(request)
            return True

        assistant, model_client = self._build_assistant(
            confirmation_handler=confirm
        )

        assistant.process_message("Je m'appelle Alice.")

        matches = self.repository.find_persons_by_display_name("Alice")
        self.assertEqual(len(matches), 1)
        self.assertEqual(
            model_client.observed_people,
            [(matches[0].id, "Alice")],
        )
        self.assertEqual(len(requests), 1)

    def test_reset_returns_to_persistent_este(self) -> None:
        alice = self.repository.create_person("Alice")
        assistant, model_client = self._build_assistant()
        assistant.process_message("Je m'appelle Alice.")

        assistant.reset_conversation()

        self.assertEqual(
            assistant.person_context.active_person_id,
            DEFAULT_PERSON_ID,
        )
        self.assertEqual(
            assistant.person_context.active_person.name,
            DEFAULT_PERSON_NAME,
        )
        self.assertNotEqual(alice.id, DEFAULT_PERSON_ID)
        self.assertEqual(model_client.observed_people, [(alice.id, "Alice")])

    def test_new_application_session_returns_to_este(self) -> None:
        first_assistant, first_model_client = self._build_assistant(
            confirmation_handler=lambda request: True
        )
        first_assistant.process_message("Je m'appelle Alice.")

        second_assistant, second_model_client = self._build_assistant()

        self.assertNotEqual(
            first_assistant.person_context.active_person_id,
            DEFAULT_PERSON_ID,
        )
        self.assertEqual(
            second_assistant.person_context.active_person_id,
            DEFAULT_PERSON_ID,
        )
        self.assertEqual(
            second_assistant.person_context.active_person.name,
            DEFAULT_PERSON_NAME,
        )
        self.assertEqual(len(self.repository.list_persons()), 2)
        self.assertEqual(len(first_model_client.observed_people), 1)
        self.assertEqual(second_model_client.observed_people, [])

    def test_homonyms_do_not_call_model_or_change_active_person(self) -> None:
        alice = self.repository.create_person("Alice")
        self.repository.create_person("Alex")
        self.repository.create_person("Alex")
        assistant, model_client = self._build_assistant(
            confirmation_handler=lambda request: self.fail(
                "Ambiguity must not request confirmation."
            )
        )
        assistant.process_message("Je m'appelle Alice.")

        result = assistant.process_message("Je m'appelle Alex.")

        self.assertIn("Plusieurs personnes", result)
        self.assertEqual(
            model_client.observed_people,
            [(alice.id, "Alice")],
        )
        self.assertEqual(
            assistant.person_context.active_person_id,
            alice.id,
        )

    def test_third_party_mention_does_not_change_person(self) -> None:
        self.repository.create_person("Alice")
        assistant, model_client = self._build_assistant()

        assistant.process_message("J'ai parlé à Alice.")

        self.assertEqual(
            model_client.observed_people,
            [(DEFAULT_PERSON_ID, DEFAULT_PERSON_NAME)],
        )
        self.assertIsNone(assistant.last_person_resolution)

    def test_prompt_keeps_resolved_canonical_display_name(self) -> None:
        alice = self.repository.create_person("Alice")
        assistant, model_client = self._build_assistant()
        assistant.process_message("  Je m'appelle ALICE.  ")

        prompt = build_system_prompt(
            build_default_identity(),
            assistant.person_context,
        )

        self.assertIn("Current user: Alice", prompt)
        self.assertIn(
            "The current user's name is exactly: Alice.",
            prompt,
        )
        self.assertEqual(model_client.observed_people, [(alice.id, "Alice")])


if __name__ == "__main__":
    unittest.main()
