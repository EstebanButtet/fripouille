from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request
from unittest.mock import patch

from assistant_ia.application import build_default_assistant
from assistant_ia.core.context import ConversationMessage
from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.intelligence.model_client import OllamaModelClient
from assistant_ia.intelligence.prompt import build_system_prompt
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.models import PersonProfile


class FakeHTTPResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def build_model_response() -> bytes:
    structured_content = {
        "content": "Réponse.",
        "intent": {
            "name": "conversation",
            "parameters": {},
        },
    }

    return json.dumps(
        {
            "model": "qwen3.5:4b",
            "message": {
                "content": json.dumps(
                    structured_content,
                    ensure_ascii=False,
                ),
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")


class PeoplePromptPipelineTests(unittest.TestCase):
    def test_prompt_identifies_both_participants(self) -> None:
        identity = build_default_identity()
        person_context = ActivePersonContext(
            assistant_name=identity.name,
            default_person=PersonProfile(
                name="Este",
            ),
        )

        prompt = build_system_prompt(
            identity,
            person_context,
        )

        self.assertIn(
            "Current conversation participants:",
            prompt,
        )
        self.assertIn(
            "Assistant: Fripouille",
            prompt,
        )
        self.assertIn(
            "Current user: Este",
            prompt,
        )
        self.assertIn(
            "The name Fripouille belongs exclusively to the assistant.",
            prompt,
        )
        self.assertIn(
            "It never identifies the current user or any other person.",
            prompt,
        )
        self.assertLess(
            prompt.index(
                "Current conversation participants:"
            ),
            prompt.index(
                "Conversational response quality rules:"
            ),
        )

    def test_prompt_requires_exact_user_name_spelling(
        self,
    ) -> None:
        identity = build_default_identity()
        person_context = ActivePersonContext(
            assistant_name=identity.name,
            default_person=PersonProfile(
                name="Este",
            ),
        )

        prompt = build_system_prompt(
            identity,
            person_context,
        )

        self.assertIn(
            "The current user's name is exactly: Este.",
            prompt,
        )
        self.assertIn(
            "If you use the current user's name, preserve its exact spelling.",
            prompt,
        )

    def test_prompt_forbids_invented_user_nicknames(
        self,
    ) -> None:
        identity = build_default_identity()
        person_context = ActivePersonContext(
            assistant_name=identity.name,
            default_person=PersonProfile(
                name="Este",
            ),
        )

        prompt = build_system_prompt(
            identity,
            person_context,
        )

        self.assertIn(
            "Never alter, shorten or invent a nickname for the current user "
            "unless the user explicitly introduced it.",
            prompt,
        )
        self.assertIn(
            "Do not use the user's name merely as emotional emphasis.",
            prompt,
        )

    def test_current_user_context_is_authoritative(
        self,
    ) -> None:
        identity = build_default_identity()
        person_context = ActivePersonContext(
            assistant_name=identity.name,
            default_person=PersonProfile(
                name="Lucas",
            ),
        )

        prompt = build_system_prompt(
            identity,
            person_context,
        )

        self.assertIn(
            "The Current user field is authoritative for who is speaking.",
            prompt,
        )
        self.assertIn(
            "Do not infer a different speaker merely because another "
            "person's name appears in the message.",
            prompt,
        )

    def test_third_party_names_do_not_replace_current_user(
        self,
    ) -> None:
        identity = build_default_identity()
        person_context = ActivePersonContext(
            assistant_name=identity.name,
            default_person=PersonProfile(
                name="Lucas",
            ),
        )

        prompt = build_system_prompt(
            identity,
            person_context,
        )

        self.assertIn(
            "Names mentioned by the current user normally refer to "
            "other people.",
            prompt,
        )
        self.assertIn(
            "Only an application-confirmed explicit self-presentation "
            "changes the Current user.",
            prompt,
        )

    def test_model_client_uses_current_active_person(
        self,
    ) -> None:
        """Conversation generation should read active person dynamically."""
        identity = build_default_identity()
        person_context = ActivePersonContext(
            assistant_name=identity.name,
            default_person=PersonProfile(
                name="Este",
            ),
        )
        client = OllamaModelClient(
            identity=identity,
            person_context=person_context,
        )
        messages = (
            ConversationMessage(
                role="user",
                content="Bonjour.",
            ),
        )

        conversation_prompts: list[str] = []

        def fake_urlopen(
            request: Request,
            timeout: float,
        ) -> FakeHTTPResponse:
            payload = json.loads(
                request.data.decode("utf-8")
            )

            if "format" in payload:
                response_content = json.dumps(
                    {
                        "name": "conversation",
                        "parameters": {},
                        "conversation": {
                            "mode": "standard",
                            "target_text": None,
                        },
                    }
                )
            else:
                conversation_prompts.append(
                    payload["messages"][0]["content"]
                )
                response_content = (
                    "Reponse conversationnelle."
                )

            response_data = {
                "model": "qwen3.5:4b",
                "message": {
                    "content": response_content,
                },
            }

            return FakeHTTPResponse(
                json.dumps(
                    response_data
                ).encode("utf-8")
            )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            side_effect=fake_urlopen,
        ):
            client.generate_response(
                messages
            )

            person_context.set_active_person(
                PersonProfile(
                    name="Lucas",
                )
            )

            client.generate_response(
                messages
            )

        self.assertEqual(
            len(conversation_prompts),
            2,
        )

        self.assertIn(
            "Current user: Este",
            conversation_prompts[0],
        )
        self.assertIn(
            "The current user's name is exactly: Este.",
            conversation_prompts[0],
        )

        self.assertIn(
            "Current user: Lucas",
            conversation_prompts[1],
        )
        self.assertIn(
            "The current user's name is exactly: Lucas.",
            conversation_prompts[1],
        )

    def test_default_application_uses_este(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            database = SQLiteDatabase(
                Path(directory) / "assistant.db"
            )

            assistant = build_default_assistant(
                database=database,
            )

            self.assertEqual(
                assistant.person_context.assistant_name,
                "Fripouille",
            )
            self.assertEqual(
                assistant.person_context.active_person.name,
                "Este",
            )

    def test_reset_restores_este(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            database = SQLiteDatabase(
                Path(directory) / "assistant.db"
            )

            assistant = build_default_assistant(
                database=database,
            )

            assistant.person_context.set_active_person(
                PersonProfile(
                    name="Lucas",
                )
            )

            self.assertEqual(
                assistant.person_context.active_person.name,
                "Lucas",
            )

            assistant.reset_conversation()

            self.assertEqual(
                assistant.person_context.active_person.name,
                "Este",
            )


if __name__ == "__main__":
    unittest.main()
