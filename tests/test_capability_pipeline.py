"""Tests for capability context integration with the model pipeline."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.request import Request

from assistant_ia.application import build_default_assistant
from assistant_ia.capabilities.context import CapabilityContext
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
        "content": "OK.",
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
                    structured_content
                ),
            },
        }
    ).encode("utf-8")


class CapabilityPipelineTests(unittest.TestCase):
    def test_prompt_places_capabilities_before_conversation_rules(
        self,
    ) -> None:
        identity = build_default_identity()
        person_context = ActivePersonContext(
            assistant_name=identity.name,
            default_person=PersonProfile(
                name="Este",
            ),
        )
        capability_context = CapabilityContext(
            available_actions=(
                "save_memory",
                "launch_application",
            ),
        )

        prompt = build_system_prompt(
            identity,
            person_context,
            capability_context,
        )

        self.assertIn(
            "Current assistant capabilities:",
            prompt,
        )
        self.assertIn(
            "- launch_application",
            prompt,
        )
        self.assertIn(
            "- save_memory",
            prompt,
        )
        self.assertLess(
            prompt.index(
                "Current assistant capabilities:"
            ),
            prompt.index(
                "Conversational response quality rules:"
            ),
        )

    def test_model_client_uses_capability_context(
        self,
    ) -> None:
        """Conversation generation should receive current capabilities."""
        identity = build_default_identity()
        person_context = ActivePersonContext(
            assistant_name=identity.name,
            default_person=PersonProfile(
                name="Este",
            ),
        )
        capability_context = CapabilityContext(
            available_actions=(
                "save_memory",
                "launch_application",
            ),
        )
        client = OllamaModelClient(
            identity=identity,
            person_context=person_context,
            capability_context=capability_context,
        )
        captured_payloads: list[dict[str, object]] = []

        def fake_urlopen(
            request: Request,
            timeout: float,
        ) -> FakeHTTPResponse:
            payload = json.loads(
                request.data.decode("utf-8")
            )
            captured_payloads.append(
                payload
            )

            if "format" in payload:
                content = json.dumps(
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
                content = (
                    "R\u00e9ponse "
                    "conversationnelle."
                )

            return FakeHTTPResponse(
                json.dumps(
                    {
                        "model": "qwen3.5:4b",
                        "message": {
                            "content": content,
                        },
                    }
                ).encode("utf-8")
            )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            side_effect=fake_urlopen,
        ):
            client.generate_response(
                (
                    ConversationMessage(
                        role="user",
                        content="Bonjour.",
                    ),
                )
            )

        self.assertEqual(
            len(captured_payloads),
            2,
        )

        interpretation_prompt = (
            captured_payloads[0][
                "messages"
            ][0]["content"]
        )
        conversation_prompt = (
            captured_payloads[1][
                "messages"
            ][0]["content"]
        )

        self.assertNotIn(
            "Current assistant capabilities:",
            interpretation_prompt,
        )
        self.assertIn(
            "- launch_application",
            conversation_prompt,
        )
        self.assertIn(
            "- save_memory",
            conversation_prompt,
        )

    def test_application_derives_capabilities_from_registry(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            database = SQLiteDatabase(
                Path(directory) / "assistant.db"
            )

            with patch(
                "assistant_ia.application.OllamaModelClient"
            ) as model_client_class:
                assistant = build_default_assistant(
                    database=database,
                )

            capability_context = (
                model_client_class.call_args.kwargs[
                    "capability_context"
                ]
            )

            self.assertEqual(
                capability_context.available_actions,
                tuple(
                    sorted(
                        assistant.action_registry.action_names
                    )
                ),
            )
            self.assertIn(
                "save_memory",
                capability_context.available_actions,
            )
            self.assertIn(
                "launch_application",
                capability_context.available_actions,
            )


if __name__ == "__main__":
    unittest.main()
