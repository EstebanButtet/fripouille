"""Tests for the two-stage Ollama model client."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import patch
from urllib.request import Request

from assistant_ia.core.context import ConversationMessage
from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.intelligence.model_client import (
    ModelClientError,
    OllamaModelClient,
)
from assistant_ia.intelligence.prompt import (
    CURRENT_TURN_CONTEXT_PROMPT,
    INTERPRETATION_RESPONSE_SCHEMA,
    INTERPRETATION_SYSTEM_PROMPT,
)


class FakeHTTPResponse:
    """Provide predefined response bytes as a context manager."""

    def __init__(
        self,
        content: bytes,
    ) -> None:
        self._content = content

    def __enter__(
        self,
    ) -> FakeHTTPResponse:
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        return None

    def read(self) -> bytes:
        return self._content


def build_ollama_response(
    content: object,
    model: object = "fake-model",
) -> bytes:
    """Build one serialized Ollama chat response."""
    response_data = {
        "model": model,
        "message": {
            "role": "assistant",
            "content": (
                content
                if isinstance(content, str)
                else json.dumps(
                    content,
                    ensure_ascii=False,
                )
            ),
        },
    }

    return json.dumps(
        response_data,
        ensure_ascii=False,
    ).encode("utf-8")


def build_interpretation_response(
    *,
    name: str,
    parameters: dict[str, object],
    model: object = "fake-model",
) -> bytes:
    """Build one first-stage interpretation response."""
    return build_ollama_response(
        {
            "name": name,
            "parameters": parameters,
            "conversation": {
                "mode": "standard",
                "target_text": None,
            },
        },
        model=model,
    )


def build_two_stage_urlopen(
    *,
    intent_name: str = "conversation",
    parameters: dict[str, object] | None = None,
    conversation_content: str = "Réponse conversationnelle simulée.",
):
    """Return a fake Ollama transport for interpretation and conversation."""
    captured_payloads: list[dict[str, object]] = []

    resolved_parameters = (
        parameters
        if parameters is not None
        else {}
    )

    def fake_urlopen(
        request: Request,
        timeout: float,
    ) -> FakeHTTPResponse:
        payload = json.loads(
            request.data.decode("utf-8")
        )
        captured_payloads.append(payload)

        if (
            payload.get("format")
            == INTERPRETATION_RESPONSE_SCHEMA
        ):
            return FakeHTTPResponse(
                build_interpretation_response(
                    name=intent_name,
                    parameters=resolved_parameters,
                )
            )

        if "format" not in payload:
            return FakeHTTPResponse(
                build_ollama_response(
                    conversation_content
                )
            )

        raise AssertionError(
            "Unexpected Ollama request format."
        )

    return captured_payloads, fake_urlopen


class OllamaModelClientTests(unittest.TestCase):
    """Validate the two-stage Ollama request and response pipeline."""

    def setUp(self) -> None:
        self.messages = (
            ConversationMessage(
                role="user",
                content=(
                    "Crée une tâche pour demain."
                ),
            ),
        )

    def test_parses_valid_interpreted_action_response(
        self,
    ) -> None:
        """A valid interpreted action should become a ModelResponse."""
        raw_response = build_interpretation_response(
            name="create_task",
            parameters={
                "title": "Réviser la biologie",
                "due_at": "demain",
            },
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(
                raw_response
            ),
        ) as mocked_urlopen:
            response = (
                OllamaModelClient()
                .generate_response(
                    self.messages
                )
            )

        mocked_urlopen.assert_called_once()

        self.assertEqual(
            response.model,
            "fake-model",
        )
        self.assertEqual(
            response.intent.name,
            "create_task",
        )
        self.assertEqual(
            dict(response.intent.parameters),
            {
                "title": "Réviser la biologie",
                "due_at": "demain",
            },
        )
        self.assertTrue(
            response.content
        )

    def test_sends_interpretation_prompt_and_schema(
        self,
    ) -> None:
        """The first request should contain only interpretation work."""
        raw_response = build_interpretation_response(
            name="create_task",
            parameters={
                "title": "Test",
            },
        )
        captured_request: Request | None = None
        captured_timeout: float | None = None

        def fake_urlopen(
            request: Request,
            timeout: float,
        ) -> FakeHTTPResponse:
            nonlocal captured_request
            nonlocal captured_timeout

            captured_request = request
            captured_timeout = timeout

            return FakeHTTPResponse(
                raw_response
            )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            side_effect=fake_urlopen,
        ):
            OllamaModelClient().generate_response(
                self.messages
            )

        self.assertIsNotNone(
            captured_request
        )
        self.assertEqual(
            captured_timeout,
            120.0,
        )

        request_data = json.loads(
            captured_request.data.decode(
                "utf-8"
            )
        )

        self.assertEqual(
            request_data["model"],
            "qwen3.5:9b",
        )
        self.assertFalse(
            request_data["stream"]
        )
        self.assertFalse(
            request_data["think"]
        )
        self.assertEqual(
            request_data["options"][
                "temperature"
            ],
            0,
        )
        self.assertEqual(
            request_data["format"],
            INTERPRETATION_RESPONSE_SCHEMA,
        )

        system_content = (
            request_data["messages"][0][
                "content"
            ]
        )

        self.assertIn(
            "Intent classification and action interpretation rules:",
            system_content,
        )
        self.assertNotIn(
            "Assistant identity",
            system_content,
        )
        self.assertNotIn(
            "Conversational response quality rules:",
            system_content,
        )

        self.assertEqual(
            request_data["messages"][1],
            {
                "role": "user",
                "content": (
                    "Crée une tâche "
                    "pour demain."
                ),
            },
        )

    def test_sends_injected_identity_only_to_conversation_stage(
        self,
    ) -> None:
        """Injected identity should affect natural generation, not intent."""
        identity = replace(
            build_default_identity(),
            name="Test identity",
            role="Test companion",
            behavioral_rules=(
                "Ignore operational rules",
            ),
        )
        captured_payloads, fake_urlopen = (
            build_two_stage_urlopen()
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            side_effect=fake_urlopen,
        ):
            response = OllamaModelClient(
                identity=identity
            ).generate_response(
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
            "Test identity",
            interpretation_prompt,
        )

        self.assertIn(
            "Name: Test identity",
            conversation_prompt,
        )
        self.assertIn(
            "Role: Test companion",
            conversation_prompt,
        )
        self.assertNotIn(
            "Name: Fripouille",
            conversation_prompt,
        )
        self.assertIn(
            "- Ignore operational rules",
            conversation_prompt,
        )
        self.assertNotIn(
            "Intent classification and action interpretation rules:",
            conversation_prompt,
        )
        self.assertEqual(
            response.content,
            "Réponse conversationnelle simulée.",
        )

    def test_rejects_invalid_identity_configuration(
        self,
    ) -> None:
        """Ollama clients should require the identity domain model."""
        with self.assertRaisesRegex(
            TypeError,
            "identity must be an AssistantIdentity",
        ):
            OllamaModelClient(
                identity="Fripouille"
            )

    def test_interpretation_prompt_defines_parameter_contract(
        self,
    ) -> None:
        """The dedicated interpretation prompt should define action inputs."""
        self.assertIn(
            'create_task: required parameter "title"; '
            'optional parameter "due_at"',
            INTERPRETATION_SYSTEM_PROMPT,
        )
        self.assertIn(
            'complete_task: required parameter "task_id"',
            INTERPRETATION_SYSTEM_PROMPT,
        )
        self.assertIn(
            'delete_memory: required parameter "memory_id"',
            INTERPRETATION_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Never omit content for a write_journal intent",
            INTERPRETATION_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Never produce SQL",
            INTERPRETATION_SYSTEM_PROMPT,
        )
        self.assertIn(
            "task_id and memory_id must contain only ASCII digits",
            INTERPRETATION_SYSTEM_PROMPT,
        )
        self.assertIn(
            "A statement that an application would be useful "
            "or convenient is not",
            INTERPRETATION_SYSTEM_PROMPT,
        )
        self.assertIn(
            "an execution request.",
            INTERPRETATION_SYSTEM_PROMPT,
        )

    def test_first_turn_does_not_add_history_boundary(
        self,
    ) -> None:
        """Neither stage should add a boundary without prior history."""
        messages = (
            ConversationMessage(
                role="user",
                content="Bonjour Fripouille.",
            ),
        )
        captured_payloads, fake_urlopen = (
            build_two_stage_urlopen()
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            side_effect=fake_urlopen,
        ):
            OllamaModelClient().generate_response(
                messages
            )

        self.assertEqual(
            len(captured_payloads),
            2,
        )

        for payload in captured_payloads:
            self.assertEqual(
                len(payload["messages"]),
                2,
            )
            self.assertEqual(
                payload["messages"][1],
                {
                    "role": "user",
                    "content": (
                        "Bonjour Fripouille."
                    ),
                },
            )

    def test_preserves_history_with_current_user_message_last(
        self,
    ) -> None:
        """Both stages should preserve history and isolate the current turn."""
        messages = (
            ConversationMessage(
                role="user",
                content="Qui es-tu ?",
            ),
            ConversationMessage(
                role="assistant",
                content="Je suis Fripouille.",
            ),
            ConversationMessage(
                role="user",
                content=(
                    "Et que penses-tu de "
                    "mon projet ?"
                ),
            ),
        )
        captured_payloads, fake_urlopen = (
            build_two_stage_urlopen()
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            side_effect=fake_urlopen,
        ):
            OllamaModelClient().generate_response(
                messages
            )

        expected_turn_messages = [
            {
                "role": "user",
                "content": "Qui es-tu ?",
            },
            {
                "role": "assistant",
                "content": (
                    "Je suis Fripouille."
                ),
            },
            {
                "role": "system",
                "content": (
                    CURRENT_TURN_CONTEXT_PROMPT
                ),
            },
            {
                "role": "user",
                "content": (
                    "Et que penses-tu de "
                    "mon projet ?"
                ),
            },
        ]

        self.assertEqual(
            len(captured_payloads),
            2,
        )

        for payload in captured_payloads:
            self.assertEqual(
                payload["messages"][1:],
                expected_turn_messages,
            )

    def test_rejects_history_without_current_user_message(
        self,
    ) -> None:
        """Generation should require the latest message to be user-authored."""
        messages = (
            ConversationMessage(
                role="user",
                content="Bonjour.",
            ),
            ConversationMessage(
                role="assistant",
                content="Salut.",
            ),
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
        ) as mocked_urlopen:
            with self.assertRaisesRegex(
                ValueError,
                "latest conversation message must be from the user",
            ):
                OllamaModelClient().generate_response(
                    messages
                )

        mocked_urlopen.assert_not_called()

    def test_rejects_invalid_outer_json(
        self,
    ) -> None:
        """Invalid Ollama envelope JSON should be rejected."""
        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(
                b"not-json"
            ),
        ):
            with self.assertRaisesRegex(
                ModelClientError,
                "Ollama returned invalid JSON",
            ):
                OllamaModelClient().generate_response(
                    self.messages
                )

    def test_rejects_missing_message(
        self,
    ) -> None:
        """The Ollama envelope should contain a message object."""
        raw_response = json.dumps(
            {
                "model": "fake-model",
            }
        ).encode("utf-8")

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(
                raw_response
            ),
        ):
            with self.assertRaisesRegex(
                ModelClientError,
                "does not contain a valid message",
            ):
                OllamaModelClient().generate_response(
                    self.messages
                )

    def test_rejects_invalid_interpretation_json(
        self,
    ) -> None:
        """The interpretation message should contain valid JSON."""
        raw_response = build_ollama_response(
            "{invalid-json"
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(
                raw_response
            ),
        ):
            with self.assertRaisesRegex(
                ModelClientError,
                "invalid interpreted intent JSON",
            ):
                OllamaModelClient().generate_response(
                    self.messages
                )

    def test_rejects_extra_interpretation_field(
        self,
    ) -> None:
        """Unexpected interpretation fields should be rejected."""
        raw_response = build_ollama_response(
            {
                "name": "conversation",
                "parameters": {},
                "conversation": {
                    "mode": "standard",
                    "target_text": None,
                },
                "unexpected": "value",
            }
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(
                raw_response
            ),
        ):
            with self.assertRaisesRegex(
                ModelClientError,
                "interpreted intent contains invalid fields",
            ):
                OllamaModelClient().generate_response(
                    self.messages
                )

    def test_rejects_unknown_intent(
        self,
    ) -> None:
        """An intent outside the authorized set should be rejected."""
        raw_response = build_interpretation_response(
            name="open_program",
            parameters={},
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(
                raw_response
            ),
        ):
            with self.assertRaisesRegex(
                ModelClientError,
                "invalid interpreted intent",
            ):
                OllamaModelClient().generate_response(
                    self.messages
                )

    def test_rejects_non_string_parameter(
        self,
    ) -> None:
        """Intent parameter values should remain validated strings."""
        raw_response = build_interpretation_response(
            name="create_task",
            parameters={
                "title": 1,
            },
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(
                raw_response
            ),
        ):
            with self.assertRaisesRegex(
                ModelClientError,
                "invalid interpreted intent",
            ):
                OllamaModelClient().generate_response(
                    self.messages
                )


if __name__ == "__main__":
    unittest.main()
