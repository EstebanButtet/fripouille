"""Tests for the structured Ollama model client."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from urllib.request import Request

from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.model_client import (
    INTENT_SYSTEM_PROMPT,
    ModelClientError,
    OllamaModelClient,
)


class FakeHTTPResponse:
    """Provide predefined response bytes as a context manager."""

    def __init__(self, content: bytes) -> None:
        """Store the bytes returned by read()."""
        self._content = content

    def __enter__(self) -> FakeHTTPResponse:
        """Return this fake response."""
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        """Leave the fake response context."""

    def read(self) -> bytes:
        """Return the predefined response bytes."""
        return self._content


def build_ollama_response(
    structured_content: object,
    model: object = "fake-model",
) -> bytes:
    """Build a serialized response matching the Ollama API envelope."""
    response_data = {
        "model": model,
        "message": {
            "role": "assistant",
            "content": (
                structured_content
                if isinstance(structured_content, str)
                else json.dumps(
                    structured_content,
                    ensure_ascii=False,
                )
            ),
        },
    }

    return json.dumps(
        response_data,
        ensure_ascii=False,
    ).encode("utf-8")


class OllamaModelClientTests(unittest.TestCase):
    """Validate structured Ollama requests and responses."""

    def setUp(self) -> None:
        """Create one valid conversation message for each test."""
        self.messages = (
            ConversationMessage(
                role="user",
                content="Crée une tâche pour demain.",
            ),
        )

    def test_parses_valid_structured_response(self) -> None:
        """A valid structured response should become a ModelResponse."""
        raw_response = build_ollama_response(
            {
                "content": "La demande a été identifiée.",
                "intent": {
                    "name": "create_task",
                    "parameters": {
                        "title": "Réviser la biologie",
                        "due_at": "demain",
                    },
                },
            }
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(raw_response),
        ):
            response = OllamaModelClient().generate_response(self.messages)

        self.assertEqual(
            response.content,
            "La demande a été identifiée.",
        )
        self.assertEqual(response.model, "fake-model")
        self.assertEqual(response.intent.name, "create_task")
        self.assertEqual(
            dict(response.intent.parameters),
            {
                "title": "Réviser la biologie",
                "due_at": "demain",
            },
        )

    def test_sends_system_prompt_and_structured_schema(self) -> None:
        """The request should configure structured local generation."""
        raw_response = build_ollama_response(
            {
                "content": "Réponse simulée.",
                "intent": {
                    "name": "conversation",
                    "parameters": {},
                },
            }
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

            return FakeHTTPResponse(raw_response)

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            side_effect=fake_urlopen,
        ):
            OllamaModelClient().generate_response(self.messages)

        self.assertIsNotNone(captured_request)
        self.assertEqual(captured_timeout, 120.0)

        request_data = json.loads(captured_request.data.decode("utf-8"))

        self.assertEqual(request_data["model"], "qwen3.5:4b")
        self.assertFalse(request_data["stream"])
        self.assertFalse(request_data["think"])
        self.assertEqual(
            request_data["options"]["temperature"],
            0,
        )
        self.assertEqual(
            request_data["messages"][0],
            {
                "role": "system",
                "content": INTENT_SYSTEM_PROMPT,
            },
        )
        self.assertEqual(
            request_data["messages"][1],
            {
                "role": "user",
                "content": "Crée une tâche pour demain.",
            },
        )
        self.assertEqual(
            request_data["format"]["type"],
            "object",
        )

    def test_system_prompt_defines_parameter_contract(self) -> None:
        """The system prompt should define safe action parameters."""
        self.assertIn(
            'create_task: required parameter "title"; '
            'optional parameter "due_at"',
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            'complete_task: required parameter "task_id"',
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            'delete_memory: required parameter "memory_id"',
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "never claim that a task, memory, journal entry or",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Never produce SQL",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "task_id and memory_id must contain only ASCII digits",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Never omit content for a write_journal intent",
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            '"content": "TEST E8 journal local."',
            INTENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            '"entry_date": "2026-08-07"',
            INTENT_SYSTEM_PROMPT,
        )

    def test_rejects_invalid_outer_json(self) -> None:
        """Invalid JSON returned by Ollama should be rejected."""
        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(b"not-json"),
        ):
            with self.assertRaisesRegex(
                ModelClientError,
                "Ollama returned invalid JSON",
            ):
                OllamaModelClient().generate_response(self.messages)

    def test_rejects_missing_message(self) -> None:
        """The Ollama envelope should contain a message object."""
        raw_response = json.dumps(
            {
                "model": "fake-model",
            }
        ).encode("utf-8")

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(raw_response),
        ):
            with self.assertRaisesRegex(
                ModelClientError,
                "does not contain a valid message",
            ):
                OllamaModelClient().generate_response(self.messages)

    def test_rejects_invalid_structured_json(self) -> None:
        """The assistant message should contain valid structured JSON."""
        raw_response = build_ollama_response("{invalid-json")

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(raw_response),
        ):
            with self.assertRaisesRegex(
                ModelClientError,
                "invalid structured content",
            ):
                OllamaModelClient().generate_response(self.messages)

    def test_rejects_extra_structured_field(self) -> None:
        """Unexpected top-level structured fields should be rejected."""
        raw_response = build_ollama_response(
            {
                "content": "Réponse simulée.",
                "intent": {
                    "name": "conversation",
                    "parameters": {},
                },
                "unexpected": "value",
            }
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(raw_response),
        ):
            with self.assertRaisesRegex(
                ModelClientError,
                "contains invalid fields",
            ):
                OllamaModelClient().generate_response(self.messages)

    def test_rejects_unknown_intent(self) -> None:
        """An intent outside the authorized list should be rejected."""
        raw_response = build_ollama_response(
            {
                "content": "Réponse simulée.",
                "intent": {
                    "name": "open_program",
                    "parameters": {},
                },
            }
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(raw_response),
        ):
            with self.assertRaisesRegex(
                ModelClientError,
                "invalid structured model response",
            ):
                OllamaModelClient().generate_response(self.messages)

    def test_rejects_non_string_parameter(self) -> None:
        """Structured intent parameter values should be strings."""
        raw_response = build_ollama_response(
            {
                "content": "Réponse simulée.",
                "intent": {
                    "name": "create_task",
                    "parameters": {
                        "priority": 1,
                    },
                },
            }
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            return_value=FakeHTTPResponse(raw_response),
        ):
            with self.assertRaisesRegex(
                ModelClientError,
                "invalid structured model response",
            ):
                OllamaModelClient().generate_response(self.messages)


if __name__ == "__main__":
    unittest.main()
