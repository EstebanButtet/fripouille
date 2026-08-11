from __future__ import annotations

import json
import unittest
from urllib.request import Request
from unittest.mock import patch

from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.model_client import OllamaModelClient
from assistant_ia.intelligence.prompt import (
    INTERPRETATION_RESPONSE_SCHEMA,
)


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


def ollama_response(
    content: str,
) -> FakeHTTPResponse:
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


class TwoStageModelClientTests(unittest.TestCase):
    def _build_fake_urlopen(
        self,
        *,
        intent_name: str,
        parameters: dict[str, str],
    ):
        captured_payloads: list[dict[str, object]] = []

        def fake_urlopen(
            request: Request,
            timeout: float,
        ) -> FakeHTTPResponse:
            payload = json.loads(
                request.data.decode("utf-8")
            )
            captured_payloads.append(payload)

            response_format = payload.get("format")

            if (
                isinstance(response_format, dict)
                and set(
                    response_format.get(
                        "properties",
                        {}
                    )
                )
                == {"name", "parameters", "conversation"}
            ):
                return ollama_response(
                    json.dumps(
                        {
                            "name": intent_name,
                            "parameters": parameters,
                            "conversation": {
                                "mode": "standard",
                                "target_text": None,
                            },
                        }
                    )
                )

            if response_format is None:
                return ollama_response(
                    "Réponse conversationnelle dédiée."
                )

            # Compatibility response for the current one-call client.
            return ollama_response(
                json.dumps(
                    {
                        "content": "Ancienne réponse combinée.",
                        "intent": {
                            "name": intent_name,
                            "parameters": parameters,
                        },
                    }
                )
            )

        return captured_payloads, fake_urlopen

    def test_conversation_uses_two_ollama_calls(
        self,
    ) -> None:
        captured_payloads, fake_urlopen = (
            self._build_fake_urlopen(
                intent_name="conversation",
                parameters={},
            )
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            side_effect=fake_urlopen,
        ):
            response = OllamaModelClient().generate_response(
                (
                    ConversationMessage(
                        role="user",
                        content="Que penses-tu de mon projet ?",
                    ),
                )
            )

        self.assertEqual(
            len(captured_payloads),
            2,
        )
        self.assertEqual(
            response.intent.name,
            "conversation",
        )
        self.assertEqual(
            response.content,
            "Réponse conversationnelle dédiée.",
        )

    def test_first_call_is_interpretation_only(
        self,
    ) -> None:
        captured_payloads, fake_urlopen = (
            self._build_fake_urlopen(
                intent_name="conversation",
                parameters={},
            )
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            side_effect=fake_urlopen,
        ):
            OllamaModelClient().generate_response(
                (
                    ConversationMessage(
                        role="user",
                        content="Bonjour.",
                    ),
                )
            )

        interpretation_payload = captured_payloads[0]
        interpretation_prompt = (
            interpretation_payload["messages"][0]["content"]
        )

        self.assertEqual(
            interpretation_payload["format"],
            INTERPRETATION_RESPONSE_SCHEMA,
        )
        self.assertIn(
            "Intent classification and action interpretation rules:",
            interpretation_prompt,
        )
        self.assertNotIn(
            "Assistant identity",
            interpretation_prompt,
        )
        self.assertNotIn(
            "Conversational response quality rules:",
            interpretation_prompt,
        )

    def test_second_call_is_natural_conversation_only(
        self,
    ) -> None:
        captured_payloads, fake_urlopen = (
            self._build_fake_urlopen(
                intent_name="conversation",
                parameters={},
            )
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            side_effect=fake_urlopen,
        ):
            OllamaModelClient().generate_response(
                (
                    ConversationMessage(
                        role="user",
                        content="Bonjour.",
                    ),
                )
            )

        conversation_payload = captured_payloads[1]
        conversation_prompt = (
            conversation_payload["messages"][0]["content"]
        )

        self.assertNotIn(
            "format",
            conversation_payload,
        )
        self.assertIn(
            "Current conversation participants:",
            conversation_prompt,
        )
        self.assertIn(
            "Conversational response quality rules:",
            conversation_prompt,
        )
        self.assertIn(
            "Assistant identity",
            conversation_prompt,
        )
        self.assertNotIn(
            "Intent classification and action interpretation rules:",
            conversation_prompt,
        )

    def test_action_uses_only_interpretation_call(
        self,
    ) -> None:
        captured_payloads, fake_urlopen = (
            self._build_fake_urlopen(
                intent_name="launch_application",
                parameters={
                    "application": "Edge",
                },
            )
        )

        with patch(
            "assistant_ia.intelligence.model_client.urlopen",
            side_effect=fake_urlopen,
        ):
            response = OllamaModelClient().generate_response(
                (
                    ConversationMessage(
                        role="user",
                        content="Lance Edge.",
                    ),
                )
            )

        self.assertEqual(
            len(captured_payloads),
            1,
        )
        self.assertEqual(
            captured_payloads[0]["format"],
            INTERPRETATION_RESPONSE_SCHEMA,
        )
        self.assertEqual(
            response.intent.name,
            "launch_application",
        )
        self.assertEqual(
            dict(response.intent.parameters),
            {
                "application": "Edge",
            },
        )


if __name__ == "__main__":
    unittest.main()
