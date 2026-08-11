"""Tests for deterministic Ollama runtime options."""

import json
import unittest

from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.model_client import OllamaModelClient


class RecordingOllamaModelClient(OllamaModelClient):
    """Record payloads without performing network requests."""

    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[dict[str, object]] = []

    def _request_ollama(
        self,
        payload: dict[str, object],
    ) -> tuple[str, str]:
        self.payloads.append(payload)

        if len(self.payloads) == 1:
            return (
                "qwen3.5:9b",
                json.dumps(
                    {
                        "name": "conversation",
                        "parameters": {},
                        "conversation": {
                            "mode": "standard",
                            "target_text": None,
                        },
                    }
                ),
            )

        return (
            "qwen3.5:9b",
            "R?ponse conversationnelle.",
        )


class ModelRuntimeOptionsTests(unittest.TestCase):
    """Validate the runtime used by both model phases."""

    def test_both_phases_use_validated_runtime_options(
        self,
    ) -> None:
        client = RecordingOllamaModelClient()

        client.generate_response(
            (
                ConversationMessage(
                    role="user",
                    content="Bonjour.",
                ),
            )
        )

        self.assertEqual(
            len(client.payloads),
            2,
        )

        for payload in client.payloads:
            self.assertEqual(
                payload["model"],
                "qwen3.5:9b",
            )
            self.assertFalse(
                payload["think"],
            )
            self.assertEqual(
                payload["keep_alive"],
                "10m",
            )

            options = payload["options"]

            self.assertIsInstance(
                options,
                dict,
            )
            self.assertEqual(
                options["temperature"],
                0,
            )
            self.assertEqual(
                options["num_ctx"],
                4096,
            )


if __name__ == "__main__":
    unittest.main()
