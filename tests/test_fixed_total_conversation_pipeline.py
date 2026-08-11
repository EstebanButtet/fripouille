"""Tests for the complete fixed-total conversational pipeline."""

import json
import unittest

from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.allocation import (
    ALLOCATION_PROPOSAL_SCHEMA,
)
from assistant_ia.intelligence.model_client import OllamaModelClient
from assistant_ia.intelligence.prompt import (
    INTERPRETATION_RESPONSE_SCHEMA,
)


class PipelineClient(OllamaModelClient):
    """Return controlled responses for each internal model phase."""

    def __init__(
        self,
        *,
        interpretation: dict[str, object],
        allocation: dict[str, object] | None = None,
        conversation: str = "R?ponse conversationnelle.",
    ) -> None:
        super().__init__()

        self.interpretation = interpretation
        self.allocation = allocation
        self.conversation = conversation
        self.payloads: list[dict[str, object]] = []

    def _request_ollama(
        self,
        payload: dict[str, object],
    ) -> tuple[str, str]:
        self.payloads.append(payload)

        response_format = payload.get(
            "format"
        )

        if response_format == INTERPRETATION_RESPONSE_SCHEMA:
            return (
                "qwen3.5:9b",
                json.dumps(
                    self.interpretation
                ),
            )

        if response_format == ALLOCATION_PROPOSAL_SCHEMA:
            if self.allocation is None:
                raise AssertionError(
                    "Unexpected allocation request."
                )

            return (
                "qwen3.5:9b",
                json.dumps(
                    self.allocation
                ),
            )

        return (
            "qwen3.5:9b",
            self.conversation,
        )


class FixedTotalConversationPipelineTests(unittest.TestCase):
    """Validate routing from interpretation to certified allocation."""

    def test_exact_fixed_total_uses_certified_allocation_path(
        self,
    ) -> None:
        client = PipelineClient(
            interpretation={
                "name": "conversation",
                "parameters": {},
                "conversation": {
                    "mode": "fixed_total_allocation",
                    "target_text": "trois heures",
                },
            },
            allocation={
                "parts": [
                    {
                        "label": "Examen 2",
                        "amount": 95,
                    },
                    {
                        "label": "Pause",
                        "amount": 15,
                    },
                    {
                        "label": "Examen 1",
                        "amount": 70,
                    },
                ],
            },
        )

        response = client.generate_response(
            (
                ConversationMessage(
                    role="user",
                    content=(
                        "J'ai exactement trois heures ce soir. "
                        "R?partis-les entre mes deux examens."
                    ),
                ),
            )
        )

        self.assertEqual(
            response.intent.name,
            "conversation",
        )

        self.assertEqual(
            len(client.payloads),
            2,
        )

        self.assertEqual(
            client.payloads[0]["format"],
            INTERPRETATION_RESPONSE_SCHEMA,
        )

        self.assertEqual(
            client.payloads[1]["format"],
            ALLOCATION_PROPOSAL_SCHEMA,
        )

        self.assertIn(
            "180 minutes",
            response.content,
        )
        self.assertIn(
            "Examen 2 : 95 minutes",
            response.content,
        )
        self.assertIn(
            "Pause : 15 minutes",
            response.content,
        )
        self.assertIn(
            "Examen 1 : 70 minutes",
            response.content,
        )
        self.assertIn(
            "Total : 180 minutes",
            response.content,
        )

    def test_standard_conversation_still_uses_natural_generation(
        self,
    ) -> None:
        client = PipelineClient(
            interpretation={
                "name": "conversation",
                "parameters": {},
                "conversation": {
                    "mode": "standard",
                    "target_text": None,
                },
            },
            conversation=(
                "R?ponse naturelle normale."
            ),
        )

        response = client.generate_response(
            (
                ConversationMessage(
                    role="user",
                    content="Explique-moi la photosynth?se.",
                ),
            )
        )

        self.assertEqual(
            len(client.payloads),
            2,
        )
        self.assertNotIn(
            "format",
            client.payloads[1],
        )
        self.assertEqual(
            response.content,
            "R?ponse naturelle normale.",
        )

    def test_untrusted_invalid_fixed_total_falls_back_to_standard(
        self,
    ) -> None:
        client = PipelineClient(
            interpretation={
                "name": "conversation",
                "parameters": {},
                "conversation": {
                    "mode": "fixed_total_allocation",
                    "target_text": "quatre heures",
                },
            },
            conversation=(
                "R?ponse conversationnelle de secours."
            ),
        )

        response = client.generate_response(
            (
                ConversationMessage(
                    role="user",
                    content=(
                        "J'ai exactement trois heures ce soir."
                    ),
                ),
            )
        )

        self.assertEqual(
            len(client.payloads),
            2,
        )
        self.assertNotIn(
            "format",
            client.payloads[1],
        )
        self.assertEqual(
            response.content,
            "R?ponse conversationnelle de secours.",
        )


if __name__ == "__main__":
    unittest.main()
