"""Tests for structured turn interpretation in OllamaModelClient."""

import json
import unittest

from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.conversation import ConversationMode
from assistant_ia.intelligence.interpretation import TurnInterpretation
from assistant_ia.intelligence.model_client import (
    ModelClientError,
    OllamaModelClient,
)
from assistant_ia.intelligence.turn import build_conversation_turn


class RecordingInterpretationClient(OllamaModelClient):
    """Return controlled structured interpretation content."""

    def __init__(
        self,
        response_data: dict[str, object],
    ) -> None:
        super().__init__()
        self.response_data = response_data
        self.payloads: list[dict[str, object]] = []

    def _request_ollama(
        self,
        payload: dict[str, object],
    ) -> tuple[str, str]:
        self.payloads.append(payload)

        return (
            "qwen3.5:9b",
            json.dumps(
                self.response_data
            ),
        )


def build_turn():
    """Build one current user turn."""
    return build_conversation_turn(
        (
            ConversationMessage(
                role="user",
                content=(
                    "J'ai exactement trois heures. "
                    "Repartis-les entre mes deux examens."
                ),
            ),
        )
    )


class TurnInterpretationModelClientTests(unittest.TestCase):
    """Validate parsing of intent and conversation metadata."""

    def test_standard_conversation_returns_turn_interpretation(
        self,
    ) -> None:
        client = RecordingInterpretationClient(
            {
                "name": "conversation",
                "parameters": {},
                "conversation": {
                    "mode": "standard",
                    "target_text": None,
                },
            }
        )

        interpretation = client._interpret_turn(
            build_turn()
        )

        self.assertIsInstance(
            interpretation,
            TurnInterpretation,
        )
        self.assertEqual(
            interpretation.intent.name,
            "conversation",
        )
        self.assertEqual(
            interpretation.conversation_directive_proposal.mode,
            ConversationMode.STANDARD,
        )
        self.assertEqual(
            interpretation.model,
            "qwen3.5:9b",
        )

    def test_fixed_total_metadata_is_preserved_as_proposal(
        self,
    ) -> None:
        client = RecordingInterpretationClient(
            {
                "name": "conversation",
                "parameters": {},
                "conversation": {
                    "mode": "fixed_total_allocation",
                    "target_text": "trois heures",
                },
            }
        )

        interpretation = client._interpret_turn(
            build_turn()
        )

        proposal = (
            interpretation.conversation_directive_proposal
        )

        self.assertEqual(
            proposal.mode,
            ConversationMode.FIXED_TOTAL_ALLOCATION,
        )
        self.assertEqual(
            proposal.target_text,
            "trois heures",
        )

    def test_action_cannot_return_fixed_total_metadata(
        self,
    ) -> None:
        client = RecordingInterpretationClient(
            {
                "name": "launch_application",
                "parameters": {
                    "application": "Edge",
                },
                "conversation": {
                    "mode": "fixed_total_allocation",
                    "target_text": "trois heures",
                },
            }
        )

        with self.assertRaises(
            ModelClientError
        ):
            client._interpret_turn(
                build_turn()
            )

    def test_invalid_conversation_metadata_fields_are_rejected(
        self,
    ) -> None:
        client = RecordingInterpretationClient(
            {
                "name": "conversation",
                "parameters": {},
                "conversation": {
                    "mode": "standard",
                    "target_text": None,
                    "reasoning": "hidden",
                },
            }
        )

        with self.assertRaises(
            ModelClientError
        ):
            client._interpret_turn(
                build_turn()
            )

    def test_fixed_total_requires_string_target_text(
        self,
    ) -> None:
        client = RecordingInterpretationClient(
            {
                "name": "conversation",
                "parameters": {},
                "conversation": {
                    "mode": "fixed_total_allocation",
                    "target_text": None,
                },
            }
        )

        with self.assertRaises(
            ModelClientError
        ):
            client._interpret_turn(
                build_turn()
            )


if __name__ == "__main__":
    unittest.main()
