"""Tests for deterministic allocation generation through Ollama."""

from decimal import Decimal
import unittest

from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.allocation import (
    ALLOCATION_PROPOSAL_SCHEMA,
    AllocationTarget,
    FixedTotalAllocation,
)
from assistant_ia.intelligence.model_client import (
    ModelClientError,
    OllamaModelClient,
)
from assistant_ia.intelligence.turn import build_conversation_turn


class RecordingAllocationClient(OllamaModelClient):
    """Record allocation requests and return controlled model content."""

    def __init__(
        self,
        response_content: str,
    ) -> None:
        super().__init__()
        self.response_content = response_content
        self.payloads: list[dict[str, object]] = []

    def _request_ollama(
        self,
        payload: dict[str, object],
    ) -> tuple[str, str]:
        self.payloads.append(payload)

        return (
            "qwen3.5:9b",
            self.response_content,
        )


def build_turn():
    """Build one simple fixed-total conversation turn."""
    return build_conversation_turn(
        (
            ConversationMessage(
                role="user",
                content=(
                    "I have exactly 180 minutes. "
                    "Allocate them between two exams and a break."
                ),
            ),
        )
    )


def build_target():
    """Build the authoritative application-owned target."""
    return AllocationTarget(
        total=Decimal("180"),
        unit="minutes",
    )


class AllocationModelClientTests(unittest.TestCase):
    """Validate the hidden allocation-generation primitive."""

    def test_exact_proposal_returns_validated_allocation(
        self,
    ) -> None:
        client = RecordingAllocationClient(
            """{
                "parts": [
                    {
                        "label": "Second exam",
                        "amount": 95
                    },
                    {
                        "label": "Break",
                        "amount": 15
                    },
                    {
                        "label": "First exam",
                        "amount": 70
                    }
                ]
            }"""
        )

        allocation = client._generate_fixed_total_allocation(
            build_turn(),
            build_target(),
        )

        self.assertIsInstance(
            allocation,
            FixedTotalAllocation,
        )
        self.assertEqual(
            allocation.total,
            Decimal("180"),
        )
        self.assertEqual(
            allocation.allocated_total,
            Decimal("180"),
        )
        self.assertTrue(
            allocation.is_exact
        )

    def test_request_uses_validated_runtime_and_schema(
        self,
    ) -> None:
        client = RecordingAllocationClient(
            """{
                "parts": [
                    {
                        "label": "Study",
                        "amount": 180
                    }
                ]
            }"""
        )

        client._generate_fixed_total_allocation(
            build_turn(),
            build_target(),
        )

        self.assertEqual(
            len(client.payloads),
            1,
        )

        payload = client.payloads[0]

        self.assertEqual(
            payload["model"],
            "qwen3.5:9b",
        )
        self.assertEqual(
            payload["format"],
            ALLOCATION_PROPOSAL_SCHEMA,
        )
        self.assertFalse(
            payload["stream"],
        )
        self.assertFalse(
            payload["think"],
        )
        self.assertEqual(
            payload["keep_alive"],
            "10m",
        )
        self.assertEqual(
            payload["options"],
            {
                "temperature": 0,
                "num_ctx": 4096,
            },
        )

        messages = payload["messages"]

        self.assertIsInstance(
            messages,
            list,
        )

        system_prompt = messages[0]["content"]

        self.assertIn(
            "total: 180",
            system_prompt,
        )
        self.assertIn(
            "unit: minutes",
            system_prompt,
        )
        self.assertIn(
            "Return only allocation parts.",
            system_prompt,
        )

    def test_inexact_model_proposal_is_rejected(
        self,
    ) -> None:
        client = RecordingAllocationClient(
            """{
                "parts": [
                    {
                        "label": "Study",
                        "amount": 100
                    },
                    {
                        "label": "Break",
                        "amount": 20
                    }
                ]
            }"""
        )

        with self.assertRaises(
            ModelClientError
        ):
            client._generate_fixed_total_allocation(
                build_turn(),
                build_target(),
            )

    def test_malformed_model_proposal_is_rejected(
        self,
    ) -> None:
        client = RecordingAllocationClient(
            """{
                "parts": [
                    {
                        "label": "Study",
                        "amount": "180"
                    }
                ]
            }"""
        )

        with self.assertRaises(
            ModelClientError
        ):
            client._generate_fixed_total_allocation(
                build_turn(),
                build_target(),
            )


if __name__ == "__main__":
    unittest.main()
