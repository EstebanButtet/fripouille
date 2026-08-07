"""Tests for structured intents and language model responses."""

from __future__ import annotations

import unittest

from assistant_ia.intelligence.intent import (
    ALLOWED_INTENT_NAMES,
    INTENT_PARAMETER_SPECIFICATIONS,
    Intent,
)
from assistant_ia.intelligence.response import ModelResponse


class IntentTests(unittest.TestCase):
    """Validate structured assistant intents."""

    def test_normalizes_and_protects_parameters(self) -> None:
        """Intent parameters should be normalized, copied and immutable."""
        source_parameters = {
            "title": " Réviser la biologie ",
            "due_at": " demain ",
        }

        intent = Intent(
            name="create_task",
            parameters=source_parameters,
        )

        source_parameters["title"] = "Dormir"

        self.assertEqual(intent.name, "create_task")
        self.assertEqual(
            dict(intent.parameters),
            {
                "title": "Réviser la biologie",
                "due_at": "demain",
            },
        )

        with self.assertRaises(TypeError):
            intent.parameters["title"] = "Dormir"

    def test_accepts_empty_parameters(self) -> None:
        """Conversation intents should accept an empty parameter mapping."""
        intent = Intent(name="conversation")

        self.assertEqual(intent.name, "conversation")
        self.assertEqual(dict(intent.parameters), {})

    def test_rejects_unknown_intent_name(self) -> None:
        """Intent names outside the allowed set should be rejected."""
        with self.assertRaisesRegex(
            ValueError,
            "Unknown intent name",
        ):
            Intent(name="open_program")

    def test_rejects_non_string_parameter_value(self) -> None:
        """Intent parameter values should always be strings."""
        with self.assertRaisesRegex(
            TypeError,
            "Intent parameter values must be strings",
        ):
            Intent(
                name="create_task",
                parameters={"priority": 1},
            )

    def test_rejects_empty_parameter_value(self) -> None:
        """Intent parameter values should not be empty."""
        with self.assertRaisesRegex(
            ValueError,
            "Intent parameter values cannot be empty",
        ):
            Intent(
                name="create_task",
                parameters={"title": "   "},
            )

    def test_parameter_specifications_cover_allowed_intents(self) -> None:
        """Every allowed intent should have one parameter contract."""
        self.assertEqual(
            frozenset(INTENT_PARAMETER_SPECIFICATIONS),
            ALLOWED_INTENT_NAMES,
        )

        for specification in INTENT_PARAMETER_SPECIFICATIONS.values():
            self.assertTrue(
                specification.required.isdisjoint(
                    specification.optional
                )
            )

    def test_create_task_parameter_specification(self) -> None:
        """Task creation should expose its exact parameter contract."""
        specification = INTENT_PARAMETER_SPECIFICATIONS["create_task"]

        self.assertEqual(
            specification.required,
            frozenset(
                {
                    "title",
                }
            ),
        )
        self.assertEqual(
            specification.optional,
            frozenset(
                {
                    "due_at",
                }
            ),
        )


class ModelResponseTests(unittest.TestCase):
    """Validate structured language model responses."""

    def test_normalizes_valid_response(self) -> None:
        """Response content and model names should be normalized."""
        intent = Intent(name="conversation")

        response = ModelResponse(
            content=" Réponse simulée. ",
            model=" fake-model ",
            intent=intent,
        )

        self.assertEqual(response.content, "Réponse simulée.")
        self.assertEqual(response.model, "fake-model")
        self.assertIs(response.intent, intent)

    def test_rejects_non_intent_object(self) -> None:
        """Model responses should contain a validated Intent instance."""
        with self.assertRaisesRegex(
            TypeError,
            "Model response intent must be an Intent",
        ):
            ModelResponse(
                content="Réponse.",
                model="fake-model",
                intent={"name": "conversation"},
            )


if __name__ == "__main__":
    unittest.main()
