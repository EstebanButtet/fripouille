"""Tests for conversation metadata in structured interpretation."""

import unittest

from assistant_ia.intelligence.prompt import (
    INTERPRETATION_RESPONSE_SCHEMA,
    build_interpretation_prompt,
)


class InterpretationDirectiveSchemaTests(unittest.TestCase):
    """Validate the extended interpretation contract."""

    def test_interpretation_requires_conversation_metadata(
        self,
    ) -> None:
        self.assertEqual(
            INTERPRETATION_RESPONSE_SCHEMA["required"],
            [
                "name",
                "parameters",
                "conversation",
            ],
        )

        self.assertFalse(
            INTERPRETATION_RESPONSE_SCHEMA[
                "additionalProperties"
            ]
        )

    def test_conversation_metadata_has_exact_fields(
        self,
    ) -> None:
        conversation_schema = (
            INTERPRETATION_RESPONSE_SCHEMA[
                "properties"
            ]["conversation"]
        )

        self.assertEqual(
            conversation_schema["required"],
            [
                "mode",
                "target_text",
            ],
        )

        self.assertEqual(
            set(
                conversation_schema[
                    "properties"
                ]
            ),
            {
                "mode",
                "target_text",
            },
        )

        self.assertFalse(
            conversation_schema[
                "additionalProperties"
            ]
        )

    def test_conversation_mode_has_exact_supported_values(
        self,
    ) -> None:
        mode_schema = (
            INTERPRETATION_RESPONSE_SCHEMA[
                "properties"
            ]["conversation"]["properties"]["mode"]
        )

        self.assertEqual(
            mode_schema["enum"],
            [
                "fixed_total_allocation",
                "standard",
            ],
        )

    def test_target_text_accepts_string_or_null(
        self,
    ) -> None:
        target_schema = (
            INTERPRETATION_RESPONSE_SCHEMA[
                "properties"
            ]["conversation"]["properties"]["target_text"]
        )

        self.assertEqual(
            target_schema["type"],
            [
                "string",
                "null",
            ],
        )

    def test_prompt_defines_fixed_total_routing_boundary(
        self,
    ) -> None:
        prompt = build_interpretation_prompt()
        normalized = " ".join(
            prompt.split()
        )

        self.assertIn(
            "fixed_total_allocation",
            normalized,
        )
        self.assertIn(
            "target_text",
            normalized,
        )
        self.assertIn(
            "exact duration fragment",
            normalized,
        )
        self.assertIn(
            "For every non-conversation intent, use standard",
            normalized,
        )


if __name__ == "__main__":
    unittest.main()
