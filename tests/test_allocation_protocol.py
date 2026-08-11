"""Tests for the structured allocation model protocol."""

from decimal import Decimal
import unittest

from assistant_ia.intelligence.allocation import (
    ALLOCATION_RESPONSE_SCHEMA,
    AllocationFormatError,
    FixedTotalAllocation,
    parse_allocation_payload,
)


class AllocationProtocolTests(unittest.TestCase):
    """Validate the hidden structured allocation contract."""

    def test_schema_requires_total_unit_and_parts(
        self,
    ) -> None:
        self.assertEqual(
            ALLOCATION_RESPONSE_SCHEMA["required"],
            [
                "total",
                "unit",
                "parts",
            ],
        )
        self.assertFalse(
            ALLOCATION_RESPONSE_SCHEMA["additionalProperties"]
        )

        properties = ALLOCATION_RESPONSE_SCHEMA[
            "properties"
        ]

        self.assertEqual(
            properties["total"]["type"],
            "string",
        )
        self.assertEqual(
            properties["unit"]["type"],
            "string",
        )
        self.assertEqual(
            properties["parts"]["type"],
            "array",
        )

    def test_exact_payload_becomes_fixed_total_allocation(
        self,
    ) -> None:
        allocation = parse_allocation_payload(
            {
                "total": "180",
                "unit": "minutes",
                "parts": [
                    {
                        "label": "Second examen",
                        "amount": "95",
                    },
                    {
                        "label": "Pause",
                        "amount": "15",
                    },
                    {
                        "label": "Premier examen",
                        "amount": "70",
                    },
                ],
            }
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

    def test_mismatched_payload_is_parsed_but_not_exact(
        self,
    ) -> None:
        allocation = parse_allocation_payload(
            {
                "total": "180",
                "unit": "minutes",
                "parts": [
                    {
                        "label": "Study",
                        "amount": "100",
                    },
                    {
                        "label": "Pause",
                        "amount": "20",
                    },
                ],
            }
        )

        self.assertFalse(
            allocation.is_exact
        )
        self.assertEqual(
            allocation.difference,
            Decimal("60"),
        )

    def test_payload_rejects_unexpected_top_level_fields(
        self,
    ) -> None:
        with self.assertRaises(
            AllocationFormatError
        ):
            parse_allocation_payload(
                {
                    "total": "180",
                    "unit": "minutes",
                    "parts": [
                        {
                            "label": "Study",
                            "amount": "180",
                        },
                    ],
                    "reasoning": "hidden text",
                }
            )

    def test_payload_rejects_invalid_decimal_amount(
        self,
    ) -> None:
        with self.assertRaises(
            AllocationFormatError
        ):
            parse_allocation_payload(
                {
                    "total": "180",
                    "unit": "minutes",
                    "parts": [
                        {
                            "label": "Study",
                            "amount": "not-a-number",
                        },
                    ],
                }
            )

    def test_payload_rejects_unexpected_part_fields(
        self,
    ) -> None:
        with self.assertRaises(
            AllocationFormatError
        ):
            parse_allocation_payload(
                {
                    "total": "180",
                    "unit": "minutes",
                    "parts": [
                        {
                            "label": "Study",
                            "amount": "180",
                            "comment": "extra",
                        },
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
