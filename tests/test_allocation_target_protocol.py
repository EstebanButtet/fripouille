"""Tests for application-authoritative allocation targets."""

from decimal import Decimal
import unittest

from assistant_ia.intelligence.allocation import (
    ALLOCATION_PROPOSAL_SCHEMA,
    AllocationFormatError,
    AllocationTarget,
    parse_allocation_proposal_json,
)


class AllocationTargetProtocolTests(unittest.TestCase):
    """Validate targets owned by the application, not the LLM."""

    def test_target_requires_positive_decimal_total(
        self,
    ) -> None:
        target = AllocationTarget(
            total=Decimal("180"),
            unit="minutes",
        )

        self.assertEqual(
            target.total,
            Decimal("180"),
        )
        self.assertEqual(
            target.unit,
            "minutes",
        )

    def test_target_rejects_non_decimal_total(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            AllocationTarget(
                total=180,
                unit="minutes",
            )

    def test_proposal_schema_contains_only_parts(
        self,
    ) -> None:
        self.assertEqual(
            ALLOCATION_PROPOSAL_SCHEMA["required"],
            [
                "parts",
            ],
        )
        self.assertFalse(
            ALLOCATION_PROPOSAL_SCHEMA["additionalProperties"]
        )

        properties = ALLOCATION_PROPOSAL_SCHEMA[
            "properties"
        ]

        self.assertEqual(
            set(properties),
            {
                "parts",
            },
        )

    def test_amount_is_real_json_number(
        self,
    ) -> None:
        amount_schema = (
            ALLOCATION_PROPOSAL_SCHEMA[
                "properties"
            ]["parts"]["items"]["properties"]["amount"]
        )

        self.assertEqual(
            amount_schema["type"],
            "number",
        )

    def test_json_proposal_uses_authoritative_target(
        self,
    ) -> None:
        target = AllocationTarget(
            total=Decimal("180"),
            unit="minutes",
        )

        allocation = parse_allocation_proposal_json(
            """{
                "parts": [
                    {
                        "label": "Second examen",
                        "amount": 95
                    },
                    {
                        "label": "Pause",
                        "amount": 15
                    },
                    {
                        "label": "Premier examen",
                        "amount": 70
                    }
                ]
            }""",
            target=target,
        )

        self.assertEqual(
            allocation.total,
            Decimal("180"),
        )
        self.assertEqual(
            allocation.unit,
            "minutes",
        )
        self.assertEqual(
            allocation.allocated_total,
            Decimal("180"),
        )

        allocation.require_exact()

    def test_json_numbers_are_parsed_as_decimal(
        self,
    ) -> None:
        target = AllocationTarget(
            total=Decimal("0.3"),
            unit="hours",
        )

        allocation = parse_allocation_proposal_json(
            """{
                "parts": [
                    {
                        "label": "A",
                        "amount": 0.1
                    },
                    {
                        "label": "B",
                        "amount": 0.2
                    }
                ]
            }""",
            target=target,
        )

        self.assertEqual(
            allocation.allocated_total,
            Decimal("0.3"),
        )
        self.assertTrue(
            allocation.is_exact
        )

    def test_proposal_cannot_supply_its_own_total(
        self,
    ) -> None:
        target = AllocationTarget(
            total=Decimal("180"),
            unit="minutes",
        )

        with self.assertRaises(
            AllocationFormatError
        ):
            parse_allocation_proposal_json(
                """{
                    "total": 155,
                    "parts": [
                        {
                            "label": "Study",
                            "amount": 155
                        }
                    ]
                }""",
                target=target,
            )

    def test_proposal_cannot_supply_its_own_unit(
        self,
    ) -> None:
        target = AllocationTarget(
            total=Decimal("180"),
            unit="minutes",
        )

        with self.assertRaises(
            AllocationFormatError
        ):
            parse_allocation_proposal_json(
                """{
                    "unit": "hours",
                    "parts": [
                        {
                            "label": "Study",
                            "amount": 180
                        }
                    ]
                }""",
                target=target,
            )


if __name__ == "__main__":
    unittest.main()
