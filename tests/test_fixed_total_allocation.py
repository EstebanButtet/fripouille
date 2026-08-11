"""Tests for deterministic fixed-total allocation validation."""

from decimal import Decimal
import unittest

from assistant_ia.intelligence.allocation import (
    AllocationMismatchError,
    AllocationPart,
    FixedTotalAllocation,
)


class FixedTotalAllocationTests(unittest.TestCase):
    """Validate exact fixed-total arithmetic independently of the LLM."""

    def test_exact_allocation_is_accepted(
        self,
    ) -> None:
        allocation = FixedTotalAllocation(
            total=Decimal("180"),
            unit="minutes",
            parts=(
                AllocationPart(
                    label="Second examen",
                    amount=Decimal("95"),
                ),
                AllocationPart(
                    label="Pause",
                    amount=Decimal("15"),
                ),
                AllocationPart(
                    label="Premier examen",
                    amount=Decimal("70"),
                ),
            ),
        )

        self.assertEqual(
            allocation.allocated_total,
            Decimal("180"),
        )
        self.assertEqual(
            allocation.difference,
            Decimal("0"),
        )
        self.assertTrue(
            allocation.is_exact
        )

        allocation.require_exact()

    def test_incomplete_allocation_is_detected(
        self,
    ) -> None:
        allocation = FixedTotalAllocation(
            total=Decimal("180"),
            unit="minutes",
            parts=(
                AllocationPart(
                    label="Second examen",
                    amount=Decimal("90"),
                ),
                AllocationPart(
                    label="Pause",
                    amount=Decimal("15"),
                ),
                AllocationPart(
                    label="Premier examen",
                    amount=Decimal("50"),
                ),
            ),
        )

        self.assertEqual(
            allocation.allocated_total,
            Decimal("155"),
        )
        self.assertEqual(
            allocation.difference,
            Decimal("25"),
        )
        self.assertFalse(
            allocation.is_exact
        )

        with self.assertRaises(
            AllocationMismatchError
        ):
            allocation.require_exact()

    def test_decimal_quantities_remain_exact(
        self,
    ) -> None:
        allocation = FixedTotalAllocation(
            total=Decimal("3"),
            unit="hours",
            parts=(
                AllocationPart(
                    label="A",
                    amount=Decimal("1.5"),
                ),
                AllocationPart(
                    label="B",
                    amount=Decimal("1.5"),
                ),
            ),
        )

        self.assertTrue(
            allocation.is_exact
        )
        self.assertEqual(
            allocation.allocated_total,
            Decimal("3.0"),
        )

    def test_part_amount_must_be_decimal(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            AllocationPart(
                label="Invalid",
                amount=1.5,
            )

    def test_part_amount_must_be_positive(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            AllocationPart(
                label="Invalid",
                amount=Decimal("-1"),
            )

    def test_unit_cannot_be_empty(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            FixedTotalAllocation(
                total=Decimal("180"),
                unit="   ",
                parts=(
                    AllocationPart(
                        label="Study",
                        amount=Decimal("180"),
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
