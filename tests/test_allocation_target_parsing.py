"""Tests for deterministic duration target parsing."""

from decimal import Decimal
import unittest

from assistant_ia.intelligence.allocation import (
    AllocationTargetParseError,
    parse_duration_target_text,
)


class AllocationTargetParsingTests(unittest.TestCase):
    """Validate exact user-supplied duration parsing."""

    def test_parses_french_word_hours(
        self,
    ) -> None:
        target = parse_duration_target_text(
            "trois heures"
        )

        self.assertEqual(
            target.total,
            Decimal("180"),
        )
        self.assertEqual(
            target.unit,
            "minutes",
        )

    def test_parses_numeric_hours(
        self,
    ) -> None:
        target = parse_duration_target_text(
            "3 heures"
        )

        self.assertEqual(
            target.total,
            Decimal("180"),
        )

    def test_parses_minutes(
        self,
    ) -> None:
        target = parse_duration_target_text(
            "180 minutes"
        )

        self.assertEqual(
            target.total,
            Decimal("180"),
        )

    def test_parses_compact_hours_and_minutes(
        self,
    ) -> None:
        target = parse_duration_target_text(
            "1h30"
        )

        self.assertEqual(
            target.total,
            Decimal("90"),
        )

    def test_parses_spaced_hours_and_minutes(
        self,
    ) -> None:
        target = parse_duration_target_text(
            "1 h 30 min"
        )

        self.assertEqual(
            target.total,
            Decimal("90"),
        )

    def test_parses_decimal_hours(
        self,
    ) -> None:
        target = parse_duration_target_text(
            "1,5 heure"
        )

        self.assertEqual(
            target.total,
            Decimal("90.0"),
        )

    def test_normalizes_case_and_whitespace(
        self,
    ) -> None:
        target = parse_duration_target_text(
            "  TROIS   HEURES  "
        )

        self.assertEqual(
            target.total,
            Decimal("180"),
        )

    def test_rejects_approximate_duration(
        self,
    ) -> None:
        with self.assertRaises(
            AllocationTargetParseError
        ):
            parse_duration_target_text(
                "environ trois heures"
            )

    def test_rejects_duration_range(
        self,
    ) -> None:
        with self.assertRaises(
            AllocationTargetParseError
        ):
            parse_duration_target_text(
                "2-3 heures"
            )

    def test_rejects_extra_context(
        self,
    ) -> None:
        with self.assertRaises(
            AllocationTargetParseError
        ):
            parse_duration_target_text(
                "trois heures ce soir"
            )

    def test_rejects_empty_text(
        self,
    ) -> None:
        with self.assertRaises(
            AllocationTargetParseError
        ):
            parse_duration_target_text(
                "   "
            )


if __name__ == "__main__":
    unittest.main()
