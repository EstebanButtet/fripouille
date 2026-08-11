"""Tests for structured person and active-user context."""

from __future__ import annotations

import unittest

from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.defaults import build_default_person
from assistant_ia.people.models import PersonProfile


class PersonProfileTests(unittest.TestCase):
    def test_normalizes_name(self) -> None:
        person = PersonProfile(
            name="  Este  ",
        )

        self.assertEqual(
            person.name,
            "Este",
        )

    def test_rejects_empty_name(self) -> None:
        with self.assertRaises(ValueError):
            PersonProfile(
                name="   ",
            )

    def test_default_person_is_este(self) -> None:
        person = build_default_person()

        self.assertEqual(
            person.name,
            "Este",
        )


class ActivePersonContextTests(unittest.TestCase):
    def test_starts_with_default_person(self) -> None:
        context = ActivePersonContext(
            assistant_name="Fripouille",
            default_person=PersonProfile(
                name="Este",
            ),
        )

        self.assertEqual(
            context.active_person.name,
            "Este",
        )

    def test_can_switch_active_person(self) -> None:
        context = ActivePersonContext(
            assistant_name="Fripouille",
            default_person=PersonProfile(
                name="Este",
            ),
        )

        context.set_active_person(
            PersonProfile(
                name="Lucas",
            )
        )

        self.assertEqual(
            context.active_person.name,
            "Lucas",
        )

    def test_reset_restores_default_person(self) -> None:
        context = ActivePersonContext(
            assistant_name="Fripouille",
            default_person=PersonProfile(
                name="Este",
            ),
        )

        context.set_active_person(
            PersonProfile(
                name="Lucas",
            )
        )
        context.reset()

        self.assertEqual(
            context.active_person.name,
            "Este",
        )

    def test_assistant_name_is_exclusive(self) -> None:
        context = ActivePersonContext(
            assistant_name="Fripouille",
            default_person=PersonProfile(
                name="Este",
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "reserved exclusively",
        ):
            context.set_active_person(
                PersonProfile(
                    name="fripouille",
                )
            )


if __name__ == "__main__":
    unittest.main()
