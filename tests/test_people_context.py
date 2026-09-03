"""Tests for structured person and active-user context."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from assistant_ia.people.context import ActivePersonContext
from assistant_ia.people.defaults import build_default_person
from assistant_ia.people.models import Person, PersonProfile


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
    def test_tracks_persistent_default_person_id(self) -> None:
        context = ActivePersonContext(
            assistant_name="Fripouille",
            default_person=Person(
                id=1,
                display_name="Este",
                created_at=datetime(
                    2026,
                    8,
                    7,
                    tzinfo=timezone.utc,
                ),
            ),
        )

        self.assertEqual(context.default_person_id, 1)
        self.assertEqual(context.active_person_id, 1)

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
            default_person=Person(
                id=1,
                display_name="Este",
                created_at=datetime(
                    2026,
                    8,
                    7,
                    tzinfo=timezone.utc,
                ),
            ),
        )

        context.set_active_persistent_person(
            Person(
                id=2,
                display_name="Lucas",
                created_at=datetime(
                    2026,
                    8,
                    7,
                    tzinfo=timezone.utc,
                ),
            )
        )
        context.reset()

        self.assertEqual(
            context.active_person.name,
            "Este",
        )
        self.assertEqual(context.active_person_id, 1)

    def test_unresolved_profile_has_no_persistent_id(self) -> None:
        context = ActivePersonContext(
            assistant_name="Fripouille",
            default_person=Person(
                id=1,
                display_name="Este",
                created_at=datetime(
                    2026,
                    8,
                    7,
                    tzinfo=timezone.utc,
                ),
            ),
        )

        context.set_active_person(PersonProfile(name="Visiteur"))

        self.assertEqual(context.active_person.name, "Visiteur")
        self.assertIsNone(context.active_person_id)

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
