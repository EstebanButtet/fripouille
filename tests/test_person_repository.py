"""Tests for the persistent person registry introduced by FRP-IA-04A."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.people.defaults import (
    DEFAULT_PERSON_ID,
    DEFAULT_PERSON_NAME,
)
from assistant_ia.people.models import Person
from assistant_ia.people.person_repository import PersonRepository


class SequenceClock:
    """Return deterministic timestamps in a predefined order."""

    def __init__(self, *values: datetime) -> None:
        self._values = list(values)

    def __call__(self) -> datetime:
        if not self._values:
            raise AssertionError(
                "No deterministic clock value remains."
            )

        return self._values.pop(0)


class PersonModelTests(unittest.TestCase):
    """Validate the minimal persistent person model."""

    def test_normalizes_display_name_and_creation_time(self) -> None:
        source_time = datetime(
            2026,
            8,
            7,
            3,
            0,
            tzinfo=timezone(timedelta(hours=2)),
        )

        person = Person(
            id=7,
            display_name="  Alex  ",
            created_at=source_time,
        )

        self.assertEqual(person.id, 7)
        self.assertEqual(person.display_name, "Alex")
        self.assertEqual(
            person.created_at,
            datetime(2026, 8, 7, 1, 0, tzinfo=timezone.utc),
        )


class PersonRepositoryTests(unittest.TestCase):
    """Validate creation and deterministic retrieval of people."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name) / "assistant.db"
        )
        self.database.initialize()
        self.first_time = datetime(
            2026,
            8,
            7,
            1,
            0,
            tzinfo=timezone.utc,
        )
        self.second_time = datetime(
            2026,
            8,
            7,
            2,
            0,
            tzinfo=timezone.utc,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _repository(
        self,
        *clock_values: datetime,
    ) -> PersonRepository:
        return PersonRepository(
            self.database,
            clock=SequenceClock(*clock_values),
        )

    def test_creates_person_with_stable_id_distinct_from_name(self) -> None:
        repository = self._repository(self.first_time)

        created = repository.create_person("  Lucas  ")
        reloaded = repository.get_person(created.id)

        self.assertIsInstance(created.id, int)
        self.assertIsInstance(created.display_name, str)
        self.assertEqual(created.display_name, "Lucas")
        self.assertEqual(created.created_at, self.first_time)
        self.assertEqual(reloaded, created)

    def test_rejects_empty_display_name(self) -> None:
        repository = self._repository()

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            repository.create_person("   ")

    def test_allows_two_people_with_same_display_name(self) -> None:
        repository = self._repository(
            self.first_time,
            self.second_time,
        )

        first = repository.create_person("Alex")
        second = repository.create_person("Alex")

        self.assertNotEqual(first.id, second.id)
        self.assertEqual(first.display_name, second.display_name)

    def test_get_person_returns_none_for_unknown_id(self) -> None:
        repository = self._repository()

        self.assertIsNone(repository.get_person(999))

    def test_finds_name_with_nfc_casefold_and_outer_spaces(self) -> None:
        repository = self._repository(self.first_time)
        created = repository.create_person("Élodie")

        matches = repository.find_persons_by_display_name(
            "  e\u0301LODIE  "
        )

        self.assertEqual(matches, (created,))

    def test_name_lookup_returns_every_homonym(self) -> None:
        repository = self._repository(
            self.first_time,
            self.second_time,
        )
        first = repository.create_person("Alex")
        second = repository.create_person("alex")

        matches = repository.find_persons_by_display_name("ALEX")

        self.assertEqual(matches, (first, second))

    def test_lists_default_and_created_people_by_id(self) -> None:
        repository = self._repository(
            self.first_time,
            self.second_time,
        )
        first = repository.create_person("Lucas")
        second = repository.create_person("Sarah")

        persons = repository.list_persons()

        self.assertEqual(
            tuple(person.id for person in persons),
            (DEFAULT_PERSON_ID, first.id, second.id),
        )
        self.assertEqual(
            tuple(person.display_name for person in persons),
            (DEFAULT_PERSON_NAME, "Lucas", "Sarah"),
        )

    def test_listing_respects_requested_limit(self) -> None:
        repository = self._repository(self.first_time)
        repository.create_person("Lucas")

        persons = repository.list_persons(limit=1)

        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0].id, DEFAULT_PERSON_ID)


if __name__ == "__main__":
    unittest.main()
