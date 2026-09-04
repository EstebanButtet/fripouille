"""Tests for inspectable associations between memories and people."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assistant_ia.memory.errors import MemoryPersonLinkNotFoundError
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.models import MemoryCandidate, MemoryPersonLink
from assistant_ia.memory.repository import DatabaseError, SQLiteDatabase
from assistant_ia.people.defaults import DEFAULT_PERSON_ID
from assistant_ia.people.person_repository import PersonRepository


class IncrementingClock:
    def __init__(self) -> None:
        self._next = datetime(2026, 9, 3, 10, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        value = self._next
        self._next += timedelta(minutes=1)
        return value


class MemoryPersonLinkModelTests(unittest.TestCase):
    def test_link_has_two_stable_ids_and_bounded_subject_role(self) -> None:
        link = MemoryPersonLink(memory_id=4, person_id=7)

        self.assertEqual(link.memory_id, 4)
        self.assertEqual(link.person_id, 7)
        self.assertEqual(link.role, "subject")

        with self.assertRaisesRegex(ValueError, "Unknown memory person role"):
            MemoryPersonLink(
                memory_id=4,
                person_id=7,
                role="participant",  # type: ignore[arg-type]
            )


class MemoryPeopleRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name) / "assistant.db"
        )
        self.database.initialize()
        self.person_repository = PersonRepository(self.database)
        self.este = self.person_repository.get_person(DEFAULT_PERSON_ID)
        assert self.este is not None
        self.alice = self.person_repository.create_person("Alice")
        self.repository = MemoryRepository(
            self.database,
            clock=IncrementingClock(),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_link_and_unlink_are_inspectable(self) -> None:
        memory = self.repository.save_memory("Projet Fripouille.")

        link = self.repository.link_person(memory.id, self.este.id)

        self.assertEqual(
            self.repository.list_person_links(memory.id),
            (link,),
        )
        removed = self.repository.unlink_person(memory.id, self.este.id)
        self.assertEqual(removed, link)
        self.assertEqual(self.repository.list_person_links(memory.id), ())
        self.assertEqual(self.repository.list_unassigned_memories(), (memory,))

        with self.assertRaises(MemoryPersonLinkNotFoundError):
            self.repository.unlink_person(memory.id, self.este.id)

    def test_one_memory_can_have_two_person_subjects(self) -> None:
        memory = self.repository.save_memory("Voyage partagé à Genève.")
        este_link = self.repository.link_person(memory.id, self.este.id)
        alice_link = self.repository.link_person(memory.id, self.alice.id)

        self.assertEqual(
            self.repository.list_person_links(memory.id),
            (este_link, alice_link),
        )
        self.assertEqual(
            self.repository.list_people_for_memory(memory.id),
            (self.este, self.alice),
        )
        self.assertEqual(
            self.repository.list_memories_for_person(self.este.id),
            (memory,),
        )
        self.assertEqual(
            self.repository.list_memories_for_person(self.alice.id),
            (memory,),
        )

    def test_linking_twice_does_not_duplicate_association(self) -> None:
        memory = self.repository.save_memory("Même association.")

        first = self.repository.link_person(memory.id, self.este.id)
        second = self.repository.link_person(memory.id, self.este.id)

        self.assertEqual(second, first)
        self.assertEqual(
            self.repository.list_person_links(memory.id),
            (first,),
        )

    def test_foreign_keys_reject_unknown_memory_or_person(self) -> None:
        memory = self.repository.save_memory("Association contrôlée.")

        with self.assertRaises(DatabaseError):
            self.repository.link_person(999, self.este.id)
        with self.assertRaises(DatabaseError):
            self.repository.link_person(memory.id, 999)

    def test_memory_deletion_cascades_only_its_links(self) -> None:
        first = self.repository.save_memory("Premier souvenir.")
        second = self.repository.save_memory("Deuxième souvenir.")
        self.repository.link_person(first.id, self.este.id)
        second_link = self.repository.link_person(second.id, self.este.id)

        self.repository.delete_memory(first.id)

        self.assertEqual(self.repository.list_person_links(first.id), ())
        self.assertEqual(
            self.repository.list_person_links(second.id),
            (second_link,),
        )

    def test_person_deletion_cascades_links_without_deleting_memory(self) -> None:
        memory = self.repository.save_memory("Souvenir conservé.")
        self.repository.link_person(memory.id, self.alice.id)

        with self.database.connect() as connection:
            connection.execute(
                "DELETE FROM persons WHERE id = ?",
                (self.alice.id,),
            )

        self.assertEqual(self.repository.list_person_links(memory.id), ())
        self.assertIn(memory, self.repository.list_unassigned_memories())

    def test_person_and_unassigned_listings_are_strictly_separated(self) -> None:
        este_memory = self.repository.save_memory("Souvenir Este.")
        alice_memory = self.repository.save_memory("Souvenir Alice.")
        general_memory = self.repository.save_memory("Souvenir général.")
        self.repository.link_person(este_memory.id, self.este.id)
        self.repository.link_person(alice_memory.id, self.alice.id)

        self.assertEqual(
            self.repository.list_memories_for_person(self.este.id),
            (este_memory,),
        )
        self.assertEqual(
            self.repository.list_memories_for_person(self.alice.id),
            (alice_memory,),
        )
        self.assertEqual(
            self.repository.list_unassigned_memories(),
            (general_memory,),
        )

    def test_candidate_and_subject_link_are_saved_atomically(self) -> None:
        candidate = MemoryCandidate(
            content="Je travaille sur Fripouille.",
            source_text="Je travaille sur Fripouille.",
            confidence=0.9,
        )

        memory = self.repository.save_candidate(
            candidate,
            subject_person_id=self.este.id,
        )

        self.assertEqual(
            self.repository.list_person_links(memory.id),
            (MemoryPersonLink(memory.id, self.este.id),),
        )

        invalid = MemoryCandidate(
            content="Je travaille sur Atlas.",
            source_text="Je travaille sur Atlas.",
            confidence=0.9,
        )
        with self.assertRaises(DatabaseError):
            self.repository.save_candidate(invalid, subject_person_id=999)
        self.assertEqual(len(self.repository.list_memories()), 1)

    def test_memory_correction_preserves_existing_person_links(self) -> None:
        memory = self.repository.save_memory("Projet initial.")
        link = self.repository.link_person(memory.id, self.este.id)
        correction = MemoryCandidate(
            content="Projet corrigé.",
            source_text="En fait, projet corrigé.",
            confidence=0.92,
        )

        updated = self.repository.update_memory(memory.id, correction)

        self.assertEqual(updated.id, memory.id)
        self.assertEqual(
            self.repository.list_person_links(memory.id),
            (link,),
        )

    def test_schema_exposes_two_foreign_keys_unique_key_and_person_index(
        self,
    ) -> None:
        with self.database.connect() as connection:
            foreign_keys = connection.execute(
                "PRAGMA foreign_key_list(memory_people)"
            ).fetchall()
            indexes = connection.execute(
                "PRAGMA index_list(memory_people)"
            ).fetchall()

        targets = {(row[2], row[3], row[4], row[6]) for row in foreign_keys}
        self.assertEqual(
            targets,
            {
                ("memories", "memory_id", "id", "CASCADE"),
                ("persons", "person_id", "id", "CASCADE"),
            },
        )
        index_names = {row[1] for row in indexes}
        self.assertIn("idx_memory_people_person_id", index_names)
        primary_index = next(row[1] for row in indexes if row[3] == "pk")
        with self.database.connect() as connection:
            primary_columns = connection.execute(
                "SELECT name FROM pragma_index_info(?) ORDER BY seqno",
                (primary_index,),
            ).fetchall()
        self.assertEqual(
            primary_columns,
            [("memory_id",), ("person_id",), ("role",)],
        )

        memory = self.repository.save_memory("Rôle SQL borné.")
        with self.assertRaises(DatabaseError):
            with self.database.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO memory_people (memory_id, person_id, role)
                    VALUES (?, ?, ?)
                    """,
                    (memory.id, self.este.id, "participant"),
                )


if __name__ == "__main__":
    unittest.main()
