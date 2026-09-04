"""Tests for deterministic contextual memory retrieval."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.memory.retrieval import ContextualMemoryRetriever
from assistant_ia.people.defaults import DEFAULT_PERSON_ID
from assistant_ia.people.person_repository import PersonRepository


class SequenceClock:
    """Return deterministic timestamps in a predefined order."""

    def __init__(self, *values: datetime) -> None:
        """Store timestamps returned by the clock."""
        self._values = list(values)

    def __call__(self) -> datetime:
        """Return the next deterministic timestamp."""
        return self._values.pop(0)


class ContextualMemoryRetrieverTests(unittest.TestCase):
    """Validate deterministic lexical selection of memories."""

    def setUp(self) -> None:
        """Create one isolated initialized memory repository."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name) / "assistant.db"
        )
        self.database.initialize()
        self.base_time = datetime(
            2026,
            8,
            20,
            10,
            0,
            tzinfo=timezone.utc,
        )

    def tearDown(self) -> None:
        """Remove the isolated database."""
        self.temporary_directory.cleanup()

    def _build(
        self,
        *contents: str,
    ) -> tuple[MemoryRepository, ContextualMemoryRetriever]:
        """Save ordered contents and return repository and retriever."""
        timestamps = tuple(
            self.base_time + timedelta(minutes=index)
            for index in range(len(contents))
        )
        repository = MemoryRepository(
            self.database,
            clock=SequenceClock(*timestamps),
        )

        for content in contents:
            repository.save_memory(content)

        return repository, ContextualMemoryRetriever(repository)

    def test_returns_nothing_for_empty_or_ignored_query(self) -> None:
        """Empty text and stop words should not retrieve memories."""
        _, retriever = self._build("Examen de biologie lundi.")

        for query in ("", "   ", "et avec pour", "!!!"):
            with self.subTest(query=query):
                self.assertEqual(retriever.retrieve(query), ())

    def test_ignores_french_and_english_question_terms(self) -> None:
        """Question grammar alone should not retrieve a memory."""
        _, retriever = self._build(
            "Quand quel quoi comment pourquoi combien où.",
            "What when where which who whom whose how why.",
        )

        self.assertEqual(
            retriever.retrieve(
                "qu quand quel quelle quels quelles quoi "
                "comment pourquoi combien où"
            ),
            (),
        )
        self.assertEqual(
            retriever.retrieve(
                "what when where which who whom whose how why"
            ),
            (),
        )

    def test_question_term_neither_matches_nor_reduces_score(self) -> None:
        """Interrogatives should not match or enter score coverage."""
        _, retriever = self._build(
            "Je préfère les réponses courtes quand la question est simple.",
            "Mon examen de biologie est prévu lundi.",
        )

        results = retriever.retrieve(
            "Quand est mon examen de biologie ?"
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].memory.content,
            "Mon examen de biologie est prévu lundi.",
        )
        self.assertEqual(results[0].score, 1.0)
        self.assertEqual(
            results[0].matched_terms,
            ("biologie", "examen"),
        )

    def test_retrieves_one_simple_match(self) -> None:
        """One shared useful term should select its memory."""
        repository, retriever = self._build(
            "Acheter du pain.",
            "Examen de biologie lundi.",
        )
        expected = repository.find_memories("biologie")[0]

        results = retriever.retrieve("biologie")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].memory, expected)
        self.assertEqual(results[0].score, 1.0)
        self.assertEqual(results[0].matched_terms, ("biologie",))

    def test_ranks_more_matching_terms_first(self) -> None:
        """Query coverage should dominate weaker lexical matches."""
        _, retriever = self._build(
            "Examen prévu demain.",
            "Examen de biologie prévu lundi.",
        )

        results = retriever.retrieve("examen biologie")

        self.assertEqual(
            [result.memory.content for result in results],
            [
                "Examen de biologie prévu lundi.",
                "Examen prévu demain.",
            ],
        )
        self.assertEqual(
            [result.score for result in results],
            [1.0, 0.5],
        )

    def test_old_relevant_memory_beats_recent_partial_match(self) -> None:
        """Recency should not override stronger lexical relevance."""
        _, retriever = self._build(
            "Biologie et examen final en septembre.",
            "Examen demain matin.",
        )

        results = retriever.retrieve("biologie examen")

        self.assertEqual(
            results[0].memory.content,
            "Biologie et examen final en septembre.",
        )

    def test_confidence_only_breaks_equal_lexical_scores(self) -> None:
        """Confidence should be secondary to lexical query coverage."""
        repository, retriever = self._build(
            "Projet ancien précis.",
            "Projet récent précis.",
        )
        older, newer = reversed(repository.list_memories())

        with self.database.connect() as connection:
            connection.execute(
                "UPDATE memories SET confidence = ? WHERE id = ?",
                (0.8, older.id),
            )
            connection.execute(
                "UPDATE memories SET confidence = ? WHERE id = ?",
                (0.2, newer.id),
            )

        results = retriever.retrieve("projet précis")

        self.assertEqual(results[0].memory.id, older.id)

    def test_normalizes_case_accents_unicode_and_punctuation(self) -> None:
        """Unicode diacritics, case and punctuation should normalize."""
        _, retriever = self._build(
            "Café préféré à Zürich; excellent!",
        )

        results = retriever.retrieve("CAFE, zurich?")

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].matched_terms,
            ("cafe", "zurich"),
        )

    def test_query_token_duplicates_do_not_inflate_score(self) -> None:
        """Repeated query terms should count only once."""
        _, retriever = self._build(
            "Examen de biologie.",
            "Examen de mathématiques.",
        )

        results = retriever.retrieve(
            "examen examen examen biologie"
        )

        self.assertEqual(
            [result.score for result in results],
            [1.0, 0.5],
        )

    def test_ignores_tokens_shorter_than_two_characters(self) -> None:
        """Single-character tokens should not create noisy matches."""
        _, retriever = self._build(
            "Le code x est réservé.",
            "Le code y est public.",
        )

        results = retriever.retrieve("x")

        self.assertEqual(results, ())

    def test_respects_result_limit(self) -> None:
        """Only the requested number of ranked matches should return."""
        _, retriever = self._build(
            "Biologie végétale.",
            "Biologie marine.",
            "Biologie cellulaire.",
        )

        results = retriever.retrieve("biologie", limit=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(
            [result.memory.content for result in results],
            ["Biologie cellulaire.", "Biologie marine."],
        )

    def test_uses_recency_then_identifier_for_stable_ties(self) -> None:
        """Equal lexical and confidence scores should have stable ties."""
        same_time = (self.base_time,) * 3
        repository = MemoryRepository(
            self.database,
            clock=SequenceClock(*same_time),
        )
        first = repository.save_memory("Projet alpha.")
        second = repository.save_memory("Projet bêta.")
        third = repository.save_memory("Projet gamma.")
        retriever = ContextualMemoryRetriever(repository)

        results = retriever.retrieve("projet")

        self.assertEqual(
            [result.memory for result in results],
            [third, second, first],
        )

    def test_returns_nothing_without_shared_terms(self) -> None:
        """Unrelated memories should not be returned."""
        _, retriever = self._build("Acheter du pain.")

        self.assertEqual(retriever.retrieve("biologie"), ())

    def test_retrieval_does_not_modify_persisted_memories(self) -> None:
        """Retrieval must leave every persisted field unchanged."""
        repository, retriever = self._build(
            "Examen de biologie lundi.",
        )
        before = repository.list_memories()

        retriever.retrieve("examen biologie")

        self.assertEqual(repository.list_memories(), before)

    def test_find_memories_keeps_literal_search_semantics(self) -> None:
        """Contextual retrieval must not alter explicit LIKE search."""
        repository, retriever = self._build(
            "Examen de biologie.",
            "Biologie puis examen.",
        )

        retriever.retrieve("examen biologie")

        self.assertEqual(
            tuple(
                memory.content
                for memory in repository.find_memories(
                    "examen de biologie"
                )
            ),
            ("Examen de biologie.",),
        )

    def test_active_person_retrieval_is_symmetric_and_allows_shared_memories(
        self,
    ) -> None:
        person_repository = PersonRepository(self.database)
        este = person_repository.get_person(DEFAULT_PERSON_ID)
        assert este is not None
        alice = person_repository.create_person("Alice")
        repository = MemoryRepository(
            self.database,
            clock=SequenceClock(
                self.base_time,
                self.base_time + timedelta(minutes=1),
                self.base_time + timedelta(minutes=2),
                self.base_time + timedelta(minutes=3),
            ),
        )
        este_memory = repository.save_memory("Projet privé d'Este.")
        alice_memory = repository.save_memory("Projet privé d'Alice.")
        general_memory = repository.save_memory("Projet général Fripouille.")
        shared_memory = repository.save_memory("Projet partagé à Genève.")
        repository.link_person(este_memory.id, este.id)
        repository.link_person(alice_memory.id, alice.id)
        repository.link_person(shared_memory.id, este.id)
        repository.link_person(shared_memory.id, alice.id)
        retriever = ContextualMemoryRetriever(repository)

        este_results = retriever.retrieve("projet", person_id=este.id)
        alice_results = retriever.retrieve("projet", person_id=alice.id)

        self.assertEqual(
            tuple(result.memory for result in este_results),
            (shared_memory, este_memory, general_memory),
        )
        self.assertNotIn(
            alice_memory,
            tuple(result.memory for result in este_results),
        )
        self.assertEqual(
            tuple(result.memory for result in alice_results),
            (shared_memory, alice_memory, general_memory),
        )
        self.assertNotIn(
            este_memory,
            tuple(result.memory for result in alice_results),
        )

    def test_unresolved_retrieval_sees_only_unassigned_memories(self) -> None:
        person_repository = PersonRepository(self.database)
        este = person_repository.get_person(DEFAULT_PERSON_ID)
        assert este is not None
        repository = MemoryRepository(
            self.database,
            clock=SequenceClock(
                self.base_time,
                self.base_time + timedelta(minutes=1),
            ),
        )
        private_memory = repository.save_memory("Biologie privée.")
        general_memory = repository.save_memory("Biologie générale.")
        repository.link_person(private_memory.id, este.id)
        retriever = ContextualMemoryRetriever(repository)

        results = retriever.retrieve("biologie")

        self.assertEqual(
            tuple(result.memory for result in results),
            (general_memory,),
        )

    def test_person_memories_have_priority_over_general_matches(self) -> None:
        person_repository = PersonRepository(self.database)
        este = person_repository.get_person(DEFAULT_PERSON_ID)
        assert este is not None
        repository = MemoryRepository(
            self.database,
            clock=SequenceClock(
                self.base_time,
                self.base_time + timedelta(minutes=1),
            ),
        )
        personal = repository.save_memory("Projet personnel.")
        general = repository.save_memory("Projet robotique Fripouille.")
        repository.link_person(personal.id, este.id)
        retriever = ContextualMemoryRetriever(repository)

        results = retriever.retrieve(
            "projet robotique Fripouille",
            person_id=este.id,
        )

        self.assertEqual(results[0].memory, personal)
        self.assertEqual(results[1].memory, general)


if __name__ == "__main__":
    unittest.main()
