"""FRP-IA-12 : activation contrôlée et portée d'apprentissage conservée."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from assistant_ia.roles import FunctionalRole, RoleService
from assistant_ia.core.assistant import AssistantCore
from assistant_ia.people.models import PersonProfile
from assistant_ia.application import build_default_assistant
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.learning.service import BehavioralLearningScopeError
from assistant_ia.learning.models import ExperienceProvenance
from assistant_ia.learning.outcomes import ExperienceOutcome


class RoleTests(unittest.TestCase):
    def setUp(self):
        self.caps = frozenset({"conversation"})
        self.roles = RoleService(lambda: self.caps)

    def test_default_and_activation(self):
        self.assertIsNone(self.roles.active)
        self.assertTrue(self.roles.activate("guide").accepted)
        self.assertEqual(self.roles.active_id, "guide")

    def test_unavailable_cameraman_preserves_role(self):
        self.roles.activate("guide")
        decision = self.roles.activate("cameraman")
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.missing_capabilities, ("follow_target", "record_video"))
        self.assertEqual(self.roles.active_id, "guide")

    def test_unknown_role(self):
        self.assertFalse(self.roles.activate("admin").accepted)

    def test_replace_one_role(self):
        self.caps |= {"social_vision"}
        self.roles.activate("guide")
        self.roles.activate("observateur")
        self.assertEqual(self.roles.active_id, "observateur")

    def test_lost_capability_deactivates(self):
        self.roles.activate("guide")
        self.caps = frozenset()
        self.assertIsNone(self.roles.active)
        self.assertEqual(self.roles.last_decision.reason, "capability_lost")

    def test_explicit_request_and_reset(self):
        self.roles.handle_request("Passe en mode guide.")
        self.assertEqual(self.roles.active_id, "guide")
        self.roles.handle_request("/role off")
        self.assertIsNone(self.roles.active)

    def test_quoted_or_ambiguous_request_not_executed(self):
        self.assertIsNone(self.roles.handle_request('Il a dit « Passe en mode guide. »'))
        self.assertIsNone(self.roles.active)

    def test_catalogue_immutable(self):
        with self.assertRaises(TypeError):
            self.roles.catalogue["admin"] = self.roles.catalogue["guide"]

    def test_bounds(self):
        with self.assertRaises(ValueError):
            FunctionalRole("guide", "Guide", "x" * 201, frozenset(), (), ())

    def test_core_resolves_request_without_model(self):
        model = Mock()
        core = AssistantCore(model_client=model)
        self.assertIn("activé", core.process_message("Passe en mode guide."))
        self.assertEqual(core.roles.active_id, "guide")
        model.generate_response.assert_not_called()
        self.assertEqual(len(core.context.messages), 2)

    def test_core_reset(self):
        core = AssistantCore(model_client=Mock())
        core.process_message("/role guide")
        core.reset_conversation()
        self.assertIsNone(core.roles.active)

    def test_person_change_resets_role(self):
        core = AssistantCore(model_client=Mock())
        core.process_message("/role guide")
        core.person_context.set_active_person(PersonProfile("Alice"))
        core._sync_state_person()
        self.assertIsNone(core.roles.active)


class RoleLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.core = build_default_assistant(database=SQLiteDatabase(Path(self.tmp.name) / "test.db"), model_client=Mock())
        self.service = self.core.behavioral_learning_service
        self.provenance = ExperienceProvenance(source_type="manual_entry")

    def test_role_attempt_retained_but_never_persisted_as_default(self):
        self.core.roles.activate("guide")
        attempt = self.service.begin_active_person_attempt(context="cours", objective="expliquer", strategy="étapes")
        self.assertEqual(attempt.role_id, "guide")
        self.core.roles.reset()
        with self.assertRaises(BehavioralLearningScopeError):
            self.service.record_active_person_outcome(attempt, ExperienceOutcome("success", "reported_result", "compris"), provenance=self.provenance)
        self.assertEqual(self.service.list_active_person_experiences(), ())

    def test_role_cannot_write_raw_global_experience(self):
        self.core.roles.activate("guide")
        with self.assertRaises(BehavioralLearningScopeError):
            self.service.record_global_experience(context="cours", objective="expliquer", strategy="étapes", result="compris", provenance=self.provenance)

    def test_default_learning_preserved(self):
        attempt = self.service.begin_active_person_attempt(context="cours", objective="expliquer", strategy="étapes")
        result = self.service.record_active_person_outcome(attempt, ExperienceOutcome("success", "reported_result", "compris"), provenance=self.provenance)
        self.assertEqual(result.outcome_status, "success")
