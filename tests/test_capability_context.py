from __future__ import annotations

import unittest

from assistant_ia.actions.action import Action
from assistant_ia.actions.registry import ActionRegistry
from assistant_ia.capabilities.context import (
    CapabilityContext,
    build_capability_context,
    render_capability_context,
)


def fake_handler(parameters) -> str:
    return "OK"


class CapabilityContextTests(unittest.TestCase):
    def test_uses_registered_actions_as_source_of_truth(
        self,
    ) -> None:
        registry = ActionRegistry(
            (
                Action(
                    name="create_task",
                    handler=fake_handler,
                ),
                Action(
                    name="save_memory",
                    handler=fake_handler,
                ),
                Action(
                    name="launch_application",
                    handler=fake_handler,
                ),
            )
        )

        context = build_capability_context(
            registry
        )

        self.assertEqual(
            context.available_actions,
            (
                "create_task",
                "launch_application",
                "save_memory",
            ),
        )

    def test_capabilities_are_immutable(
        self,
    ) -> None:
        registry = ActionRegistry()

        context = build_capability_context(
            registry
        )

        with self.assertRaises(
            (AttributeError, TypeError),
        ):
            context.available_actions = ()

    def test_current_limits_are_explicit(
        self,
    ) -> None:
        registry = ActionRegistry()

        context = build_capability_context(
            registry
        )

        self.assertFalse(
            context.automatic_memory_retrieval
        )
        self.assertFalse(
            context.visual_input
        )
        self.assertFalse(
            context.audio_input
        )
        self.assertFalse(
            context.robot_control
        )

    def test_builder_enables_only_explicit_automatic_retrieval(
        self,
    ) -> None:
        """Assembly must opt in when a working retriever is present."""
        registry = ActionRegistry()

        default_context = build_capability_context(registry)
        retrieval_context = build_capability_context(
            registry,
            automatic_memory_retrieval=True,
        )

        self.assertFalse(
            default_context.automatic_memory_retrieval
        )
        self.assertTrue(
            retrieval_context.automatic_memory_retrieval
        )

    def test_render_lists_only_real_actions(
        self,
    ) -> None:
        registry = ActionRegistry(
            (
                Action(
                    name="save_memory",
                    handler=fake_handler,
                ),
                Action(
                    name="launch_application",
                    handler=fake_handler,
                ),
            )
        )

        rendered = render_capability_context(
            build_capability_context(
                registry
            )
        )

        self.assertIn(
            "- launch_application",
            rendered,
        )
        self.assertIn(
            "- save_memory",
            rendered,
        )
        self.assertNotIn(
            "- create_task",
            rendered,
        )

    def test_render_explains_registered_memory_search(
        self,
    ) -> None:
        registry = ActionRegistry(
            (
                Action(
                    name="find_memory",
                    handler=fake_handler,
                ),
            )
        )

        rendered = render_capability_context(
            build_capability_context(
                registry
            )
        )

        self.assertIn(
            "find_memory: search explicitly saved persistent "
            "memories when the user asks.",
            rendered,
        )

    def test_render_omits_unregistered_action_description(
        self,
    ) -> None:
        registry = ActionRegistry(
            (
                Action(
                    name="save_memory",
                    handler=fake_handler,
                ),
            )
        )

        rendered = render_capability_context(
            build_capability_context(
                registry
            )
        )

        self.assertNotIn(
            "find_memory:",
            rendered,
        )

    def test_render_explains_memory_search_in_ordinary_language(
        self,
    ) -> None:
        registry = ActionRegistry(
            (
                Action(
                    name="find_memory",
                    handler=fake_handler,
                ),
            )
        )

        rendered = render_capability_context(
            build_capability_context(
                registry
            )
        )

        self.assertIn(
            "The assistant can search and retrieve memories that were "
            "explicitly saved, when the user asks.",
            rendered,
        )

    def test_render_explains_application_launch_in_ordinary_language(
        self,
    ) -> None:
        registry = ActionRegistry(
            (
                Action(
                    name="launch_application",
                    handler=fake_handler,
                ),
            )
        )

        rendered = render_capability_context(
            build_capability_context(
                registry
            )
        )

        self.assertIn(
            "The assistant can request the launch of an allowed "
            "application on the user's computer.",
            rendered,
        )

    def test_execution_restrictions_do_not_erase_capability(
        self,
    ) -> None:
        registry = ActionRegistry(
            (
                Action(
                    name="launch_application",
                    handler=fake_handler,
                ),
            )
        )

        rendered = render_capability_context(
            build_capability_context(
                registry
            )
        )

        self.assertIn(
            "Registered actions are real current capabilities.",
            rendered,
        )
        self.assertIn(
            "Permissions and confirmations affect execution, not whether "
            "a registered capability exists.",
            rendered,
        )
        self.assertIn(
            "Explicit memory actions are different from automatic "
            "contextual memory retrieval.",
            rendered,
        )

    def test_memory_search_is_explicitly_a_current_capability(
        self,
    ) -> None:
        registry = ActionRegistry(
            (
                Action(
                    name="find_memory",
                    handler=fake_handler,
                ),
            )
        )

        rendered = render_capability_context(
            build_capability_context(
                registry
            )
        )

        self.assertIn(
            "If asked whether the assistant can search explicitly "
            "saved memories on request, the correct answer is yes.",
            rendered,
        )
        self.assertIn(
            "Automatic contextual memory retrieval means passive or "
            "automatic recall without an explicit search request.",
            rendered,
        )
        self.assertIn(
            "It is not the same capability as find_memory.",
            rendered,
        )

    def test_capability_context_forbids_invented_rationales(
        self,
    ) -> None:
        rendered = render_capability_context(
            build_capability_context(
                ActionRegistry()
            )
        )

        self.assertIn(
            "Do not invent security reasons, design rationales or "
            "implementation reasons for capability limits.",
            rendered,
        )

    def test_render_distinguishes_memory_modes(
        self,
    ) -> None:
        registry = ActionRegistry(
            (
                Action(
                    name="save_memory",
                    handler=fake_handler,
                ),
                Action(
                    name="find_memory",
                    handler=fake_handler,
                ),
            )
        )

        rendered = render_capability_context(
            build_capability_context(
                registry
            )
        )

        self.assertIn(
            "Persistent memories are available only through "
            "explicit registered memory actions.",
            rendered,
        )
        self.assertIn(
            "Automatic contextual memory retrieval is not available.",
            rendered,
        )
        self.assertIn(
            "Conversation history is temporary to the current conversation.",
            rendered,
        )

    def test_render_describes_unavailable_hardware(
        self,
    ) -> None:
        rendered = render_capability_context(
            build_capability_context(
                ActionRegistry()
            )
        )

        self.assertIn(
            "Visual or webcam input is not currently available.",
            rendered,
        )
        self.assertIn(
            "Audio input is not currently available.",
            rendered,
        )
        self.assertIn(
            "Robot or physical hardware control is not currently available.",
            rendered,
        )


    def test_render_respects_available_hardware_flags(
        self,
    ) -> None:
        context = CapabilityContext(
            available_actions=(),
            visual_input=True,
            audio_input=True,
            robot_control=True,
        )

        rendered = render_capability_context(
            context
        )

        self.assertIn(
            "Visual or webcam input is currently available.",
            rendered,
        )
        self.assertIn(
            "Audio input is currently available.",
            rendered,
        )
        self.assertIn(
            "Robot or physical hardware control is currently available.",
            rendered,
        )

        self.assertNotIn(
            "Visual or webcam input is not currently available.",
            rendered,
        )
        self.assertNotIn(
            "Audio input is not currently available.",
            rendered,
        )
        self.assertNotIn(
            "Robot or physical hardware control is not currently available.",
            rendered,
        )

    def test_render_respects_automatic_memory_retrieval_flag(
        self,
    ) -> None:
        context = CapabilityContext(
            available_actions=(),
            automatic_memory_retrieval=True,
        )

        rendered = render_capability_context(
            context
        )

        self.assertIn(
            "Automatic contextual memory retrieval is available.",
            rendered,
        )
        self.assertNotIn(
            "Automatic contextual memory retrieval is not available.",
            rendered,
        )
        self.assertIn(
            "Persistent memories can provide automatic contextual data",
            rendered,
        )
        self.assertNotIn(
            "Persistent memories are available only through",
            rendered,
        )

    def test_render_does_not_invent_persistent_memory_actions(
        self,
    ) -> None:
        context = CapabilityContext(
            available_actions=(),
        )

        rendered = render_capability_context(
            context
        )

        self.assertNotIn(
            "Persistent memories are available only through "
            "explicit registered memory actions.",
            rendered,
        )
        self.assertIn(
            "No registered persistent memory action is currently available.",
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
