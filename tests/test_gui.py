"""Headless tests for the temporary tkinter conversation interface."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from assistant_ia.core.assistant import AssistantCore
from assistant_ia.intelligence.intent import Intent
from assistant_ia.intelligence.response import ModelResponse
from assistant_ia.interfaces.gui import (
    INITIAL_ASSISTANT_MESSAGE,
    RUNTIME_ERROR_MESSAGE,
    GuiConversationController,
)
from assistant_ia.runtime import AssistantRuntime


class SequenceModelClient:
    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)

    def generate_response(self, messages: tuple[object, ...]) -> ModelResponse:
        return ModelResponse(
            content=self._responses.pop(0),
            model="fake-model",
            intent=Intent(name="conversation"),
        )


class GuiConversationControllerTests(unittest.TestCase):
    def _controller(self, *responses: str) -> GuiConversationController:
        return GuiConversationController(
            AssistantRuntime(
                AssistantCore(
                    model_client=SequenceModelClient(*responses)
                )
            )
        )

    def test_send_replaces_user_then_response_at_completion(self) -> None:
        controller = self._controller(
            "Première réponse.",
            "Deuxième réponse.",
        )

        controller.begin_message("Premier message.")
        self.assertEqual(
            controller.state.assistant_message,
            INITIAL_ASSISTANT_MESSAGE,
        )
        first_response = controller.generate_response()

        self.assertEqual(
            controller.state.assistant_message,
            INITIAL_ASSISTANT_MESSAGE,
        )
        self.assertTrue(controller.state.is_waiting)

        controller.complete_response(first_response)
        controller.begin_message("Deuxième message.")

        self.assertEqual(
            controller.state.user_message,
            "Deuxième message.",
        )
        self.assertEqual(
            controller.state.assistant_message,
            "Première réponse.",
        )

        second_response = controller.generate_response()
        controller.complete_response(second_response)

        self.assertEqual(
            controller.state.assistant_message,
            "Deuxième réponse.",
        )
        self.assertFalse(controller.state.is_waiting)

    def test_controller_calls_runtime_boundary(self) -> None:
        controller = self._controller("Réponse.")
        controller.begin_message("Question.")

        with patch.object(
            controller.runtime,
            "process_message",
            wraps=controller.runtime.process_message,
        ) as process_message:
            response = controller.generate_response()

        process_message.assert_called_once_with("Question.")
        self.assertEqual(response, "Réponse.")

    def test_runtime_error_becomes_clean_bubble_text(self) -> None:
        controller = self._controller("Réponse inutilisée.")
        controller.begin_message("Question.")

        with patch.object(
            controller.runtime,
            "process_message",
            side_effect=RuntimeError("secret technique"),
        ):
            response = controller.generate_response()

        controller.complete_response(response)

        self.assertEqual(
            controller.state.assistant_message,
            RUNTIME_ERROR_MESSAGE,
        )
        self.assertNotIn(
            "secret technique",
            controller.state.assistant_message,
        )

    def test_gui_has_worker_after_canvas_and_no_hardware_import(self) -> None:
        gui_path = (
            Path(__file__).parents[1]
            / "src"
            / "assistant_ia"
            / "interfaces"
            / "gui.py"
        )
        source = gui_path.read_text(encoding="utf-8")

        self.assertIn("threading.Thread", source)
        self.assertIn("self._root.after(", source)
        self.assertIn("tk.Canvas", source)
        self.assertNotIn("assistant_ia.hardware", source)


if __name__ == "__main__":
    unittest.main()
