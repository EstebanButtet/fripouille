"""Temporary lightweight tkinter conversation interface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading
import tkinter as tk

from assistant_ia.application import (
    ApplicationInitializationError,
    build_default_runtime,
)
from assistant_ia.core.assistant import AssistantCoreError
from assistant_ia.interfaces.diagnostics import (
    ConsoleDiagnosticReporter,
    display_runtime_error,
)
from assistant_ia.runtime import AssistantRuntime

WINDOW_TITLE = "Fripouille"
INITIAL_ASSISTANT_MESSAGE = "Salut. Moi, c'est Fripouille."
MODEL_ERROR_MESSAGE = (
    "Je n'arrive pas à joindre mon modèle local pour le moment."
)
RUNTIME_ERROR_MESSAGE = (
    "Quelque chose a coincé. Le détail reste dans le diagnostic."
)
DATABASE_ERROR_MESSAGE = (
    "Je n'arrive pas à ouvrir ma mémoire locale."
)


@dataclass(slots=True)
class ChatDisplayState:
    """Represent only the exchange currently visible in the GUI."""

    user_message: str = ""
    assistant_message: str = INITIAL_ASSISTANT_MESSAGE
    is_waiting: bool = False

    def begin_turn(self, user_message: str) -> None:
        """Replace the user bubble and preserve the assistant bubble."""
        normalized_message = user_message.strip()
        if not normalized_message:
            raise ValueError("GUI user message cannot be empty.")
        if self.is_waiting:
            raise RuntimeError("A GUI response is already pending.")
        self.user_message = normalized_message
        self.is_waiting = True

    def finish_turn(self, assistant_message: str) -> None:
        """Replace the assistant bubble when the response is ready."""
        normalized_message = assistant_message.strip()
        if not normalized_message:
            raise ValueError("GUI assistant message cannot be empty.")
        self.assistant_message = normalized_message
        self.is_waiting = False


class GuiConversationController:
    """Keep testable bubble state around the shared AssistantRuntime."""

    def __init__(self, runtime: AssistantRuntime) -> None:
        if not isinstance(runtime, AssistantRuntime):
            raise TypeError("GUI requires an AssistantRuntime.")
        self._runtime = runtime
        self.state = ChatDisplayState()

    @property
    def runtime(self) -> AssistantRuntime:
        """Return the only assistant boundary called by the GUI."""
        return self._runtime

    def begin_message(self, user_message: str) -> None:
        """Start one visible turn without clearing the previous answer."""
        self.state.begin_turn(user_message)

    def generate_response(
        self,
        error_reporter: Callable[[BaseException], None] | None = None,
    ) -> str:
        """Generate the current response; callers may run this in a worker."""
        if not self.state.is_waiting:
            raise RuntimeError("No GUI response is pending.")
        try:
            return self._runtime.process_message(
                self.state.user_message
            )
        except AssistantCoreError as error:
            if error_reporter is not None:
                error_reporter(error)
            return MODEL_ERROR_MESSAGE
        except Exception as error:
            if error_reporter is not None:
                error_reporter(error)
            return RUNTIME_ERROR_MESSAGE

    def complete_response(self, assistant_message: str) -> None:
        """Finish the visible turn on the GUI thread."""
        self.state.finish_turn(assistant_message)


class FripouilleWindow:
    """Display one comic-like exchange around a fixed temporary face."""

    def __init__(
        self,
        root: tk.Tk,
        controller: GuiConversationController,
        *,
        debug: bool = False,
    ) -> None:
        self._root = root
        self._controller = controller
        self._debug = debug

        root.title(WINDOW_TITLE)
        root.configure(background="#20242b")
        root.minsize(720, 420)

        content = tk.Frame(root, background="#20242b")
        content.pack(fill=tk.BOTH, expand=True, padx=24, pady=24)

        self._create_face(content).pack(
            side=tk.LEFT,
            padx=(0, 24),
            anchor=tk.N,
        )

        conversation = tk.Frame(
            content,
            background="#20242b",
        )
        conversation.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
        )

        self._user_bubble = self._create_bubble(
            conversation,
            background="#d9e8ff",
        )
        self._user_bubble.pack(
            fill=tk.X,
            pady=(0, 16),
        )

        self._assistant_bubble = self._create_bubble(
            conversation,
            background="#f5e7be",
        )
        self._assistant_bubble.pack(
            fill=tk.X,
            pady=(0, 10),
        )

        self._status = tk.Label(
            conversation,
            text="",
            background="#20242b",
            foreground="#c7ccd4",
            anchor=tk.W,
        )
        self._status.pack(fill=tk.X)

        input_row = tk.Frame(root, background="#20242b")
        input_row.pack(fill=tk.X, padx=24, pady=(0, 24))

        self._entry = tk.Entry(
            input_row,
            font=("Segoe UI", 11),
        )
        self._entry.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            ipady=7,
        )
        self._entry.bind("<Return>", self._submit_message)

        self._send_button = tk.Button(
            input_row,
            text="Envoyer",
            command=self._submit_message,
            padx=16,
            pady=7,
        )
        self._send_button.pack(side=tk.LEFT, padx=(10, 0))

        self._render_state()
        self._entry.focus_set()

    @staticmethod
    def _create_bubble(
        parent: tk.Misc,
        *,
        background: str,
    ) -> tk.Label:
        return tk.Label(
            parent,
            text="",
            justify=tk.LEFT,
            anchor=tk.NW,
            wraplength=470,
            background=background,
            foreground="#17191d",
            font=("Segoe UI", 11),
            relief=tk.SOLID,
            borderwidth=1,
            padx=14,
            pady=12,
        )

    @staticmethod
    def _create_face(parent: tk.Misc) -> tk.Canvas:
        """Create the replaceable provisional face widget."""
        face = tk.Canvas(
            parent,
            width=170,
            height=170,
            background="#20242b",
            highlightthickness=0,
        )
        face.create_oval(
            12,
            12,
            158,
            158,
            fill="#f5d36c",
            outline="#17191d",
            width=3,
        )
        face.create_oval(52, 58, 68, 78, fill="#17191d")
        face.create_oval(102, 58, 118, 78, fill="#17191d")
        face.create_arc(
            50,
            72,
            120,
            130,
            start=200,
            extent=140,
            style=tk.ARC,
            outline="#17191d",
            width=4,
        )
        return face

    def _submit_message(self, event: object | None = None) -> str:
        if self._controller.state.is_waiting:
            return "break"

        user_message = self._entry.get().strip()
        if not user_message:
            return "break"

        self._controller.begin_message(user_message)
        self._entry.delete(0, tk.END)
        self._entry.configure(state=tk.DISABLED)
        self._send_button.configure(state=tk.DISABLED)
        self._render_state()

        worker = threading.Thread(
            target=self._process_message,
            name="fripouille-gui-response",
            daemon=True,
        )
        worker.start()
        return "break"

    def _process_message(self) -> None:
        response = self._controller.generate_response(
            display_runtime_error if self._debug else None
        )
        self._root.after(
            0,
            lambda: self._finish_response(response),
        )

    def _finish_response(self, response: str) -> None:
        self._controller.complete_response(response)
        self._entry.configure(state=tk.NORMAL)
        self._send_button.configure(state=tk.NORMAL)
        self._render_state()
        self._entry.focus_set()

    def _render_state(self) -> None:
        state = self._controller.state
        user_text = (
            f"Vous\n{state.user_message}"
            if state.user_message
            else "Vous\n—"
        )
        self._user_bubble.configure(text=user_text)
        self._assistant_bubble.configure(
            text=f"Fripouille\n{state.assistant_message}"
        )
        self._status.configure(
            text="..." if state.is_waiting else ""
        )


def run_gui(*, debug: bool = False) -> None:
    """Build and run the temporary Windows GUI."""
    try:
        if debug:
            runtime = build_default_runtime(
                diagnostic_reporter=ConsoleDiagnosticReporter(),
            )
        else:
            runtime = build_default_runtime()
    except ApplicationInitializationError:
        print(DATABASE_ERROR_MESSAGE)
        return

    root = tk.Tk()
    FripouilleWindow(
        root,
        GuiConversationController(runtime),
        debug=debug,
    )
    root.mainloop()
