"""Interface conversationnelle tkinter légère et provisoire (FRP-IA-02B).

La GUI possède un contrôleur testable, un état d'affichage et une fenêtre
tkinter. Le traitement potentiellement lent du runtime s'effectue dans un
thread worker afin de ne pas bloquer la boucle graphique ; toute modification
des widgets revient ensuite sur le thread tkinter grâce à ``root.after``.

Le visage dessiné ici est un substitut temporaire, pas le « vrai visage » futur
de FRP-IA-10. Cette interface n'ajoute aucune capacité au coeur et, faute de
handler de confirmation GUI, les actions sensibles restent refusées par défaut.
"""

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
    """Représenter uniquement l'échange actuellement visible.

    Cette dataclass mutable est un état de vue, pas l'historique de
    ``ConversationContext``. La GUI remplace ses deux bulles à chaque tour,
    tandis que le runtime conserve la conversation complète bornée.
    """

    user_message: str = ""
    assistant_message: str = INITIAL_ASSISTANT_MESSAGE
    is_waiting: bool = False

    def begin_turn(self, user_message: str) -> None:
        """Remplacer la bulle utilisateur et marquer la réponse en attente."""
        normalized_message = user_message.strip()
        if not normalized_message:
            raise ValueError("GUI user message cannot be empty.")
        if self.is_waiting:
            raise RuntimeError("A GUI response is already pending.")
        self.user_message = normalized_message
        self.is_waiting = True

    def finish_turn(self, assistant_message: str) -> None:
        """Remplacer la bulle assistant et terminer l'attente visible."""
        normalized_message = assistant_message.strip()
        if not normalized_message:
            raise ValueError("GUI assistant message cannot be empty.")
        self.assistant_message = normalized_message
        self.is_waiting = False


class GuiConversationController:
    """Isoler l'état testable de la vue autour du runtime partagé.

    Le contrôleur ignore les widgets tkinter. Il peut donc être testé sans
    écran réel et constitue l'unique chemin de la GUI vers l'assistant.
    """

    def __init__(self, runtime: AssistantRuntime) -> None:
        """Créer le contrôleur avec l'unique runtime de la session GUI."""
        if not isinstance(runtime, AssistantRuntime):
            raise TypeError("GUI requires an AssistantRuntime.")
        self._runtime = runtime
        self.state = ChatDisplayState()

    @property
    def runtime(self) -> AssistantRuntime:
        """Retourner l'unique frontière assistant appelée par la GUI."""
        return self._runtime

    def begin_message(self, user_message: str) -> None:
        """Commencer un tour visible sans effacer encore l'ancienne réponse."""
        self.state.begin_turn(user_message)

    def generate_response(
        self,
        error_reporter: Callable[[BaseException], None] | None = None,
    ) -> str:
        """Générer la réponse courante, éventuellement dans un worker.

        Les erreurs attendues du modèle reçoivent un message dédié. Toute autre
        exception est rapportée si un callback de diagnostic a été fourni,
        puis convertie en texte générique pour la bulle.
        """
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
        """Terminer l'état visible depuis le thread GUI."""
        self.state.finish_turn(assistant_message)


class FripouilleWindow:
    """Afficher un échange à deux bulles autour d'un visage provisoire.

    L'objet possède les widgets pendant toute la vie de la fenêtre. Il délègue
    les décisions au contrôleur et ne lit jamais directement le coeur.
    """

    def __init__(
        self,
        root: tk.Tk,
        controller: GuiConversationController,
        *,
        debug: bool = False,
    ) -> None:
        """Construire les widgets et rendre l'état initial de la conversation."""
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
        """Créer une bulle réutilisable avec le style commun de la fenêtre."""
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
        """Créer le widget de visage provisoire destiné à être remplacé."""
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
        """Valider la saisie et lancer un seul worker de réponse.

        Le bouton et le champ sont désactivés jusqu'au retour afin d'empêcher
        deux appels concurrents sur le même runtime et son contexte mutable.
        ``"break"`` indique à tkinter de ne pas poursuivre le traitement de
        l'événement Entrée.
        """
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

        # Aucun widget ne doit être modifié depuis ce worker ; il appelle
        # seulement le contrôleur et remet ensuite le résultat à tkinter.
        worker = threading.Thread(
            target=self._process_message,
            name="fripouille-gui-response",
            daemon=True,
        )
        worker.start()
        return "break"

    def _process_message(self) -> None:
        """Calculer la réponse hors du thread GUI puis planifier sa livraison."""
        response = self._controller.generate_response(
            display_runtime_error if self._debug else None
        )
        # ``after(0, ...)`` replace l'appel de finition dans la boucle tkinter,
        # seule autorisée à toucher aux widgets de la fenêtre.
        self._root.after(
            0,
            lambda: self._finish_response(response),
        )

    def _finish_response(self, response: str) -> None:
        """Réactiver la saisie et afficher la réponse sur le thread GUI."""
        self._controller.complete_response(response)
        self._entry.configure(state=tk.NORMAL)
        self._send_button.configure(state=tk.NORMAL)
        self._render_state()
        self._entry.focus_set()

    def _render_state(self) -> None:
        """Projeter l'état du contrôleur dans les widgets visibles."""
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
    """Construire le runtime, la fenêtre provisoire et lancer tkinter."""
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
