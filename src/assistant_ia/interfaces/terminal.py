"""Interface terminal interactive et historique de Fripouille.

Ce module affiche l'invite, traite trois commandes locales et transmet les
autres messages à un unique :class:`AssistantRuntime`. Il fournit aussi le
handler interactif de confirmation utilisé par les actions sensibles.
Les erreurs connues de base et de modèle sont transformées en messages utiles
sans exposer une trace technique dans la conversation.
"""

from __future__ import annotations

from assistant_ia.application import (
    ApplicationInitializationError,
    build_default_runtime,
)
from assistant_ia.core.assistant import AssistantCoreError
from assistant_ia.interfaces.diagnostics import ConsoleDiagnosticReporter
from assistant_ia.security.confirmation import ConfirmationRequest

APP_TITLE = "Assistant IA personnel"
USER_PROMPT = "Vous > "
ASSISTANT_PREFIX = "Assistant > "

COMMAND_HELP = "/help"
COMMAND_RESET = "/reset"
COMMAND_QUIT = "/quit"

MODEL_ERROR_MESSAGE = (
    "Le modèle local n'a pas pu produire de réponse. "
    "Vérifiez qu'Ollama est démarré et que le modèle configuré est installé."
)

DATABASE_ERROR_MESSAGE = (
    "La base de données locale n'a pas pu être initialisée. "
    "Vérifiez les droits d'accès au dossier de données de l'application."
)


def display_welcome() -> None:
    """Afficher le titre de l'application et l'instruction initiale."""
    print()
    print(f"=== {APP_TITLE} ===")
    print("Interface terminal locale — modèle d'IA local via Ollama.")
    print(f"Tapez {COMMAND_HELP} pour afficher les commandes disponibles.")
    print()


def display_help() -> None:
    """Afficher les commandes gérées localement par le terminal."""
    print()
    print("Commandes disponibles :")
    print(f"  {COMMAND_HELP:<8} Afficher cette aide")
    print(f"  {COMMAND_RESET:<8} Réinitialiser la conversation")
    print(f"  {COMMAND_QUIT:<8} Quitter l'assistant")
    print()


def display_assistant_message(message: str) -> None:
    """Afficher un texte final déjà produit par le runtime."""
    print(f"{ASSISTANT_PREFIX}{message}")


def request_terminal_confirmation(
    request: ConfirmationRequest,
) -> bool:
    """Demander une confirmation explicite à la personne dans le terminal.

    Seuls ``o`` et ``oui`` autorisent l'action ; Entrée et toute autre réponse
    refusent. Ce défaut fermé empêche une validation accidentelle.
    """
    if not isinstance(request, ConfirmationRequest):
        raise TypeError(
            "Terminal confirmation requires a ConfirmationRequest."
        )

    response = input(
        f"Confirmer : {request.description} ? [o/N] "
    ).strip().casefold()

    return response in {
        "o",
        "oui",
    }


def run_terminal(*, debug: bool = False) -> None:
    """Construire le runtime puis gérer la boucle interactive.

    Une seule instance est conservée pendant la boucle, ce qui préserve le
    contexte conversationnel. ``/reset`` réinitialise cet état temporaire sans
    effacer les données SQLite ; ``/quit`` ne passe jamais par Ollama.
    """
    display_welcome()

    try:
        if debug:
            runtime = build_default_runtime(
                confirmation_handler=request_terminal_confirmation,
                diagnostic_reporter=ConsoleDiagnosticReporter(),
            )
        else:
            runtime = build_default_runtime(
                confirmation_handler=request_terminal_confirmation,
            )
    except ApplicationInitializationError:
        display_assistant_message(
            DATABASE_ERROR_MESSAGE
        )
        return

    # Les commandes d'interface sont résolues avant ``process_message`` : elles
    # restent du contrôle terminal et ne peuvent devenir des intentions LLM.
    while True:
        try:
            user_message = input(USER_PROMPT).strip()

            if not user_message:
                continue

            normalized_message = user_message.casefold()

            if normalized_message == COMMAND_QUIT:
                display_assistant_message("À bientôt.")
                break

            if normalized_message == COMMAND_HELP:
                display_help()
                continue

            if normalized_message == COMMAND_RESET:
                runtime.reset_conversation()
                display_assistant_message(
                    "La conversation a été réinitialisée."
                )
                continue

            response = runtime.process_message(
                user_message
            )
            display_assistant_message(response)
        except (KeyboardInterrupt, EOFError):
            print()
            display_assistant_message(
                "Arrêt demandé. À bientôt."
            )
            break
        except AssistantCoreError:
            display_assistant_message(
                MODEL_ERROR_MESSAGE
            )


if __name__ == "__main__":
    run_terminal()
