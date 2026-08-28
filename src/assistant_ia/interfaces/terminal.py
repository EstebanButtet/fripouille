"""Interactive terminal interface for the personal AI assistant."""

from __future__ import annotations

from assistant_ia.application import (
    ApplicationInitializationError,
    build_default_runtime,
)
from assistant_ia.core.assistant import AssistantCoreError
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
    """Display the application title and initial instructions."""
    print()
    print(f"=== {APP_TITLE} ===")
    print("Interface terminal locale — modèle d'IA local via Ollama.")
    print(f"Tapez {COMMAND_HELP} pour afficher les commandes disponibles.")
    print()


def display_help() -> None:
    """Display the commands supported by the terminal interface."""
    print()
    print("Commandes disponibles :")
    print(f"  {COMMAND_HELP:<8} Afficher cette aide")
    print(f"  {COMMAND_RESET:<8} Réinitialiser la conversation")
    print(f"  {COMMAND_QUIT:<8} Quitter l'assistant")
    print()


def display_assistant_message(message: str) -> None:
    """Display a message produced by the assistant."""
    print(f"{ASSISTANT_PREFIX}{message}")


def request_terminal_confirmation(
    request: ConfirmationRequest,
) -> bool:
    """Request explicit confirmation from the terminal user."""
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


def run_terminal() -> None:
    """Start and manage the interactive terminal session."""
    display_welcome()

    try:
        runtime = build_default_runtime(
            confirmation_handler=request_terminal_confirmation,
        )
    except ApplicationInitializationError:
        display_assistant_message(
            DATABASE_ERROR_MESSAGE
        )
        return

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
