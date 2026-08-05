"""Interactive terminal interface for the personal AI assistant."""

from __future__ import annotations

APP_TITLE = "Assistant IA personnel"
USER_PROMPT = "Vous > "
ASSISTANT_PREFIX = "Assistant > "

COMMAND_HELP = "/help"
COMMAND_RESET = "/reset"
COMMAND_QUIT = "/quit"


def display_welcome() -> None:
    """Display the application title and initial instructions."""
    print()
    print(f"=== {APP_TITLE} ===")
    print("Interface terminal locale — aucun modèle d'IA n'est encore connecté.")
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


def build_temporary_response(user_message: str) -> str:
    """Build a temporary response until the assistant core is implemented."""
    return (
        "Message reçu : "
        f"{user_message!r}. "
        "Le noyau de l'assistant n'est pas encore connecté."
    )


def run_terminal() -> None:
    """Start and manage the interactive terminal session."""
    conversation: list[str] = []

    display_welcome()

    while True:
        try:
            user_message = input(USER_PROMPT).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            display_assistant_message("Arrêt demandé. À bientôt.")
            break

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
            conversation.clear()
            display_assistant_message("La conversation a été réinitialisée.")
            continue

        conversation.append(user_message)
        response = build_temporary_response(user_message)
        display_assistant_message(response)


if __name__ == "__main__":
    run_terminal()