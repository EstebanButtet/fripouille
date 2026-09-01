"""Point d'entrée de Fripouille lorsqu'il est lancé comme module Python.

Ce module lit uniquement les options de la ligne de commande, choisit
l'interface terminal ou l'interface graphique provisoire, puis lui cède le
contrôle. Il ne construit pas lui-même le coeur, la mémoire ou les actions :
chaque interface demande cet assemblage à :mod:`assistant_ia.application`.

Flux de démarrage::

    python -m assistant_ia [--gui] [--debug]
        -> main()
        -> run_terminal() ou run_gui()
        -> construction de l'application par l'interface choisie
"""

from __future__ import annotations

import argparse

from assistant_ia.interfaces.terminal import run_terminal


def _build_argument_parser() -> argparse.ArgumentParser:
    """Décrire et retourner les options reconnues au démarrage.

    Garder cette fonction séparée permet aux tests de contrôler le contrat de
    la ligne de commande sans lancer réellement une interface.
    """
    parser = argparse.ArgumentParser(
        description="Lancer Fripouille.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="utiliser l'interface graphique provisoire",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="afficher les diagnostics techniques dans la console",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Sélectionner l'interface demandée et démarrer Fripouille.

    ``argv`` sert principalement aux tests ; avec ``None``, ``argparse`` lit
    les arguments réels du processus. L'import de la GUI est volontairement
    tardif afin que le terminal reste utilisable même si tkinter est absent.
    """
    arguments = _build_argument_parser().parse_args(argv)

    if arguments.gui:
        # tkinter n'est nécessaire que dans ce chemin. Cette frontière évite
        # d'imposer une dépendance graphique au lancement terminal historique.
        from assistant_ia.interfaces.gui import run_gui

        run_gui(debug=arguments.debug)
        return

    if arguments.debug:
        run_terminal(debug=True)
    else:
        run_terminal()


if __name__ == "__main__":
    main()
