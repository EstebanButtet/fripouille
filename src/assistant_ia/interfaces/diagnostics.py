"""Diagnostics console facultatifs pour les interfaces interactives.

Les diagnostics sont installés uniquement avec ``--debug`` et restent hors des
bulles ainsi que de l'historique envoyé à Ollama. Ils observent un
``TurnDiagnostics`` immuable sans influencer la réponse.
"""

from __future__ import annotations

from assistant_ia.runtime import TurnDiagnostics


class ConsoleDiagnosticReporter:
    """Afficher les données internes seulement lorsqu'il est injecté."""

    def report(self, diagnostics: TurnDiagnostics) -> None:
        """Afficher une photographie technique compacte du tour terminé."""
        intent_name = (
            diagnostics.intent.name
            if diagnostics.intent is not None
            else None
        )
        print(f"[debug] intention={intent_name!r}")
        print(f"[debug] réponse_brute={diagnostics.raw_response!r}")

        if diagnostics.memory_candidates:
            print(
                "[debug] candidats_mémoire="
                f"{diagnostics.memory_candidates!r}"
            )

        if diagnostics.memory_promotion_proposal is not None:
            print(
                "[debug] promotion_mémoire="
                f"{diagnostics.memory_promotion_proposal!r}"
            )


def display_runtime_error(error: BaseException) -> None:
    """Afficher une erreur technique sans la placer dans la conversation."""
    print(f"[debug] erreur_runtime={error!r}")
