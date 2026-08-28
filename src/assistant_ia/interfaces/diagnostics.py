"""Small opt-in console diagnostics for interactive interfaces."""

from __future__ import annotations

from assistant_ia.runtime import TurnDiagnostics


class ConsoleDiagnosticReporter:
    """Print internal turn data only when explicitly installed."""

    def report(self, diagnostics: TurnDiagnostics) -> None:
        """Print a compact technical snapshot of one completed turn."""
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
    """Print a technical interface error without putting it in a bubble."""
    print(f"[debug] erreur_runtime={error!r}")
