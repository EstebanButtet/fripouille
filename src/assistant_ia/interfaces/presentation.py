"""Présentation utilisateur des résultats internes d'un tour.

Les actions et la promotion mémoire produisent des textes techniquement précis,
avec parfois des identifiants SQLite. Cette couche transforme ces résultats en
formulations naturelles pour les interfaces, sans changer l'intention, le
consentement requis ni l'opération déjà résolue par le coeur.
"""

from __future__ import annotations

import re

from assistant_ia.intelligence.intent import Intent
from assistant_ia.memory.promotion import MemoryPromotionProposal

_PERSISTENT_IDENTIFIER = re.compile(r'\s*\[#\d+\]\s*')


def build_user_facing_response(
    raw_response: str,
    *,
    intent: Intent | None,
    memory_proposal: MemoryPromotionProposal | None,
    awaiting_memory_confirmation: bool,
) -> str:
    '''Masquer la plomberie de persistance tout en conservant le résultat.

    ``raw_response`` vient du coeur. Les autres paramètres permettent de
    distinguer une action explicite, une proposition automatique et la réponse
    à une confirmation, sans réinterpréter le texte avec un modèle.
    '''
    if not isinstance(raw_response, str):
        raise TypeError('Presented assistant response must be a string.')

    if memory_proposal is not None:
        if awaiting_memory_confirmation or intent is not None:
            return _present_memory_proposal(
                memory_proposal,
            )
        return _present_memory_confirmation(raw_response)

    if intent is None:
        return raw_response

    if intent.name == 'save_memory':
        if raw_response.startswith('Souvenir enregistré :'):
            return 'D’accord, je garde ça en tête.'
        return raw_response

    if intent.name == 'find_memory':
        return _remove_persistent_identifiers(raw_response)

    if intent.name == 'delete_memory':
        if raw_response.startswith('Souvenir supprimé :'):
            return 'D’accord, je l’ai oublié.'
        return raw_response

    return raw_response


def _present_memory_proposal(
    proposal: MemoryPromotionProposal,
) -> str:
    """Formuler naturellement une proposition mémoire déjà classée."""
    candidate_content = proposal.candidate.content

    if proposal.operation == 'already_known':
        proposal_text = 'Je l’avais déjà en tête.'
    elif proposal.operation == 'create':
        proposal_text = (
            'Ça me paraît utile à retenir. '
            'Je le garde en tête ?'
        )
    elif proposal.operation == 'possible_duplicate':
        proposal_text = (
            'Ça ressemble à quelque chose que je connais déjà. '
            'Je le garde tout de même séparément ?'
        )
    else:
        proposal_text = (
            f'Correction comprise : « {candidate_content} ». '
            'Je garde ça en tête ?'
        )

    return proposal_text


def _present_memory_confirmation(raw_response: str) -> str:
    """Présenter le résultat d'un consentement sans identifiant technique."""
    if raw_response.startswith(
        ('Souvenir enregistré :', 'Souvenir corrigé :')
    ):
        return 'D’accord.'
    return _remove_persistent_identifiers(raw_response)


def _remove_persistent_identifiers(response: str) -> str:
    """Retirer uniquement les marqueurs d'identifiant SQLite des lignes."""
    return '\n'.join(
        _PERSISTENT_IDENTIFIER.sub(' ', line).rstrip()
        for line in response.splitlines()
    )
