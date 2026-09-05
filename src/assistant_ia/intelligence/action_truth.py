"""FRP-IA-14 : garde de sincérité, aucune action ni extraction de paramètre."""
import re

from assistant_ia.intelligence.intent import Intent

CLARIFY_ACTION = (
    "Je n'ai exécuté aucune action pour cette demande. "
    "Précise une seule action et ses informations exactes, par exemple le titre de la tâche."
)


def grounded_action(intent: Intent, source: str) -> bool:
    # Réutiliser la normalisation et la preuve lexicale existantes, sans créer
    # un second extracteur. Import tardif : les analyseurs utilisent ModelClient.
    from assistant_ia.intelligence.memory_candidates import _content_is_lexically_grounded, _normalized_text
    field = {"create_task": "title", "save_memory": "content", "write_journal": "content"}.get(intent.name)
    if field is None: return True
    value = intent.parameters.get(field)
    if value is not None and not _content_is_lexically_grounded(value, source):
        return False
    if intent.name == "create_task" and intent.parameters.get("due_at"):
        if not re.search(r"\d|demain|aujourd|lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|semaine|mois|heure", _normalized_text(source)):
            return False
    return True


def truthful_conversation(content: str, source: str) -> str:
    """Filtre conservateur de formulations françaises, pas une preuve sémantique.

    La seule preuve d'exécution reste ActionExecutionResult. En cas de demande
    d'action mal classée ou de promesse repérée, demander une clarification.
    """
    from assistant_ia.intelligence.memory_candidates import _normalized_text
    request = _normalized_text(source)
    answer = _normalized_text(content).replace("’", "'")
    ambiguous_request = re.search(
        r"(?:je voudrais|je veux|peux.tu|cree|creer|enregistre).{0,70}\btache\b|rappelle[- ](?:moi|moins)", request)
    if ambiguous_request:
        return CLARIFY_ACTION
    for sentence in re.split(r"[.!?;\n]", answer):
        if re.search(r"\b(?:n'ai|ne|n'est|n'a|n'ai)\b.{0,45}\b(?:pas|jamais|aucun|aucune)\b", sentence):
            continue
        claim = re.search(r"\b(?:j'ai|je vais|je viens de|je suis en train de|j'enregistre|je cree|je te rappellerai|c'est note)\b", sentence)
        action = re.search(r"\b(?:cree\w*|enregistr\w*|supprim\w*|ajout\w*|planifi\w*|rappell\w*|lanc\w*|note)\b", sentence)
        receipt = re.search(r"\b(?:tache|souvenir|rappel|entree|application)\b.{0,35}\b(?:cree\w*|enregistr\w*|supprim\w*|ajout\w*|planifi\w*|lance\w*)\b", sentence)
        if (claim and action) or receipt:
            return CLARIFY_ACTION
    return content
