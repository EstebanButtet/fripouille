"""Analyse Ollama dédiée aux candidats mémoire non persistants.

Le message utilisateur courant est d'abord filtré localement, puis envoyé à
un prompt structuré distinct de la conversation. Chaque proposition reçue est
recoupée avec le texte autorisé, bornée et reconstruite en
:class:`MemoryCandidate`. Ce module ne consulte pas SQLite et ne décide jamais
qu'un candidat doit devenir un souvenir : cette responsabilité appartient au
service de promotion de :mod:`assistant_ia.memory.promotion`.
"""

from __future__ import annotations

import json
from json import JSONDecodeError
import math
import re
from typing import Protocol
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from assistant_ia.intelligence.model_client import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_CONTEXT_LENGTH,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_TIMEOUT,
)
from assistant_ia.memory.models import MemoryCandidate

MAX_MEMORY_CANDIDATES = 2
MAX_MEMORY_CANDIDATE_CONTENT_CHARACTERS = 500
MAX_MEMORY_CANDIDATE_SOURCE_CHARACTERS = 1000
MAX_ANALYZED_USER_MESSAGE_CHARACTERS = 4000
MIN_MEMORY_CANDIDATE_CONFIDENCE = 0.8

MEMORY_CANDIDATE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "maxItems": MAX_MEMORY_CANDIDATES,
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "minLength": 1},
                    "source_text": {"type": "string", "minLength": 1},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["content", "source_text", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

MEMORY_CANDIDATE_SYSTEM_PROMPT = """
Analyze only the current user message for possible persistent personal memories.
Return zero, one or at most two candidates using the required JSON schema.

A candidate may represent an explicit stable personal preference, practical
personal fact, durable choice, long-term project, durable constraint, explicit
correction, or information the user explicitly asks to retain.

Exclude questions, hypotheses, conditions that assert nothing, roleplay, jokes,
quotes or statements attributed to another person, temporary states, assistant
output, deductions, general behavioral instructions, lessons inferred from one
experience, social-profile structures, passwords, authentication secrets, API
tokens, 2FA codes and private keys.

content must be concise, faithful, use only information and words present in the
current user message, and introduce no inferred fact. source_text must be an
exact contiguous substring copied verbatim from the current user message. For
an explicit correction, source_text must include the complete correction
assertion verbatim, including its personal subject and correction marker, and
content must retain the complete corrected personal assertion, not only its
new value. confidence measures extraction fidelity and
admissibility, never whether the statement is true in the world.

Correction example:
Current user message: En fait, je prefere maintenant Fusion 360 pour la CAO.
Valid content: Je prefere maintenant Fusion 360 pour la CAO.
Valid source_text: En fait, je prefere maintenant Fusion 360 pour la CAO.
Invalid content or source_text: Fusion 360

Do not assign identifiers, persist anything, propose SQL, merge memories,
resolve contradictions, or use assistant messages as evidence.
""".strip()

_OLLAMA_CHAT_PATH = "/api/chat"
_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)

_INADMISSIBLE_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\?$",
        r"\b(?:si|if|maybe)\b",
        r"\bje pense que\b",
        r"\bpeut[ -]?etre\b",
        r"\b(?:imagine|fais comme si|roleplay|jeu de role)\b",
        r"\bmon ami\b.*\b(?:dit|prefere|aime)\b",
        r"\b(?:il|elle) (?:dit|prefere|aime)\b",
        r"\bquand je te demande\b",
        r"\b(?:reponds|tu dois) toujours\b",
    )
)

_TEMPORARY_PATTERN = re.compile(
    r"\b(?:aujourd'hui|ce soir|maintenant|cette fois|temporairement)\b",
    flags=re.IGNORECASE,
)
_CORRECTION_MARKER_PATTERN = re.compile(
    r"\b(?:en fait|je corrige|desormais)\b",
    flags=re.IGNORECASE,
)
_CORRECTION_EVIDENCE_PATTERN = re.compile(
    r"\b(?:en fait|je corrige|desormais|maintenant|plutot)\b",
    flags=re.IGNORECASE,
)
_PERSONAL_EVIDENCE_PATTERN = re.compile(
    r"\b(?:je|ma|mes|mon)\b",
    flags=re.IGNORECASE,
)

_SECRET_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\b(?:mot de passe|password|passwd)\b",
        r"\b(?:cle|key|token)\s*(?:d[' ]?api|api)\b",
        r"\b(?:code\s*)?(?:2fa|otp)\b",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\bsk-[A-Za-z0-9_-]{12,}\b",
        r"\bghp_[A-Za-z0-9]{12,}\b",
        r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
    )
)


class MemoryCandidateAnalysisError(RuntimeError):
    """Signaler que l'analyse ne peut pas produire un résultat sûr."""


class MemoryCandidateAnalyzer(Protocol):
    """Contrat d'analyse non persistante d'un message utilisateur autorisé."""

    def analyze(
        self,
        user_message: str,
    ) -> tuple[MemoryCandidate, ...]:
        """Retourner uniquement les candidats validés déterministement."""
        ...


class OllamaMemoryCandidateAnalyzer:
    """Extraire avec Ollama puis valider localement des candidats mémoire.

    ``confidence`` mesure ici la fidélité et l'admissibilité de l'extraction,
    jamais la vérité du fait dans le monde. L'analyse reste une proposition
    sans identifiant et sans effet de persistance.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
    ) -> None:
        """Créer l'analyseur local avec sa configuration Ollama dédiée."""
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Candidate analyzer model cannot be empty.")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Candidate analyzer base URL cannot be empty.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("Candidate analyzer timeout must be a number.")
        if timeout <= 0:
            raise ValueError("Candidate analyzer timeout must be positive.")

        self._model = model.strip()
        self._base_url = base_url.strip().rstrip("/")
        self._timeout = float(timeout)

    def analyze(self, user_message: str) -> tuple[MemoryCandidate, ...]:
        """Analyser un message autorisé sans jamais persister le résultat.

        Un message vide, trop long, secret ou manifestement inadmissible est
        rejeté avant l'appel réseau et retourne un tuple vide.
        """
        if not isinstance(user_message, str):
            raise TypeError("Candidate source message must be a string.")

        authorized_message = user_message.strip()

        if not authorized_message:
            return ()
        if len(authorized_message) > MAX_ANALYZED_USER_MESSAGE_CHARACTERS:
            return ()
        if _contains_secret(authorized_message):
            return ()
        if _is_inadmissible_statement(authorized_message):
            return ()

        # L'analyse utilise uniquement le message courant : ni historique ni
        # réponse assistant ne peuvent servir de preuve à un souvenir.
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": MEMORY_CANDIDATE_SYSTEM_PROMPT},
                {"role": "user", "content": authorized_message},
            ],
            "format": MEMORY_CANDIDATE_RESPONSE_SCHEMA,
            "stream": False,
            "think": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": 0,
                "num_ctx": DEFAULT_OLLAMA_CONTEXT_LENGTH,
            },
        }

        response_content = self._request_ollama(payload)
        return _parse_candidate_response(
            response_content,
            authorized_message=authorized_message,
        )

    def _request_ollama(self, payload: dict[str, object]) -> str:
        """Envoyer la requête structurée dédiée à l'Ollama local."""
        request = Request(
            url=f"{self._base_url}{_OLLAMA_CHAT_PATH}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except (
            HTTPError,
            URLError,
            OSError,
            TimeoutError,
            UnicodeError,
            JSONDecodeError,
        ) as error:
            raise MemoryCandidateAnalysisError(
                "Ollama candidate analysis failed."
            ) from error

        if not isinstance(response_data, dict):
            raise MemoryCandidateAnalysisError("Invalid Ollama response.")
        message = response_data.get("message")
        if not isinstance(message, dict) or not isinstance(
            message.get("content"), str
        ):
            raise MemoryCandidateAnalysisError("Invalid Ollama response.")

        return message["content"]


def _parse_candidate_response(
    response_content: str,
    *,
    authorized_message: str,
) -> tuple[MemoryCandidate, ...]:
    """Décoder le JSON strict et éliminer les entrées individuelles non sûres.

    Une enveloppe mal formée invalide toute l'analyse ; un candidat isolé qui
    échoue aux règles est simplement ignoré. Les doublons textuels sont
    supprimés en conservant le premier candidat accepté.
    """
    try:
        data = json.loads(response_content)
    except (TypeError, JSONDecodeError) as error:
        raise MemoryCandidateAnalysisError("Invalid candidate JSON.") from error

    if not isinstance(data, dict) or set(data) != {"candidates"}:
        raise MemoryCandidateAnalysisError("Invalid candidate schema.")

    candidate_data = data["candidates"]
    if not isinstance(candidate_data, list):
        raise MemoryCandidateAnalysisError("Invalid candidate schema.")
    if len(candidate_data) > MAX_MEMORY_CANDIDATES:
        raise MemoryCandidateAnalysisError("Too many memory candidates.")

    accepted: list[MemoryCandidate] = []
    seen_contents: set[str] = set()

    for item in candidate_data:
        if not isinstance(item, dict) or set(item) != {
            "content", "source_text", "confidence"
        }:
            raise MemoryCandidateAnalysisError("Invalid candidate fields.")

        candidate = _validated_candidate(
            item,
            authorized_message=authorized_message,
        )
        if candidate is None:
            continue

        duplicate_key = candidate.content.casefold()
        if duplicate_key in seen_contents:
            continue
        seen_contents.add(duplicate_key)
        accepted.append(candidate)

    return tuple(accepted)


def _validated_candidate(
    data: dict[str, object],
    *,
    authorized_message: str,
) -> MemoryCandidate | None:
    """Retourner un candidat ancré et admissible, sinon ``None``.

    Les contrôles portent notamment sur les bornes, la fidélité lexicale, la
    présence exacte de ``source_text`` et l'absence de secrets. Ils compensent
    le caractère probabiliste de la sortie Ollama.
    """
    content = data.get("content")
    source_text = data.get("source_text")
    confidence = data.get("confidence")

    if not isinstance(content, str) or not isinstance(source_text, str):
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    if not math.isfinite(float(confidence)):
        return None
    if not 0.0 <= float(confidence) <= 1.0:
        return None
    if float(confidence) < MIN_MEMORY_CANDIDATE_CONFIDENCE:
        return None
    if not content.strip():
        return None
    if len(content.strip()) > MAX_MEMORY_CANDIDATE_CONTENT_CHARACTERS:
        return None
    if len(source_text) > MAX_MEMORY_CANDIDATE_SOURCE_CHARACTERS:
        return None
    if not source_text.strip() or source_text not in authorized_message:
        return None
    if _is_explicit_personal_correction(authorized_message):
        if not _has_explicit_correction_evidence(source_text):
            return None
        if not _has_sufficient_correction_coverage(
            content,
            authorized_message,
        ):
            return None
        if not _has_sufficient_correction_coverage(
            source_text,
            authorized_message,
        ):
            return None
    if _contains_secret(content) or _contains_secret(source_text):
        return None
    if (
        _is_inadmissible_statement(source_text)
        and not _is_explicit_personal_correction(authorized_message)
    ):
        return None
    if not _content_is_lexically_grounded(content, authorized_message):
        return None

    try:
        return MemoryCandidate(
            content=content,
            source_text=source_text,
            confidence=float(confidence),
        )
    except (TypeError, ValueError):
        return None


def _normalized_text(value: str) -> str:
    """Normaliser casse et accents pour les comparaisons lexicales internes."""
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def _content_is_lexically_grounded(content: str, message: str) -> bool:
    """Vérifier que chaque terme du candidat apparaît dans le message source."""
    content_terms = set(_WORD_PATTERN.findall(_normalized_text(content)))
    message_terms = set(_WORD_PATTERN.findall(_normalized_text(message)))
    return bool(content_terms) and content_terms.issubset(message_terms)


def _is_inadmissible_statement(value: str) -> bool:
    """Détecter une question, hypothèse, instruction ou information temporaire."""
    normalized = _normalized_text(value).strip()
    if any(pattern.search(normalized) for pattern in _INADMISSIBLE_PATTERNS):
        return True
    if not _TEMPORARY_PATTERN.search(normalized):
        return False
    return not _is_explicit_personal_correction(value)


def _is_explicit_personal_correction(value: str) -> bool:
    """Reconnaître une correction explicitement personnelle dans le texte."""
    normalized = _normalized_text(value)
    return bool(
        _CORRECTION_MARKER_PATTERN.search(normalized)
        and _PERSONAL_EVIDENCE_PATTERN.search(normalized)
    )


def _has_sufficient_correction_coverage(
    content: str,
    authorized_message: str,
) -> bool:
    """Exiger qu'une correction conserve une part suffisante du message."""
    content_terms = set(_WORD_PATTERN.findall(_normalized_text(content)))
    message_terms = set(
        _WORD_PATTERN.findall(_normalized_text(authorized_message))
    )
    if not message_terms:
        return False
    return len(content_terms & message_terms) / len(message_terms) >= 0.5


def _has_explicit_correction_evidence(value: str) -> bool:
    """Vérifier la présence conjointe d'un marqueur et d'un sujet personnel."""
    normalized = _normalized_text(value)
    return bool(
        _CORRECTION_EVIDENCE_PATTERN.search(normalized)
        and _PERSONAL_EVIDENCE_PATTERN.search(normalized)
    )


def _contains_secret(value: str) -> bool:
    """Détecter les formes de secrets explicitement interdites à la mémoire."""
    normalized = _normalized_text(value)
    return any(pattern.search(normalized) for pattern in _SECRET_PATTERNS)
