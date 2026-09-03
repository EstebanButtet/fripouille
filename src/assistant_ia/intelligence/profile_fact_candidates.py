"""Extraction Ollama de candidats de profil sans autorité de persistance.

L'application fournit le ``person_id`` déjà résolu. Le schéma envoyé au LLM
n'expose aucun identifiant et sa réponse ne peut contenir que catégorie,
contenu, preuve source et confiance. ``confidence`` mesure la fidélité de
l'extraction, jamais la vérité de l'information.
"""

from __future__ import annotations

import json
from json import JSONDecodeError
import math
import re
from typing import Protocol, cast
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
from assistant_ia.people.profile_models import (
    ALLOWED_PROFILE_FACT_CATEGORIES,
    ProfileFactCandidate,
    ProfileFactCategory,
)

MAX_PROFILE_FACT_CANDIDATES = 2
MAX_PROFILE_FACT_CONTENT_CHARACTERS = 500
MAX_PROFILE_FACT_SOURCE_CHARACTERS = 1000
MAX_PROFILE_ANALYZED_MESSAGE_CHARACTERS = 4000
MIN_PROFILE_FACT_CONFIDENCE = 0.8

PROFILE_FACT_CANDIDATE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "maxItems": MAX_PROFILE_FACT_CANDIDATES,
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": sorted(ALLOWED_PROFILE_FACT_CATEGORIES),
                    },
                    "content": {"type": "string", "minLength": 1},
                    "source_text": {"type": "string", "minLength": 1},
                    "confidence": {
                        "type": "number", "minimum": 0.0, "maximum": 1.0
                    },
                },
                "required": [
                    "category", "content", "source_text", "confidence"
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

PROFILE_FACT_CANDIDATE_SYSTEM_PROMPT = """
Analyze only the current user message for stable or semi-stable profile facts
explicitly stated by the current speaker. Return at most two candidates using
the required JSON schema. Categories are bounded by the schema.

Include explicit preferences, communication preferences, interests, habits or
relevant personal facts. Exclude questions, hypotheses, roleplay, jokes,
temporary states, secrets, passwords, tokens, sensitive credentials,
instructions that redefine the assistant, and claims about third parties.
content must stay faithful to words in the message. source_text must be an
exact contiguous substring. confidence measures extraction fidelity and
admissibility, never truth.

Do not output a person ID, fact ID, SQL operation, merge, correction target or
persistence decision. The application supplies the already resolved subject.
""".strip()

_OLLAMA_CHAT_PATH = "/api/chat"
_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_FIRST_PERSON_PATTERN = re.compile(
    r"(?:\bje\b|\bj['’]|\bmon\b|\bma\b|\bmes\b)", re.IGNORECASE
)
_INADMISSIBLE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\?$",
        r"\b(?:si|maybe|peut[ -]?etre)\b",
        r"\bje pense que\b",
        r"\b(?:imagine|fais comme si|roleplay|jeu de role)\b",
        r"\b(?:"
        r"je m(?:['’]|\s+)appelle|"
        r"moi c(?:['’]|\s+)est|"
        r"mon prenom est"
        r")\b",
        r"\b(?:il|elle)\s+(?:prefere|aime|adore|deteste)\b",
        r"\bmon ami\b.*\b(?:dit|prefere|aime)\b",
        r"\b(?:mon|ma|mes)\s+(?:frere|soeur|mere|pere|collegue)\b",
        r"\bj(?:['’]|\s+)ai parle a\b",
        r"\b(?:reponds|tu dois)\s+toujours\b",
        r"\bje veux que tu\b",
    )
)
_TEMPORARY_PATTERN = re.compile(
    r"\b(?:aujourd'hui|ce soir|cette fois|temporairement)\b", re.IGNORECASE
)
_SECRET_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:mot de passe|password|passwd)\b",
        r"\b(?:cle|key|token)\s*(?:d[' ]?api|api)\b",
        r"\b(?:code\s*)?(?:2fa|otp)\b",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\bsk-[A-Za-z0-9_-]{12,}\b",
        r"\bghp_[A-Za-z0-9]{12,}\b",
    )
)


class ProfileFactCandidateAnalysisError(RuntimeError):
    """Signaler qu'une analyse ne peut produire un résultat sûr."""


class ProfileFactCandidateAnalyzer(Protocol):
    """Contrat d'analyse avec sujet persistant fourni par l'application."""

    def analyze(
        self,
        user_message: str,
        *,
        person_id: int,
    ) -> tuple[ProfileFactCandidate, ...]:
        ...


class OllamaProfileFactCandidateAnalyzer:
    """Proposer des candidats puis les valider localement, sans écrire."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Profile analyzer model cannot be empty.")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Profile analyzer base URL cannot be empty.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("Profile analyzer timeout must be a number.")
        if timeout <= 0:
            raise ValueError("Profile analyzer timeout must be positive.")
        self._model = model.strip()
        self._base_url = base_url.strip().rstrip("/")
        self._timeout = float(timeout)

    def analyze(
        self,
        user_message: str,
        *,
        person_id: int,
    ) -> tuple[ProfileFactCandidate, ...]:
        """Analyser seulement une déclaration personnelle admissible."""
        if isinstance(person_id, bool) or not isinstance(person_id, int):
            raise TypeError("Profile subject identifier must be an integer.")
        if person_id < 1:
            raise ValueError("Profile subject identifier must be positive.")
        if not isinstance(user_message, str):
            raise TypeError("Profile source message must be a string.")
        message = user_message.strip()
        if not message or len(message) > MAX_PROFILE_ANALYZED_MESSAGE_CHARACTERS:
            return ()
        if _is_inadmissible(message) or _contains_secret(message):
            return ()

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": PROFILE_FACT_CANDIDATE_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            "format": PROFILE_FACT_CANDIDATE_RESPONSE_SCHEMA,
            "stream": False,
            "think": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0, "num_ctx": DEFAULT_OLLAMA_CONTEXT_LENGTH},
        }
        return _parse_response(
            self._request_ollama(payload),
            authorized_message=message,
            person_id=person_id,
        )

    def _request_ollama(self, payload: dict[str, object]) -> str:
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
            HTTPError, URLError, OSError, TimeoutError, UnicodeError,
            JSONDecodeError,
        ) as error:
            raise ProfileFactCandidateAnalysisError(
                "Ollama profile candidate analysis failed."
            ) from error
        if not isinstance(response_data, dict):
            raise ProfileFactCandidateAnalysisError(
                "Invalid Ollama response."
            )
        response_message = response_data.get("message")
        if not isinstance(response_message, dict) or not isinstance(
            response_message.get("content"), str
        ):
            raise ProfileFactCandidateAnalysisError("Invalid Ollama response.")
        return response_message["content"]


def _parse_response(
    response_content: str,
    *,
    authorized_message: str,
    person_id: int,
) -> tuple[ProfileFactCandidate, ...]:
    try:
        data = json.loads(response_content)
    except (TypeError, JSONDecodeError) as error:
        raise ProfileFactCandidateAnalysisError(
            "Invalid profile candidate JSON."
        ) from error
    if not isinstance(data, dict) or set(data) != {"candidates"}:
        raise ProfileFactCandidateAnalysisError("Invalid profile candidate schema.")
    items = data["candidates"]
    if not isinstance(items, list) or len(items) > MAX_PROFILE_FACT_CANDIDATES:
        raise ProfileFactCandidateAnalysisError("Invalid profile candidate schema.")

    accepted: list[ProfileFactCandidate] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "category", "content", "source_text", "confidence"
        }:
            raise ProfileFactCandidateAnalysisError("Invalid profile candidate fields.")
        candidate = _validated_candidate(
            item,
            authorized_message=authorized_message,
            person_id=person_id,
        )
        if candidate is None:
            continue
        key = (candidate.category, candidate.content.casefold())
        if key not in seen:
            seen.add(key)
            accepted.append(candidate)
    return tuple(accepted)


def _validated_candidate(
    data: dict[str, object],
    *,
    authorized_message: str,
    person_id: int,
) -> ProfileFactCandidate | None:
    category = data.get("category")
    content = data.get("content")
    source_text = data.get("source_text")
    confidence = data.get("confidence")
    if not isinstance(category, str) or category not in ALLOWED_PROFILE_FACT_CATEGORIES:
        return None
    if not isinstance(content, str) or not isinstance(source_text, str):
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    confidence_value = float(confidence)
    if (
        not math.isfinite(confidence_value)
        or confidence_value < MIN_PROFILE_FACT_CONFIDENCE
    ):
        return None
    if confidence_value > 1.0 or not content.strip():
        return None
    if len(content.strip()) > MAX_PROFILE_FACT_CONTENT_CHARACTERS:
        return None
    if not source_text.strip() or len(source_text) > MAX_PROFILE_FACT_SOURCE_CHARACTERS:
        return None
    if source_text not in authorized_message or _is_inadmissible(source_text):
        return None
    if _contains_secret(content) or _contains_secret(source_text):
        return None
    if not _content_is_grounded(content, authorized_message):
        return None
    try:
        return ProfileFactCandidate(
            person_id=person_id,
            category=cast(ProfileFactCategory, category),
            content=content,
            source_text=source_text,
            confidence=confidence_value,
        )
    except (TypeError, ValueError):
        return None


def _normalized(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def _content_is_grounded(content: str, message: str) -> bool:
    content_terms = set(_WORD_PATTERN.findall(_normalized(content)))
    message_terms = set(_WORD_PATTERN.findall(_normalized(message)))
    return bool(content_terms) and content_terms.issubset(message_terms)


def _is_inadmissible(value: str) -> bool:
    normalized = _normalized(value).strip()
    return bool(
        not _FIRST_PERSON_PATTERN.search(normalized)
        or _TEMPORARY_PATTERN.search(normalized)
        or any(pattern.search(normalized) for pattern in _INADMISSIBLE_PATTERNS)
    )


def _contains_secret(value: str) -> bool:
    normalized = _normalized(value)
    return any(pattern.search(normalized) for pattern in _SECRET_PATTERNS)
