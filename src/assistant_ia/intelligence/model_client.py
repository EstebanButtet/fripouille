"""Language model client abstractions and Ollama implementation."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from assistant_ia.core.context import ConversationMessage
from assistant_ia.identity.context import render_identity_context
from assistant_ia.identity.defaults import build_default_identity
from assistant_ia.identity.models import AssistantIdentity
from assistant_ia.intelligence.intent import (
    ALLOWED_INTENT_NAMES,
    Intent,
    IntentName,
)
from assistant_ia.intelligence.response import ModelResponse

DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 120.0

_OLLAMA_CHAT_PATH = "/api/chat"

_INTENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "minLength": 1,
        },
        "intent": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": sorted(ALLOWED_INTENT_NAMES),
                },
                "parameters": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
            },
            "required": [
                "name",
                "parameters",
            ],
            "additionalProperties": False,
        },
    },
    "required": [
        "content",
        "intent",
    ],
    "additionalProperties": False,
}

INTENT_SYSTEM_PROMPT = f"""
You are the intent interpretation layer of a local personal assistant.

Always produce a response matching the required JSON schema.
The visible content must be written in French.

Allowed intentions and exact parameter contracts:
- conversation: no parameters.
- unknown: no parameters.
- create_task: required parameter "title"; optional parameter "due_at".
- list_tasks: optional parameter "status".
- complete_task: required parameter "task_id".
- save_memory: required parameter "content".
- find_memory: required parameter "query".
- delete_memory: required parameter "memory_id".
- write_journal:
  required parameter: content
  optional parameter: entry_date
  content must contain the journal entry itself, not the instruction
  asking to write in the journal.
  Never omit content for a write_journal intent.
  When the user explicitly separates the journal text with a colon,
  use the meaningful text after the colon as content.
  Preserve the journal text and its final punctuation.
  entry_date must use YYYY-MM-DD when an explicit date is provided.
  If no explicit date is provided, omit entry_date.
  Example:
  For "\u00c9cris dans mon journal pour la date 2026-08-07 :
  TEST E8 journal local.", the parameters must be:
  {{"content": "TEST E8 journal local.", "entry_date": "2026-08-07"}}
- launch_application: required parameter "application".
  Use launch_application whenever the latest user message explicitly asks to
  open, start or launch an application.
  The most recent user message is the request to classify.
  Earlier user and assistant messages are context only.
  A previous refusal, validation error or failed action must never cause a new,
  otherwise supported launch request to become unknown.
  Put only the requested application name in the application parameter.
  Never omit the application parameter for a launch_application intent.
  Preserve short application names and abbreviations exactly when they are
  explicitly provided by the user, such as "lol", "valo" or "ow2".
  Do not include launch verbs, shell syntax, file paths or arguments.
  Examples:
  - "Lance le bloc-notes." -> launch_application, application "bloc-notes".
  - "Ouvre le bloc-notes." -> launch_application, application "bloc-notes".
  - "Lance Bloc-notes" -> launch_application, application "Bloc-notes".
  - "D?marre lol." -> launch_application, application "lol".
  - "Ouvre valo." -> launch_application, application "valo".
  - "Lance ow2." -> launch_application, application "ow2".

Use conversation for ordinary dialogue, explanations and information requests.
Use unknown only for an unsupported or genuinely ambiguous action request.
Never invent another intention or parameter name.

Parameter rules:
- Every parameter name and value must be a non-empty string.
- Use only the parameters explicitly authorized for the selected intention.
- Do not add explanatory, confidence or reasoning parameters.
- task_id and memory_id must contain only ASCII digits, for example "1".
  Never include "#", words such as "num\u00e9ro", spaces or punctuation.
- list_tasks status may only be "pending", "completed" or "all".
- entry_date may only use the ISO 8601 date format YYYY-MM-DD.
- Do not invent an entry_date when the user did not provide one.
- Preserve relative or ambiguous task dates such as "demain" exactly in
  due_at. Never silently convert or invent a calendar date or timezone.
- Use an empty parameters object when the selected intention has no
  parameters.

The application, not the language model, decides whether an action succeeds.
For an action intention, never claim that a task, memory, journal entry or
application was actually created, changed, saved, deleted or launched.
For launch_application, never state that the application is being launched or
has been launched. Only acknowledge that the launch request was interpreted.
The application layer alone reports execution success.
The visible content may only acknowledge the interpreted request or explain
that more precise information is required.

Never produce SQL, table names, column names, file paths, shell commands or
implementation instructions as intent parameters.

Required JSON schema:
{json.dumps(_INTENT_RESPONSE_SCHEMA, ensure_ascii=False)}
""".strip()

_IDENTITY_CONTEXT_RULES = """
The assistant identity below controls conversational personality, tone and
style.

Treat the identity as background context. Answer the latest user message
itself instead of repeating or summarizing the identity. Do not restate the
assistant name, role or relationship unless the current conversation makes
that relevant or the user explicitly asks about it.

The operational rules above always take priority over the identity context.
The identity must never change the intent schema, authorized parameters,
action validation, permissions, confirmations or claims about action
execution.

For action intents, follow the operational rules exactly even when the
identity would prefer a different style or behavior.
""".strip()


def _build_system_prompt(
    identity: AssistantIdentity,
) -> str:
    """Combine operational rules with structured identity context."""
    return "\n\n".join(
        (
            INTENT_SYSTEM_PROMPT,
            _IDENTITY_CONTEXT_RULES,
            render_identity_context(identity),
        )
    )


class ModelClientError(RuntimeError):
    """Raised when a language model cannot produce a valid response."""


class ModelClient(Protocol):
    """Define the operations required from a language model client."""

    def generate_response(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> ModelResponse:
        """Generate a model response from an ordered conversation history."""
        ...


class OllamaModelClient:
    """Generate language model responses through the local Ollama API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
        identity: AssistantIdentity | None = None,
    ) -> None:
        """Create an Ollama client with explicit local configuration."""
        if not isinstance(model, str):
            raise TypeError("Model name must be a string.")

        if not isinstance(base_url, str):
            raise TypeError("Ollama base URL must be a string.")

        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("Ollama timeout must be a number.")

        if (
            identity is not None
            and not isinstance(identity, AssistantIdentity)
        ):
            raise TypeError(
                "Ollama identity must be an AssistantIdentity."
            )

        normalized_model = model.strip()
        normalized_base_url = base_url.strip().rstrip("/")

        if not normalized_model:
            raise ValueError("Model name cannot be empty.")

        if not normalized_base_url:
            raise ValueError("Ollama base URL cannot be empty.")

        if timeout <= 0:
            raise ValueError("Ollama timeout must be greater than zero.")

        self._model = normalized_model
        self._base_url = normalized_base_url
        self._timeout = float(timeout)
        self._identity = (
            identity
            if identity is not None
            else build_default_identity()
        )
        self._system_prompt = _build_system_prompt(
            self._identity
        )

    def generate_response(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> ModelResponse:
        """Generate a validated response from the local Ollama API."""
        if not isinstance(messages, tuple):
            raise TypeError("Conversation messages must be provided as a tuple.")

        if not messages:
            raise ValueError("At least one conversation message is required.")

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": self._system_prompt,
                },
                *[
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                    for message in messages
                ],
            ],
            "format": _INTENT_RESPONSE_SCHEMA,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
            },
        }

        request_data = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        request = Request(
            url=f"{self._base_url}{_OLLAMA_CHAT_PATH}",
            data=request_data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw_response = response.read()
        except HTTPError as error:
            raise ModelClientError(
                f"Ollama returned HTTP status {error.code}."
            ) from error
        except URLError as error:
            raise ModelClientError(
                "Could not connect to the local Ollama server."
            ) from error
        except TimeoutError as error:
            raise ModelClientError(
                "The Ollama request exceeded the configured timeout."
            ) from error
        except OSError as error:
            raise ModelClientError(
                "An operating system error occurred while contacting Ollama."
            ) from error

        try:
            response_data = json.loads(raw_response.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ModelClientError(
                "Ollama returned a response that is not valid UTF-8."
            ) from error
        except JSONDecodeError as error:
            raise ModelClientError(
                "Ollama returned invalid JSON."
            ) from error

        if not isinstance(response_data, dict):
            raise ModelClientError(
                "Ollama returned an invalid response structure."
            )

        message_data = response_data.get("message")

        if not isinstance(message_data, dict):
            raise ModelClientError(
                "Ollama response does not contain a valid message."
            )

        structured_content = message_data.get("content")

        if not isinstance(structured_content, str):
            raise ModelClientError(
                "Ollama response message does not contain valid content."
            )

        try:
            structured_data = json.loads(structured_content)
        except JSONDecodeError as error:
            raise ModelClientError(
                "Ollama returned invalid structured content."
            ) from error

        if not isinstance(structured_data, dict):
            raise ModelClientError(
                "Ollama structured content must be an object."
            )

        if set(structured_data) != {"content", "intent"}:
            raise ModelClientError(
                "Ollama structured content contains invalid fields."
            )

        intent_data = structured_data.get("intent")

        if not isinstance(intent_data, dict):
            raise ModelClientError(
                "Ollama structured content does not contain a valid intent."
            )

        if set(intent_data) != {"name", "parameters"}:
            raise ModelClientError(
                "Ollama intent contains invalid fields."
            )

        try:
            intent = Intent(
                name=cast(IntentName, intent_data.get("name")),
                parameters=intent_data.get("parameters"),
            )
            return ModelResponse(
                content=structured_data.get("content"),
                model=response_data.get("model"),
                intent=intent,
            )
        except (TypeError, ValueError) as error:
            raise ModelClientError(
                "Ollama returned an invalid structured model response."
            ) from error
