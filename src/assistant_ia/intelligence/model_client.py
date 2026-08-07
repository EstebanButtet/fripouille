"""Language model client abstractions and Ollama implementation."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from assistant_ia.core.context import ConversationMessage
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

Allowed intentions:
- conversation: normal discussion, explanation or information request.
- unknown: unsupported or genuinely ambiguous action request.
- create_task: create or schedule a task.
- list_tasks: list existing tasks.
- complete_task: mark a task as completed.
- save_memory: remember information for later.
- find_memory: search previously saved information.
- delete_memory: delete previously saved information.
- write_journal: add information to a journal.
- launch_application: open a computer application.

Use conversation for ordinary dialogue.
Use unknown only for an action request that cannot be mapped reliably.
Never invent another intention name.

No action execution capability is currently available.
For an action intention, explain that the request was identified but not
executed. Never claim that a task, memory, journal entry or application was
actually created, changed, saved, deleted or launched.

Extract only simple parameters explicitly supported by the user's request.
Every parameter name and value must be a non-empty string.
Do not invent missing dates, titles, application names or other details.
Use an empty parameters object when no parameter is needed.

Required JSON schema:
{json.dumps(_INTENT_RESPONSE_SCHEMA, ensure_ascii=False)}
""".strip()


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
    ) -> None:
        """Create an Ollama client with explicit local configuration."""
        if not isinstance(model, str):
            raise TypeError("Model name must be a string.")

        if not isinstance(base_url, str):
            raise TypeError("Ollama base URL must be a string.")

        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("Ollama timeout must be a number.")

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
                    "content": INTENT_SYSTEM_PROMPT,
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
