"""Language model client abstractions and Ollama implementation."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from assistant_ia.core.context import ConversationMessage
from assistant_ia.intelligence.response import ModelResponse

DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 120.0

_OLLAMA_CHAT_PATH = "/api/chat"


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
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
            "stream": False,
            "think": False,
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

        try:
            return ModelResponse(
                content=message_data.get("content"),
                model=response_data.get("model"),
            )
        except (TypeError, ValueError) as error:
            raise ModelClientError(
                "Ollama returned an invalid model response."
            ) from error
