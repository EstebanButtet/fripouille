"""Structured language model response definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Represent a validated response produced by a language model."""

    content: str
    model: str

    def __post_init__(self) -> None:
        """Validate and normalize the response fields."""
        if not isinstance(self.content, str):
            raise TypeError("Model response content must be a string.")

        if not isinstance(self.model, str):
            raise TypeError("Model name must be a string.")

        normalized_content = self.content.strip()
        normalized_model = self.model.strip()

        if not normalized_content:
            raise ValueError("Model response content cannot be empty.")

        if not normalized_model:
            raise ValueError("Model name cannot be empty.")

        object.__setattr__(self, "content", normalized_content)
        object.__setattr__(self, "model", normalized_model)
