"""Deterministic validation for fixed-total allocations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from json import JSONDecodeError
import re


ALLOCATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "total": {
            "type": "string",
            "minLength": 1,
        },
        "unit": {
            "type": "string",
            "minLength": 1,
        },
        "parts": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "amount": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
                "required": [
                    "label",
                    "amount",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "total",
        "unit",
        "parts",
    ],
    "additionalProperties": False,
}


ALLOCATION_PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "parts": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "amount": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                    },
                },
                "required": [
                    "label",
                    "amount",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "parts",
    ],
    "additionalProperties": False,
}


class AllocationTargetParseError(ValueError):
    """Raised when user-supplied target text is not exactly parseable."""


class AllocationFormatError(ValueError):
    """Raised when a structured allocation payload is malformed."""


class AllocationMismatchError(ValueError):
    """Raised when allocation parts do not match the required total."""


@dataclass(frozen=True, slots=True)
class AllocationTarget:
    """Define a fixed total owned by the application."""

    total: Decimal
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.total, Decimal):
            raise TypeError(
                "Allocation target total must be a Decimal."
            )

        if not isinstance(self.unit, str):
            raise TypeError(
                "Allocation target unit must be a string."
            )

        if not self.total.is_finite():
            raise ValueError(
                "Allocation target total must be finite."
            )

        if self.total <= Decimal("0"):
            raise ValueError(
                "Allocation target total must be greater than zero."
            )

        normalized_unit = self.unit.strip()

        if not normalized_unit:
            raise ValueError(
                "Allocation target unit cannot be empty."
            )

        object.__setattr__(
            self,
            "unit",
            normalized_unit,
        )


@dataclass(frozen=True, slots=True)
class AllocationPart:
    """Represent one exact portion of a fixed-total allocation."""

    label: str
    amount: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise TypeError(
                "Allocation part label must be a string."
            )

        if not isinstance(self.amount, Decimal):
            raise TypeError(
                "Allocation part amount must be a Decimal."
            )

        normalized_label = self.label.strip()

        if not normalized_label:
            raise ValueError(
                "Allocation part label cannot be empty."
            )

        if self.amount <= Decimal("0"):
            raise ValueError(
                "Allocation part amount must be greater than zero."
            )

        object.__setattr__(
            self,
            "label",
            normalized_label,
        )


@dataclass(frozen=True, slots=True)
class FixedTotalAllocation:
    """Represent and validate an exact allocation against a fixed total."""

    total: Decimal
    unit: str
    parts: tuple[AllocationPart, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.total, Decimal):
            raise TypeError(
                "Allocation total must be a Decimal."
            )

        if not isinstance(self.unit, str):
            raise TypeError(
                "Allocation unit must be a string."
            )

        if not isinstance(self.parts, tuple):
            raise TypeError(
                "Allocation parts must be a tuple."
            )

        if self.total <= Decimal("0"):
            raise ValueError(
                "Allocation total must be greater than zero."
            )

        normalized_unit = self.unit.strip()

        if not normalized_unit:
            raise ValueError(
                "Allocation unit cannot be empty."
            )

        if not self.parts:
            raise ValueError(
                "Allocation must contain at least one part."
            )

        for part in self.parts:
            if not isinstance(part, AllocationPart):
                raise TypeError(
                    "Allocation parts must contain AllocationPart "
                    "instances only."
                )

        object.__setattr__(
            self,
            "unit",
            normalized_unit,
        )

    @property
    def allocated_total(self) -> Decimal:
        """Return the exact sum of every allocation part."""
        return sum(
            (
                part.amount
                for part in self.parts
            ),
            start=Decimal("0"),
        )

    @property
    def difference(self) -> Decimal:
        """Return required total minus allocated total."""
        return self.total - self.allocated_total

    @property
    def is_exact(self) -> bool:
        """Return whether the allocation matches the total exactly."""
        return self.difference == Decimal("0")

    def require_exact(self) -> None:
        """Raise when the allocation does not match its required total."""
        if self.is_exact:
            return

        raise AllocationMismatchError(
            "Allocation does not match required total: "
            f"required={self.total} {self.unit}, "
            f"allocated={self.allocated_total} {self.unit}, "
            f"difference={self.difference} {self.unit}."
        )

def _parse_decimal(
    value: object,
    *,
    field_name: str,
) -> Decimal:
    """Parse one protocol decimal from its required string form."""
    if not isinstance(value, str):
        raise AllocationFormatError(
            f"{field_name} must be a string."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise AllocationFormatError(
            f"{field_name} cannot be empty."
        )

    try:
        parsed_value = Decimal(
            normalized_value
        )
    except InvalidOperation as error:
        raise AllocationFormatError(
            f"{field_name} must contain a valid decimal number."
        ) from error

    if not parsed_value.is_finite():
        raise AllocationFormatError(
            f"{field_name} must contain a finite decimal number."
        )

    return parsed_value


def parse_allocation_payload(
    payload: object,
) -> FixedTotalAllocation:
    """Convert one strict structured payload into an allocation."""
    if not isinstance(payload, dict):
        raise AllocationFormatError(
            "Allocation payload must be an object."
        )

    required_fields = {
        "total",
        "unit",
        "parts",
    }

    if set(payload) != required_fields:
        raise AllocationFormatError(
            "Allocation payload must contain exactly "
            "total, unit and parts."
        )

    unit = payload["unit"]

    if not isinstance(unit, str):
        raise AllocationFormatError(
            "Allocation unit must be a string."
        )

    normalized_unit = unit.strip()

    if not normalized_unit:
        raise AllocationFormatError(
            "Allocation unit cannot be empty."
        )

    raw_parts = payload["parts"]

    if not isinstance(raw_parts, list):
        raise AllocationFormatError(
            "Allocation parts must be an array."
        )

    if not raw_parts:
        raise AllocationFormatError(
            "Allocation parts cannot be empty."
        )

    parsed_parts: list[AllocationPart] = []

    for index, raw_part in enumerate(
        raw_parts
    ):
        if not isinstance(raw_part, dict):
            raise AllocationFormatError(
                "Each allocation part must be an object."
            )

        if set(raw_part) != {
            "label",
            "amount",
        }:
            raise AllocationFormatError(
                "Each allocation part must contain exactly "
                "label and amount."
            )

        label = raw_part["label"]

        if not isinstance(label, str):
            raise AllocationFormatError(
                "Allocation part label must be a string."
            )

        normalized_label = label.strip()

        if not normalized_label:
            raise AllocationFormatError(
                "Allocation part label cannot be empty."
            )

        amount = _parse_decimal(
            raw_part["amount"],
            field_name=(
                f"parts[{index}].amount"
            ),
        )

        try:
            part = AllocationPart(
                label=normalized_label,
                amount=amount,
            )
        except (TypeError, ValueError) as error:
            raise AllocationFormatError(
                f"Invalid allocation part at index {index}."
            ) from error

        parsed_parts.append(
            part
        )

    total = _parse_decimal(
        payload["total"],
        field_name="total",
    )

    try:
        return FixedTotalAllocation(
            total=total,
            unit=normalized_unit,
            parts=tuple(parsed_parts),
        )
    except (TypeError, ValueError) as error:
        raise AllocationFormatError(
            "Structured allocation values are invalid."
        ) from error

def _reject_json_constant(
    value: str,
) -> object:
    """Reject non-standard JSON numeric constants."""
    raise AllocationFormatError(
        f"Invalid JSON numeric constant: {value}."
    )


def parse_allocation_proposal_json(
    content: str,
    *,
    target: AllocationTarget,
) -> FixedTotalAllocation:
    """Parse model-proposed parts against an authoritative target."""
    if not isinstance(content, str):
        raise TypeError(
            "Allocation proposal content must be a string."
        )

    if not isinstance(target, AllocationTarget):
        raise TypeError(
            "Allocation proposal target must be an AllocationTarget."
        )

    try:
        payload = json.loads(
            content,
            parse_int=Decimal,
            parse_float=Decimal,
            parse_constant=_reject_json_constant,
        )
    except JSONDecodeError as error:
        raise AllocationFormatError(
            "Allocation proposal must contain valid JSON."
        ) from error

    if not isinstance(payload, dict):
        raise AllocationFormatError(
            "Allocation proposal must be an object."
        )

    if set(payload) != {
        "parts",
    }:
        raise AllocationFormatError(
            "Allocation proposal must contain exactly parts."
        )

    raw_parts = payload["parts"]

    if not isinstance(raw_parts, list):
        raise AllocationFormatError(
            "Allocation proposal parts must be an array."
        )

    if not raw_parts:
        raise AllocationFormatError(
            "Allocation proposal parts cannot be empty."
        )

    parsed_parts: list[AllocationPart] = []

    for index, raw_part in enumerate(
        raw_parts
    ):
        if not isinstance(raw_part, dict):
            raise AllocationFormatError(
                "Each allocation proposal part must be an object."
            )

        if set(raw_part) != {
            "label",
            "amount",
        }:
            raise AllocationFormatError(
                "Each allocation proposal part must contain exactly "
                "label and amount."
            )

        label = raw_part["label"]
        amount = raw_part["amount"]

        if not isinstance(label, str):
            raise AllocationFormatError(
                "Allocation proposal part label must be a string."
            )

        if not isinstance(amount, Decimal):
            raise AllocationFormatError(
                "Allocation proposal part amount must be a JSON number."
            )

        try:
            part = AllocationPart(
                label=label,
                amount=amount,
            )
        except (TypeError, ValueError) as error:
            raise AllocationFormatError(
                f"Invalid allocation proposal part at index {index}."
            ) from error

        parsed_parts.append(
            part
        )

    return FixedTotalAllocation(
        total=target.total,
        unit=target.unit,
        parts=tuple(parsed_parts),
    )

_FRENCH_HOUR_WORDS = {
    "un": Decimal("1"),
    "une": Decimal("1"),
    "deux": Decimal("2"),
    "trois": Decimal("3"),
    "quatre": Decimal("4"),
    "cinq": Decimal("5"),
    "six": Decimal("6"),
    "sept": Decimal("7"),
    "huit": Decimal("8"),
    "neuf": Decimal("9"),
    "dix": Decimal("10"),
    "onze": Decimal("11"),
    "douze": Decimal("12"),
}


def _duration_target_from_minutes(
    minutes: Decimal,
) -> AllocationTarget:
    """Build one positive finite duration target in minutes."""
    if not minutes.is_finite():
        raise AllocationTargetParseError(
            "Duration target must be finite."
        )

    if minutes <= Decimal("0"):
        raise AllocationTargetParseError(
            "Duration target must be greater than zero."
        )

    return AllocationTarget(
        total=minutes,
        unit="minutes",
    )


def parse_duration_target_text(
    text: str,
) -> AllocationTarget:
    """Parse one exact standalone duration fragment deterministically."""
    if not isinstance(text, str):
        raise TypeError(
            "Duration target text must be a string."
        )

    normalized = " ".join(
        text.strip().casefold().split()
    )

    if not normalized:
        raise AllocationTargetParseError(
            "Duration target text cannot be empty."
        )

    word_hours_match = re.fullmatch(
        r"(un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze) "
        r"heures?",
        normalized,
    )

    if word_hours_match is not None:
        hours = _FRENCH_HOUR_WORDS[
            word_hours_match.group(1)
        ]

        return _duration_target_from_minutes(
            hours * Decimal("60")
        )

    numeric_hours_match = re.fullmatch(
        r"([0-9]+(?:[.,][0-9]+)?)\s*heures?",
        normalized,
    )

    if numeric_hours_match is not None:
        hours_text = (
            numeric_hours_match.group(1)
            .replace(",", ".")
        )

        try:
            hours = Decimal(
                hours_text
            )
        except InvalidOperation as error:
            raise AllocationTargetParseError(
                "Invalid numeric hour duration."
            ) from error

        return _duration_target_from_minutes(
            hours * Decimal("60")
        )

    minutes_match = re.fullmatch(
        r"([0-9]+(?:[.,][0-9]+)?)\s*(?:minutes?|min)",
        normalized,
    )

    if minutes_match is not None:
        minutes_text = (
            minutes_match.group(1)
            .replace(",", ".")
        )

        try:
            minutes = Decimal(
                minutes_text
            )
        except InvalidOperation as error:
            raise AllocationTargetParseError(
                "Invalid numeric minute duration."
            ) from error

        return _duration_target_from_minutes(
            minutes
        )

    hours_minutes_match = re.fullmatch(
        r"([0-9]+)\s*h\s*([0-9]+)"
        r"(?:\s*(?:minutes?|min))?",
        normalized,
    )

    if hours_minutes_match is not None:
        hours = Decimal(
            hours_minutes_match.group(1)
        )
        minute_component = Decimal(
            hours_minutes_match.group(2)
        )

        if minute_component >= Decimal("60"):
            raise AllocationTargetParseError(
                "Minute component must be below 60."
            )

        total_minutes = (
            hours * Decimal("60")
            + minute_component
        )

        return _duration_target_from_minutes(
            total_minutes
        )

    raise AllocationTargetParseError(
        "Duration target text is unsupported, ambiguous "
        "or contains extra context."
    )
