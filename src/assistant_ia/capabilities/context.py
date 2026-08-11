"""Structured description of current assistant capabilities."""

from __future__ import annotations

from dataclasses import dataclass

from assistant_ia.actions.registry import ActionRegistry


_ACTION_DESCRIPTIONS = {
    "create_task": (
        "create a persistent task when the user asks."
    ),
    "list_tasks": (
        "list stored tasks when the user asks."
    ),
    "complete_task": (
        "mark a stored task as completed when the user asks."
    ),
    "save_memory": (
        "save a persistent memory when the user explicitly asks."
    ),
    "find_memory": (
        "search explicitly saved persistent memories when the user asks."
    ),
    "delete_memory": (
        "delete an explicitly saved persistent memory when the user asks."
    ),
    "write_journal": (
        "write a persistent journal entry when the user asks."
    ),
    "launch_application": (
        "request the launch of an allowed application, subject to "
        "application permissions and confirmation."
    ),
}


_ACTION_CAPABILITY_STATEMENTS = {
    "create_task": (
        "The assistant can create persistent tasks when the user asks."
    ),
    "list_tasks": (
        "The assistant can list stored tasks when the user asks."
    ),
    "complete_task": (
        "The assistant can mark stored tasks as completed when "
        "the user asks."
    ),
    "save_memory": (
        "The assistant can explicitly save persistent memories "
        "when the user asks."
    ),
    "find_memory": (
        "The assistant can search and retrieve memories that were "
        "explicitly saved, when the user asks."
    ),
    "delete_memory": (
        "The assistant can delete explicitly saved persistent "
        "memories when the user asks."
    ),
    "write_journal": (
        "The assistant can write persistent journal entries "
        "when the user asks."
    ),
    "launch_application": (
        "The assistant can request the launch of an allowed "
        "application on the user's computer."
    ),
}


@dataclass(frozen=True, slots=True)
class CapabilityContext:
    """Describe capabilities that are actually available now."""

    available_actions: tuple[str, ...]
    automatic_memory_retrieval: bool = False
    visual_input: bool = False
    audio_input: bool = False
    robot_control: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.available_actions, tuple):
            raise TypeError(
                "Available actions must be provided as a tuple."
            )

        normalized_actions: list[str] = []

        for action_name in self.available_actions:
            if not isinstance(action_name, str):
                raise TypeError(
                    "Available action names must be strings."
                )

            normalized_name = action_name.strip()

            if not normalized_name:
                raise ValueError(
                    "Available action names must not be empty."
                )

            normalized_actions.append(
                normalized_name
            )

        object.__setattr__(
            self,
            "available_actions",
            tuple(
                sorted(
                    set(normalized_actions)
                )
            ),
        )


def build_capability_context(
    action_registry: ActionRegistry,
) -> CapabilityContext:
    """Build capabilities from the real executable action registry."""
    if not isinstance(action_registry, ActionRegistry):
        raise TypeError(
            "Capability context requires an ActionRegistry."
        )

    return CapabilityContext(
        available_actions=tuple(
            action_registry.action_names
        ),
    )


def render_capability_context(
    context: CapabilityContext,
) -> str:
    """Render current capabilities as compact model context."""
    if not isinstance(context, CapabilityContext):
        raise TypeError(
            "Capability rendering requires a CapabilityContext."
        )

    lines = [
        "Current assistant capabilities:",
        "",
        "Available executable actions:",
    ]

    if context.available_actions:
        for action_name in context.available_actions:
            description = _ACTION_DESCRIPTIONS.get(
                action_name
            )

            if description is None:
                lines.append(
                    f"- {action_name}"
                )
            else:
                lines.append(
                    f"- {action_name}: {description}"
                )
    else:
        lines.append("- none")

    if context.available_actions:
        lines.extend(
            (
                "",
                "Meaning in ordinary language:",
            )
        )

        for action_name in context.available_actions:
            statement = _ACTION_CAPABILITY_STATEMENTS.get(
                action_name
            )

            if statement is not None:
                lines.append(
                    f"- {statement}"
                )

    lines.extend(
        (
            "",
            "Capability interpretation rules:",
            "Registered actions are real current capabilities.",
            "Permissions and confirmations affect execution, not whether "
            "a registered capability exists.",
            "Explicit memory actions are different from automatic "
            "contextual memory retrieval.",
        )
    )

    if "find_memory" in context.available_actions:
        lines.extend(
            (
                "If asked whether the assistant can search explicitly "
                "saved memories on request, the correct answer is yes.",
                "Automatic contextual memory retrieval means passive or "
                "automatic recall without an explicit search request.",
                "It is not the same capability as find_memory.",
            )
        )

    persistent_memory_actions = {
        "save_memory",
        "find_memory",
        "delete_memory",
    }
    has_persistent_memory_actions = bool(
        persistent_memory_actions.intersection(
            context.available_actions
        )
    )

    persistent_memory_statement = (
        "Persistent memories are available only through "
        "explicit registered memory actions."
        if has_persistent_memory_actions
        else "No registered persistent memory action is currently available."
    )
    automatic_memory_statement = (
        "Automatic contextual memory retrieval is available."
        if context.automatic_memory_retrieval
        else "Automatic contextual memory retrieval is not available."
    )
    visual_input_statement = (
        "Visual or webcam input is currently available."
        if context.visual_input
        else "Visual or webcam input is not currently available."
    )
    audio_input_statement = (
        "Audio input is currently available."
        if context.audio_input
        else "Audio input is not currently available."
    )
    robot_control_statement = (
        "Robot or physical hardware control is currently available."
        if context.robot_control
        else "Robot or physical hardware control is not currently available."
    )

    lines.extend(
        (
            "",
            "Capability limits:",
            "Conversation history is temporary to the current conversation.",
            persistent_memory_statement,
            automatic_memory_statement,
            visual_input_statement,
            audio_input_statement,
            robot_control_statement,
            "",
            "Do not claim capabilities that are not listed as available.",
            "Do not describe planned or future capabilities as currently "
            "available.",
            "Do not invent security reasons, design rationales or "
            "implementation reasons for capability limits.",
        )
    )

    return "\n".join(lines)
