"""Controlled Windows application launching."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from os import PathLike
from pathlib import PureWindowsPath

ProcessLauncher = Callable[[tuple[str, ...]], object]

_BLOCKED_EXECUTABLE_NAMES: frozenset[str] = frozenset(
    {
        "cmd.exe",
        "powershell.exe",
        "pwsh.exe",
        "wscript.exe",
        "cscript.exe",
        "mshta.exe",
        "rundll32.exe",
        "regsvr32.exe",
    }
)


class WindowsApplicationError(RuntimeError):
    """Base error raised by controlled Windows application access."""


class WindowsApplicationNotAllowedError(WindowsApplicationError):
    """Raised when an application is absent from the explicit allowlist."""


class WindowsApplicationLaunchError(WindowsApplicationError):
    """Raised when Windows cannot launch an allowed application."""


@dataclass(frozen=True, slots=True)
class WindowsApplication:
    """Describe one explicitly allowed Windows application."""

    name: str
    display_name: str
    command: tuple[str, ...]
    aliases: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate and normalize one allowlisted application."""
        if not isinstance(self.name, str):
            raise TypeError(
                "Windows application name must be a string."
            )

        if not isinstance(self.display_name, str):
            raise TypeError(
                "Windows application display name must be a string."
            )

        if not isinstance(self.command, tuple):
            raise TypeError(
                "Windows application command must be a tuple."
            )

        if not isinstance(self.aliases, frozenset):
            raise TypeError(
                "Windows application aliases must be a frozenset."
            )

        normalized_name = self.name.strip().casefold()
        normalized_display_name = self.display_name.strip()

        if not normalized_name:
            raise ValueError(
                "Windows application name cannot be empty."
            )

        if not normalized_display_name:
            raise ValueError(
                "Windows application display name cannot be empty."
            )

        if not self.command:
            raise ValueError(
                "Windows application command cannot be empty."
            )

        normalized_command: list[str] = []

        for component in self.command:
            if not isinstance(component, str):
                raise TypeError(
                    "Windows application command components "
                    "must be strings."
                )

            normalized_component = component.strip()

            if not normalized_component:
                raise ValueError(
                    "Windows application command components "
                    "cannot be empty."
                )

            if any(
                character in normalized_component
                for character in (
                    "\x00",
                    "\r",
                    "\n",
                )
            ):
                raise ValueError(
                    "Windows application command components "
                    "contain invalid characters."
                )

            normalized_command.append(
                normalized_component
            )

        executable = normalized_command[0]

        if not executable.casefold().endswith(".exe"):
            raise ValueError(
                "Windows application command must start "
                "with an executable."
            )

        executable_name = PureWindowsPath(
            executable
        ).name.casefold()

        if executable_name in _BLOCKED_EXECUTABLE_NAMES:
            raise ValueError(
                "Windows shell and script executables "
                "cannot be allowlisted."
            )

        normalized_aliases: set[str] = set()

        for alias in self.aliases:
            if not isinstance(alias, str):
                raise TypeError(
                    "Windows application aliases must be strings."
                )

            normalized_alias = alias.strip().casefold()

            if not normalized_alias:
                raise ValueError(
                    "Windows application aliases cannot be empty."
                )

            normalized_aliases.add(normalized_alias)

        object.__setattr__(
            self,
            "name",
            normalized_name,
        )
        object.__setattr__(
            self,
            "display_name",
            normalized_display_name,
        )
        object.__setattr__(
            self,
            "command",
            tuple(normalized_command),
        )
        object.__setattr__(
            self,
            "aliases",
            frozenset(normalized_aliases),
        )


def build_default_windows_applications(
    local_app_data: str | PathLike[str] | None = None,
) -> tuple[WindowsApplication, ...]:
    """Build the trusted Windows application catalogue."""
    applications = [
        WindowsApplication(
            name="notepad",
            display_name="Bloc-notes",
            command=(
                "notepad.exe",
            ),
            aliases=frozenset(
                {
                    "bloc-notes",
                    "bloc-note",
                    "bloc notes",
                }
            ),
        ),
        WindowsApplication(
            name="edge",
            display_name="Microsoft Edge",
            command=(
                (
                    "C:\\Program Files (x86)\\Microsoft"
                    "\\Edge\\Application\\msedge.exe"
                ),
            ),
            aliases=frozenset(
                {
                    "microsoft edge",
                }
            ),
        ),
        WindowsApplication(
            name="league_of_legends",
            display_name="League of Legends",
            command=(
                (
                    "C:\\Riot Games\\Riot Client"
                    "\\RiotClientServices.exe"
                ),
                "--launch-product=league_of_legends",
                "--launch-patchline=live",
            ),
            aliases=frozenset(
                {
                    "league of legends",
                    "league",
                    "lol",
                }
            ),
        ),
        WindowsApplication(
            name="valorant",
            display_name="VALORANT",
            command=(
                (
                    "C:\\Riot Games\\Riot Client"
                    "\\RiotClientServices.exe"
                ),
                "--launch-product=valorant",
                "--launch-patchline=live",
            ),
            aliases=frozenset(
                {
                    "valo",
                }
            ),
        ),
        WindowsApplication(
            name="overwatch_2",
            display_name="Overwatch 2",
            command=(
                (
                    "C:\\Program Files (x86)\\Overwatch"
                    "\\Overwatch Launcher.exe"
                ),
                "--productcode=pro",
            ),
            aliases=frozenset(
                {
                    "overwatch 2",
                    "overwatch",
                    "ow2",
                }
            ),
        ),
    ]

    resolved_local_app_data: str | None

    if local_app_data is None:
        resolved_local_app_data = os.environ.get(
            "LOCALAPPDATA"
        )
    else:
        try:
            resolved_local_app_data = os.fspath(
                local_app_data
            )
        except TypeError as error:
            raise TypeError(
                "Windows local application data path must "
                "be path-like."
            ) from error

    if (
        resolved_local_app_data is not None
        and resolved_local_app_data.strip()
    ):
        discord_update = str(
            PureWindowsPath(
                resolved_local_app_data.strip()
            )
            / "Discord"
            / "Update.exe"
        )

        applications.append(
            WindowsApplication(
                name="discord",
                display_name="Discord",
                command=(
                    discord_update,
                    "--processStart",
                    "Discord.exe",
                ),
            )
        )

    return tuple(applications)


class WindowsApplicationLauncher:
    """Resolve and launch applications from an explicit Windows allowlist."""

    def __init__(
        self,
        applications: Iterable[WindowsApplication] | None = None,
        process_launcher: ProcessLauncher | None = None,
    ) -> None:
        """Create a launcher with injectable process execution."""
        resolved_applications = (
            tuple(applications)
            if applications is not None
            else build_default_windows_applications()
        )

        alias_index: dict[str, WindowsApplication] = {}

        for application in resolved_applications:
            if not isinstance(application, WindowsApplication):
                raise TypeError(
                    "Windows allowlist entries must be "
                    "WindowsApplication instances."
                )

            identifiers = {
                application.name,
                *application.aliases,
            }

            for identifier in identifiers:
                existing_application = alias_index.get(
                    identifier
                )

                if (
                    existing_application is not None
                    and existing_application != application
                ):
                    raise ValueError(
                        "Windows application identifiers "
                        "must be unique."
                    )

                alias_index[identifier] = application

        if process_launcher is not None and not callable(
            process_launcher
        ):
            raise TypeError(
                "Windows process launcher must be callable."
            )

        self._applications = resolved_applications
        self._alias_index = alias_index
        self._process_launcher = (
            process_launcher
            if process_launcher is not None
            else _launch_process
        )

    @property
    def applications(
        self,
    ) -> tuple[WindowsApplication, ...]:
        """Return the immutable configured application definitions."""
        return self._applications

    def resolve_application(
        self,
        application_name: str,
    ) -> WindowsApplication:
        """Resolve one exact logical name or alias from the allowlist."""
        if not isinstance(application_name, str):
            raise TypeError(
                "Windows application lookup must be a string."
            )

        normalized_name = application_name.strip().casefold()

        if not normalized_name:
            raise ValueError(
                "Windows application lookup cannot be empty."
            )

        application = self._alias_index.get(
            normalized_name
        )

        if application is None:
            raise WindowsApplicationNotAllowedError(
                "Windows application is not allowed."
            )

        return application

    def launch(
        self,
        application_name: str,
    ) -> WindowsApplication:
        """Launch one allowlisted application without using a shell."""
        application = self.resolve_application(
            application_name
        )

        try:
            self._process_launcher(
                application.command
            )
        except OSError as error:
            raise WindowsApplicationLaunchError(
                "Windows application could not be launched."
            ) from error

        return application


def _launch_process(
    command: tuple[str, ...],
) -> object:
    """Launch one internally constructed command without a shell."""
    return subprocess.Popen(
        command,
        shell=False,
    )
