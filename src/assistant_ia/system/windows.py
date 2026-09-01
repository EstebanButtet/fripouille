"""Lancement contrôlé d'applications Windows explicitement autorisées.

Le catalogue associe des noms logiques et alias à des commandes internes
immuables. Une demande utilisateur ne fournit jamais un exécutable ou des
arguments libres : elle sélectionne exactement une entrée de cette liste.
Le lancement utilise ``subprocess`` sans shell, après les contrôles de
permission et de confirmation effectués par :mod:`assistant_ia.actions`.
"""

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
    """Base des erreurs d'accès contrôlé aux applications Windows."""


class WindowsApplicationNotAllowedError(WindowsApplicationError):
    """Signaler qu'une application est absente de la liste blanche."""


class WindowsApplicationLaunchError(WindowsApplicationError):
    """Signaler que Windows n'a pas pu lancer une application autorisée."""


@dataclass(frozen=True, slots=True)
class WindowsApplication:
    """Décrire une application explicitement autorisée.

    ``name`` et ``aliases`` servent à la résolution exacte ; ``display_name``
    sert à la confirmation ; ``command`` est construit par le code et non par
    le modèle. Les shells et moteurs de scripts sensibles sont interdits.
    """

    name: str
    display_name: str
    command: tuple[str, ...]
    aliases: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Valider, normaliser et figer une entrée de liste blanche."""
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
    """Construire le catalogue de confiance des applications Windows.

    Les chemins sont des constantes applicatives. L'emplacement local facultatif
    sert uniquement à construire l'entrée Discord connue.
    """
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
    """Résoudre puis lancer une cible de la liste blanche Windows.

    L'index des alias est préparé à la construction et refuse toute ambiguïté.
    Le lanceur de processus est injectable pour tester sans démarrer réellement
    une application.
    """

    def __init__(
        self,
        applications: Iterable[WindowsApplication] | None = None,
        process_launcher: ProcessLauncher | None = None,
    ) -> None:
        """Créer le lanceur et son index d'identifiants uniques."""
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
        """Retourner les définitions immuables du catalogue configuré."""
        return self._applications

    def resolve_application(
        self,
        application_name: str,
    ) -> WindowsApplication:
        """Résoudre exactement un nom logique ou un alias de la liste blanche."""
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
        """Lancer une application autorisée sans passer par un shell."""
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
    """Lancer la commande interne sous forme d'arguments, sans shell."""
    return subprocess.Popen(
        command,
        shell=False,
    )
