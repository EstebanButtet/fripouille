"""Tests for controlled Windows application launching."""

from __future__ import annotations

import unittest

from assistant_ia.system.windows import (
    WindowsApplication,
    WindowsApplicationLaunchError,
    WindowsApplicationLauncher,
    WindowsApplicationNotAllowedError,
    build_default_windows_applications,
)


class FakeProcessLauncher:
    """Record process launches without starting real applications."""

    def __init__(
        self,
        *,
        fail: bool = False,
    ) -> None:
        """Configure deterministic fake process execution."""
        self.fail = fail
        self.commands: list[tuple[str, ...]] = []

    def __call__(
        self,
        command: tuple[str, ...],
    ) -> object:
        """Record one command or simulate an operating system failure."""
        self.commands.append(command)

        if self.fail:
            raise OSError(
                "Simulated Windows launch failure."
            )

        return object()


class WindowsApplicationLauncherTests(unittest.TestCase):
    """Validate the Windows application allowlist boundary."""

    def _applications(
        self,
    ) -> tuple[WindowsApplication, ...]:
        """Return deterministic default applications."""
        return build_default_windows_applications(
            local_app_data=(
                "C:\\Users\\Test\\AppData\\Local"
            )
        )

    def _launcher(
        self,
        process_launcher: FakeProcessLauncher | None = None,
    ) -> WindowsApplicationLauncher:
        """Build a deterministic launcher for tests."""
        return WindowsApplicationLauncher(
            applications=self._applications(),
            process_launcher=(
                process_launcher
                if process_launcher is not None
                else FakeProcessLauncher()
            ),
        )

    def test_builds_expected_default_catalogue(self) -> None:
        """The trusted catalogue should contain supported applications."""
        self.assertEqual(
            tuple(
                application.name
                for application in self._applications()
            ),
            (
                "notepad",
                "edge",
                "league_of_legends",
                "valorant",
                "overwatch_2",
                "discord",
            ),
        )

    def test_resolves_known_application_aliases(self) -> None:
        """Known aliases should resolve to canonical applications."""
        launcher = self._launcher()

        cases = {
            "notepad": "notepad",
            "bloc-notes": "notepad",
            "bloc-note": "notepad",
            "bloc notes": "notepad",
            "edge": "edge",
            "microsoft edge": "edge",
            "league of legends": "league_of_legends",
            "league": "league_of_legends",
            "lol": "league_of_legends",
            "valorant": "valorant",
            "valo": "valorant",
            "overwatch 2": "overwatch_2",
            "overwatch": "overwatch_2",
            "ow2": "overwatch_2",
            "discord": "discord",
        }

        for supplied_name, expected_name in cases.items():
            with self.subTest(
                supplied_name=supplied_name
            ):
                application = launcher.resolve_application(
                    supplied_name
                )

                self.assertEqual(
                    application.name,
                    expected_name,
                )

    def test_default_commands_are_fixed_and_exact(self) -> None:
        """Trusted applications should expose exact internal commands."""
        applications = {
            application.name: application
            for application in self._applications()
        }

        self.assertEqual(
            applications["notepad"].command,
            (
                "notepad.exe",
            ),
        )
        self.assertEqual(
            applications["edge"].command,
            (
                (
                    "C:\\Program Files (x86)\\Microsoft"
                    "\\Edge\\Application\\msedge.exe"
                ),
            ),
        )
        self.assertEqual(
            applications["league_of_legends"].command,
            (
                (
                    "C:\\Riot Games\\Riot Client"
                    "\\RiotClientServices.exe"
                ),
                "--launch-product=league_of_legends",
                "--launch-patchline=live",
            ),
        )
        self.assertEqual(
            applications["valorant"].command,
            (
                (
                    "C:\\Riot Games\\Riot Client"
                    "\\RiotClientServices.exe"
                ),
                "--launch-product=valorant",
                "--launch-patchline=live",
            ),
        )
        self.assertEqual(
            applications["overwatch_2"].command,
            (
                (
                    "C:\\Program Files (x86)\\Overwatch"
                    "\\Overwatch Launcher.exe"
                ),
                "--productcode=pro",
            ),
        )
        self.assertEqual(
            applications["discord"].command,
            (
                (
                    "C:\\Users\\Test\\AppData\\Local"
                    "\\Discord\\Update.exe"
                ),
                "--processStart",
                "Discord.exe",
            ),
        )

    def test_rejects_unknown_application(self) -> None:
        """Applications outside the allowlist should be rejected."""
        launcher = self._launcher()

        with self.assertRaisesRegex(
            WindowsApplicationNotAllowedError,
            "not allowed",
        ):
            launcher.resolve_application(
                "unknown application"
            )

    def test_rejects_malicious_input_without_execution(self) -> None:
        """Untrusted command-like text must never reach execution."""
        process_launcher = FakeProcessLauncher()
        launcher = self._launcher(
            process_launcher
        )

        malicious_inputs = (
            "notepad && del C:\\data",
            "powershell",
            "powershell -Command Get-Process",
            "cmd /c dir",
            "C:\\Windows\\System32\\notepad.exe",
            "lol --launch-patchline=pbe",
            "edge https://example.com",
        )

        for application_name in malicious_inputs:
            with (
                self.subTest(
                    application_name=application_name
                ),
                self.assertRaises(
                    WindowsApplicationNotAllowedError
                ),
            ):
                launcher.launch(
                    application_name
                )

        self.assertEqual(
            process_launcher.commands,
            [],
        )

    def test_launches_using_exact_internal_command(self) -> None:
        """Launch should forward only the predefined trusted command."""
        process_launcher = FakeProcessLauncher()
        launcher = self._launcher(
            process_launcher
        )

        application = launcher.launch(
            "lol"
        )

        self.assertEqual(
            application.display_name,
            "League of Legends",
        )
        self.assertEqual(
            process_launcher.commands,
            [
                (
                    (
                        "C:\\Riot Games\\Riot Client"
                        "\\RiotClientServices.exe"
                    ),
                    "--launch-product=league_of_legends",
                    "--launch-patchline=live",
                ),
            ],
        )

    def test_converts_windows_process_failure(self) -> None:
        """Operating system launch failures should be controlled."""
        process_launcher = FakeProcessLauncher(
            fail=True
        )
        launcher = self._launcher(
            process_launcher
        )

        with self.assertRaisesRegex(
            WindowsApplicationLaunchError,
            "could not be launched",
        ):
            launcher.launch(
                "edge"
            )

        self.assertEqual(
            len(process_launcher.commands),
            1,
        )

    def test_accepts_absolute_path_and_fixed_arguments(self) -> None:
        """Trusted definitions may contain paths and fixed arguments."""
        application = WindowsApplication(
            name="example",
            display_name="Example",
            command=(
                "C:\\Program Files\\Example\\example.exe",
                "--safe-fixed-option",
            ),
        )

        self.assertEqual(
            application.command,
            (
                "C:\\Program Files\\Example\\example.exe",
                "--safe-fixed-option",
            ),
        )

    def test_rejects_empty_command(self) -> None:
        """Every allowlisted application needs an executable command."""
        with self.assertRaisesRegex(
            ValueError,
            "command cannot be empty",
        ):
            WindowsApplication(
                name="invalid",
                display_name="Invalid",
                command=(),
            )

    def test_rejects_non_executable_command(self) -> None:
        """The first command component must identify an executable."""
        with self.assertRaisesRegex(
            ValueError,
            "start with an executable",
        ):
            WindowsApplication(
                name="invalid",
                display_name="Invalid",
                command=(
                    "not-a-program.txt",
                ),
            )

    def test_rejects_shell_executable_in_allowlist(self) -> None:
        """Shell executables should remain forbidden even internally."""
        for executable in (
            "cmd.exe",
            "powershell.exe",
            "C:\\Windows\\System32\\cmd.exe",
        ):
            with (
                self.subTest(executable=executable),
                self.assertRaisesRegex(
                    ValueError,
                    "cannot be allowlisted",
                ),
            ):
                WindowsApplication(
                    name="unsafe",
                    display_name="Unsafe",
                    command=(
                        executable,
                    ),
                )

    def test_rejects_duplicate_application_alias(self) -> None:
        """Different applications must not share one identifier."""
        first_application = WindowsApplication(
            name="first",
            display_name="First",
            command=(
                "first.exe",
            ),
            aliases=frozenset(
                {
                    "shared",
                }
            ),
        )
        second_application = WindowsApplication(
            name="second",
            display_name="Second",
            command=(
                "second.exe",
            ),
            aliases=frozenset(
                {
                    "shared",
                }
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "identifiers must be unique",
        ):
            WindowsApplicationLauncher(
                applications=(
                    first_application,
                    second_application,
                ),
                process_launcher=FakeProcessLauncher(),
            )

    def test_rejects_non_callable_process_launcher(self) -> None:
        """Process execution must use an injectable callable."""
        with self.assertRaisesRegex(
            TypeError,
            "process launcher must be callable",
        ):
            WindowsApplicationLauncher(
                applications=self._applications(),
                process_launcher="subprocess",
            )


if __name__ == "__main__":
    unittest.main()
