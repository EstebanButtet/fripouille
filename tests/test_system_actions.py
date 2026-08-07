"""Tests for controlled default system assistant actions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant_ia.actions.action import (
    ActionExecutionError,
    ActionValidationError,
)
from assistant_ia.actions.defaults import (
    build_default_action_registry,
)
from assistant_ia.intelligence.intent import Intent
from assistant_ia.memory.journal_repository import JournalRepository
from assistant_ia.memory.memory_repository import MemoryRepository
from assistant_ia.memory.repository import SQLiteDatabase
from assistant_ia.memory.task_repository import TaskRepository
from assistant_ia.security.confirmation import ConfirmationRequest
from assistant_ia.security.permissions import PermissionPolicy
from assistant_ia.system.windows import WindowsApplicationLauncher


class RecordingProcessLauncher:
    """Record process requests without launching real applications."""

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
        """Record a command or simulate a Windows failure."""
        self.commands.append(command)

        if self.fail:
            raise OSError(
                "Simulated Windows launch failure."
            )

        return object()


class RecordingConfirmationHandler:
    """Record confirmation requests and return one fixed decision."""

    def __init__(
        self,
        result: bool,
    ) -> None:
        """Store the deterministic confirmation result."""
        self.result = result
        self.requests: list[ConfirmationRequest] = []

    def __call__(
        self,
        request: ConfirmationRequest,
    ) -> bool:
        """Record and answer one confirmation request."""
        self.requests.append(request)
        return self.result


class SystemDefaultActionTests(unittest.TestCase):
    """Validate secure system actions in the default registry."""

    def setUp(self) -> None:
        """Create an isolated initialized database."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name)
            / "assistant.db"
        )
        self.database.initialize()

    def tearDown(self) -> None:
        """Remove the temporary database."""
        self.temporary_directory.cleanup()

    def _registry(
        self,
        process_launcher: RecordingProcessLauncher,
        *,
        permission_policy: PermissionPolicy | None = None,
        confirmation_handler: RecordingConfirmationHandler | None = None,
    ):
        """Build a registry with controlled system dependencies."""
        return build_default_action_registry(
            task_repository=TaskRepository(
                self.database
            ),
            memory_repository=MemoryRepository(
                self.database
            ),
            journal_repository=JournalRepository(
                self.database
            ),
            permission_policy=permission_policy,
            confirmation_handler=confirmation_handler,
            windows_launcher=WindowsApplicationLauncher(
                process_launcher=process_launcher
            ),
        )

    def test_registers_launch_when_launcher_is_injected(self) -> None:
        """System capability injection should register launch action."""
        registry = self._registry(
            RecordingProcessLauncher()
        )

        self.assertEqual(
            registry.action_count,
            8,
        )
        self.assertTrue(
            registry.has_action("launch_application")
        )

    def test_approved_launch_executes_allowlisted_application(
        self,
    ) -> None:
        """Explicit confirmation should permit a known application."""
        process_launcher = RecordingProcessLauncher()
        confirmation_handler = RecordingConfirmationHandler(
            True
        )
        registry = self._registry(
            process_launcher,
            confirmation_handler=confirmation_handler,
        )

        result = registry.execute(
            Intent(
                name="launch_application",
                parameters={
                    "application": "bloc-notes",
                },
            )
        )

        self.assertEqual(
            result,
            "Application lancée : Bloc-notes.",
        )
        self.assertEqual(
            process_launcher.commands,
            [
                (
                    "notepad.exe",
                ),
            ],
        )
        self.assertEqual(
            len(confirmation_handler.requests),
            1,
        )
        self.assertEqual(
            confirmation_handler.requests[0].action_name,
            "launch_application",
        )
        self.assertEqual(
            confirmation_handler.requests[0].description,
            "lancer Bloc-notes",
        )

    def test_refused_confirmation_prevents_execution(self) -> None:
        """Explicit refusal should cancel before Windows execution."""
        process_launcher = RecordingProcessLauncher()
        confirmation_handler = RecordingConfirmationHandler(
            False
        )
        registry = self._registry(
            process_launcher,
            confirmation_handler=confirmation_handler,
        )

        result = registry.execute(
            Intent(
                name="launch_application",
                parameters={
                    "application": "notepad",
                },
            )
        )

        self.assertEqual(
            result,
            "Lancement annulé : Bloc-notes.",
        )
        self.assertEqual(
            process_launcher.commands,
            [],
        )

    def test_missing_confirmation_handler_fails_closed(self) -> None:
        """System actions should deny confirmation by default."""
        process_launcher = RecordingProcessLauncher()
        registry = self._registry(
            process_launcher
        )

        result = registry.execute(
            Intent(
                name="launch_application",
                parameters={
                    "application": "notepad",
                },
            )
        )

        self.assertEqual(
            result,
            "Lancement annulé : Bloc-notes.",
        )
        self.assertEqual(
            process_launcher.commands,
            [],
        )

    def test_denied_permission_prevents_confirmation_and_launch(
        self,
    ) -> None:
        """Denied actions must stop before confirmation or execution."""
        process_launcher = RecordingProcessLauncher()
        confirmation_handler = RecordingConfirmationHandler(
            True
        )
        registry = self._registry(
            process_launcher,
            permission_policy=PermissionPolicy(
                decisions={
                    "launch_application": "denied",
                }
            ),
            confirmation_handler=confirmation_handler,
        )

        with self.assertRaisesRegex(
            ActionValidationError,
            "n’est pas autorisé",
        ):
            registry.execute(
                Intent(
                    name="launch_application",
                    parameters={
                        "application": "notepad",
                    },
                )
            )

        self.assertEqual(
            confirmation_handler.requests,
            [],
        )
        self.assertEqual(
            process_launcher.commands,
            [],
        )

    def test_unknown_application_is_rejected_before_confirmation(
        self,
    ) -> None:
        """Unknown applications should never reach confirmation."""
        process_launcher = RecordingProcessLauncher()
        confirmation_handler = RecordingConfirmationHandler(
            True
        )
        registry = self._registry(
            process_launcher,
            confirmation_handler=confirmation_handler,
        )

        with self.assertRaisesRegex(
            ActionValidationError,
            "application n’est pas autorisée",
        ):
            registry.execute(
                Intent(
                    name="launch_application",
                    parameters={
                        "application": "unknown application",
                    },
                )
            )

        self.assertEqual(
            confirmation_handler.requests,
            [],
        )
        self.assertEqual(
            process_launcher.commands,
            [],
        )

    def test_malicious_application_text_never_reaches_windows(
        self,
    ) -> None:
        """Command-like application values must remain inert data."""
        process_launcher = RecordingProcessLauncher()
        confirmation_handler = RecordingConfirmationHandler(
            True
        )
        registry = self._registry(
            process_launcher,
            confirmation_handler=confirmation_handler,
        )

        for application_name in (
            "notepad && del C:\\data",
            "powershell",
            "cmd /c dir",
            "C:\\Windows\\System32\\notepad.exe",
        ):
            with (
                self.subTest(
                    application_name=application_name
                ),
                self.assertRaises(ActionValidationError),
            ):
                registry.execute(
                    Intent(
                        name="launch_application",
                        parameters={
                            "application": application_name,
                        },
                    )
                )

        self.assertEqual(
            confirmation_handler.requests,
            [],
        )
        self.assertEqual(
            process_launcher.commands,
            [],
        )

    def test_windows_launch_failure_becomes_action_error(
        self,
    ) -> None:
        """Windows failures should never produce a success result."""
        process_launcher = RecordingProcessLauncher(
            fail=True
        )
        confirmation_handler = RecordingConfirmationHandler(
            True
        )
        registry = self._registry(
            process_launcher,
            confirmation_handler=confirmation_handler,
        )

        with self.assertRaisesRegex(
            ActionExecutionError,
            "n’a pas pu être lancée",
        ):
            registry.execute(
                Intent(
                    name="launch_application",
                    parameters={
                        "application": "notepad",
                    },
                )
            )

        self.assertEqual(
            process_launcher.commands,
            [
                (
                    "notepad.exe",
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
