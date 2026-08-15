"""Tests for the Windows assistant hardware serial connection."""

from __future__ import annotations

import unittest

from assistant_ia.hardware.windows_serial import (
    WindowsSerialConnection,
    WindowsSerialConnectionError,
)


class FakeWindowsSerialApi:
    """Record Win32 serial operations without opening a real port."""

    def __init__(
        self,
        *,
        fail_operation: str | None = None,
    ) -> None:
        self.fail_operation = fail_operation
        self.calls: list[tuple[object, ...]] = []
        self.handle = object()
        self.read_result = b"example"

    def _record(
        self,
        operation: str,
        *values: object,
    ) -> None:
        self.calls.append(
            (operation, *values)
        )

        if self.fail_operation == operation:
            raise OSError(
                f"Simulated {operation} failure."
            )

    def open_port(
        self,
        device_path: str,
    ) -> object:
        self._record(
            "open",
            device_path,
        )
        return self.handle

    def configure_port(
        self,
        handle: object,
        baud_rate: int,
    ) -> None:
        self._record(
            "configure",
            handle,
            baud_rate,
        )

    def configure_timeouts(
        self,
        handle: object,
    ) -> None:
        self._record(
            "timeouts",
            handle,
        )

    def purge_input(
        self,
        handle: object,
    ) -> None:
        self._record(
            "purge",
            handle,
        )

    def read(
        self,
        handle: object,
        max_bytes: int,
    ) -> bytes:
        self._record(
            "read",
            handle,
            max_bytes,
        )
        return self.read_result[:max_bytes]

    def write(
        self,
        handle: object,
        data: bytes,
    ) -> None:
        self._record(
            "write",
            handle,
            data,
        )

    def close(
        self,
        handle: object,
    ) -> None:
        self._record(
            "close",
            handle,
        )


class WindowsSerialConnectionTests(unittest.TestCase):
    """Validate the Windows boundary without accessing real hardware."""

    def test_opens_normalized_port_with_expected_configuration(
        self,
    ) -> None:
        api = FakeWindowsSerialApi()
        sleeps: list[float] = []

        connection = WindowsSerialConnection(
            " com4 ",
            api=api,
            startup_delay=2.0,
            sleep=sleeps.append,
        )

        self.assertEqual(
            api.calls,
            [
                ("open", r"\\.\COM4"),
                ("configure", api.handle, 115200),
                ("timeouts", api.handle),
                ("purge", api.handle),
            ],
        )
        self.assertEqual(
            sleeps,
            [2.0],
        )

        connection.close()

    def test_supports_com_ports_above_nine(self) -> None:
        api = FakeWindowsSerialApi()

        connection = WindowsSerialConnection(
            "COM12",
            api=api,
            startup_delay=0,
        )

        self.assertEqual(
            api.calls[0],
            ("open", r"\\.\COM12"),
        )

        connection.close()

    def test_rejects_invalid_port_name(self) -> None:
        for port_name in (
            "",
            "4",
            "USB4",
            "COM0",
            "COM-4",
            r"\\.\COM4",
        ):
            with (
                self.subTest(port_name=port_name),
                self.assertRaises(ValueError),
            ):
                WindowsSerialConnection(
                    port_name,
                    api=FakeWindowsSerialApi(),
                )

    def test_rejects_non_string_port_name(self) -> None:
        with self.assertRaises(TypeError):
            WindowsSerialConnection(
                4,  # type: ignore[arg-type]
                api=FakeWindowsSerialApi(),
            )

    def test_rejects_invalid_startup_delay(self) -> None:
        with self.assertRaises(TypeError):
            WindowsSerialConnection(
                "COM4",
                api=FakeWindowsSerialApi(),
                startup_delay=True,
            )

        with self.assertRaises(ValueError):
            WindowsSerialConnection(
                "COM4",
                api=FakeWindowsSerialApi(),
                startup_delay=-1,
            )

    def test_read_delegates_to_windows_api(self) -> None:
        api = FakeWindowsSerialApi()
        api.read_result = b"abcdef"

        connection = WindowsSerialConnection(
            "COM4",
            api=api,
            startup_delay=0,
        )

        result = connection.read(3)

        self.assertEqual(
            result,
            b"abc",
        )
        self.assertIn(
            ("read", api.handle, 3),
            api.calls,
        )

        connection.close()

    def test_write_delegates_to_windows_api(self) -> None:
        api = FakeWindowsSerialApi()

        connection = WindowsSerialConnection(
            "COM4",
            api=api,
            startup_delay=0,
        )

        connection.write(
            b"@PING\n"
        )

        self.assertIn(
            (
                "write",
                api.handle,
                b"@PING\n",
            ),
            api.calls,
        )

        connection.close()

    def test_close_is_idempotent(self) -> None:
        api = FakeWindowsSerialApi()

        connection = WindowsSerialConnection(
            "COM4",
            api=api,
            startup_delay=0,
        )

        connection.close()
        connection.close()

        close_calls = [
            call
            for call in api.calls
            if call[0] == "close"
        ]

        self.assertEqual(
            len(close_calls),
            1,
        )

    def test_initialization_failure_closes_open_handle(self) -> None:
        api = FakeWindowsSerialApi(
            fail_operation="configure"
        )

        with self.assertRaisesRegex(
            WindowsSerialConnectionError,
            "could not be initialized",
        ):
            WindowsSerialConnection(
                "COM4",
                api=api,
                startup_delay=0,
            )

        self.assertIn(
            ("close", api.handle),
            api.calls,
        )

    def test_open_failure_is_controlled(self) -> None:
        api = FakeWindowsSerialApi(
            fail_operation="open"
        )

        with self.assertRaisesRegex(
            WindowsSerialConnectionError,
            "could not be initialized",
        ):
            WindowsSerialConnection(
                "COM4",
                api=api,
                startup_delay=0,
            )

    def test_read_failure_is_controlled(self) -> None:
        api = FakeWindowsSerialApi(
            fail_operation="read"
        )

        connection = WindowsSerialConnection(
            "COM4",
            api=api,
            startup_delay=0,
        )

        with self.assertRaisesRegex(
            WindowsSerialConnectionError,
            "could not be read",
        ):
            connection.read(64)

        connection.close()

    def test_write_failure_is_controlled(self) -> None:
        api = FakeWindowsSerialApi(
            fail_operation="write"
        )

        connection = WindowsSerialConnection(
            "COM4",
            api=api,
            startup_delay=0,
        )

        with self.assertRaisesRegex(
            WindowsSerialConnectionError,
            "could not be written",
        ):
            connection.write(
                b"@PING\n"
            )

        connection.close()

    def test_operations_after_close_are_rejected(self) -> None:
        connection = WindowsSerialConnection(
            "COM4",
            api=FakeWindowsSerialApi(),
            startup_delay=0,
        )

        connection.close()

        with self.assertRaisesRegex(
            WindowsSerialConnectionError,
            "closed",
        ):
            connection.read(1)

        with self.assertRaisesRegex(
            WindowsSerialConnectionError,
            "closed",
        ):
            connection.write(
                b"data"
            )


if __name__ == "__main__":
    unittest.main()
