"""Tests for framed assistant hardware serial transport."""

from __future__ import annotations

import unittest

from assistant_ia.hardware.serial_transport import (
    FramedSerialTransport,
)
from assistant_ia.hardware.transport import (
    HardwareTransportError,
)


class FakeSerialConnection:
    """Provide deterministic raw serial communication."""

    def __init__(
        self,
        reads: list[bytes] | None = None,
        *,
        fail_read: bool = False,
        fail_write: bool = False,
    ) -> None:
        self.reads = list(
            reads if reads is not None else []
        )
        self.fail_read = fail_read
        self.fail_write = fail_write
        self.writes: list[bytes] = []
        self.closed = False

    def read(self, max_bytes: int) -> bytes:
        if self.fail_read:
            raise OSError(
                "Simulated serial read failure."
            )

        if not self.reads:
            return b""

        return self.reads.pop(0)[:max_bytes]

    def write(self, data: bytes) -> None:
        if self.fail_write:
            raise OSError(
                "Simulated serial write failure."
            )

        self.writes.append(data)

    def close(self) -> None:
        self.closed = True


class FakeClock:
    """Provide deterministic monotonic time."""

    def __init__(self) -> None:
        self.current = 0.0

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += seconds


class FramedSerialTransportTests(unittest.TestCase):
    """Validate serial framing independently from Windows and hardware."""

    def _transport(
        self,
        connection: FakeSerialConnection,
    ) -> FramedSerialTransport:
        clock = FakeClock()

        return FramedSerialTransport(
            connection,
            response_timeout=1.0,
            poll_interval=0.1,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    def test_request_writes_expected_frame(self) -> None:
        connection = FakeSerialConnection(
            [b"@PONG\n"]
        )
        transport = self._transport(
            connection
        )

        response = transport.request(
            "PING"
        )

        self.assertEqual(
            connection.writes,
            [b"@PING\n"],
        )
        self.assertEqual(
            response,
            "PONG",
        )

    def test_request_ignores_boot_output_before_response(self) -> None:
        connection = FakeSerialConnection(
            [
                (
                    b"I (117) esp_image: "
                    b"segment size@PONG\r\n"
                ),
            ]
        )
        transport = self._transport(
            connection
        )

        self.assertEqual(
            transport.request("PING"),
            "PONG",
        )

    def test_request_handles_fragmented_response(self) -> None:
        connection = FakeSerialConnection(
            [
                b"I (10) boot log\n@PO",
                b"NG\r",
                b"\n",
            ]
        )
        transport = self._transport(
            connection
        )

        self.assertEqual(
            transport.request("PING"),
            "PONG",
        )

    def test_request_encodes_utf8_command(self) -> None:
        connection = FakeSerialConnection(
            [b"@OK TEXT\n"]
        )
        transport = self._transport(
            connection
        )

        transport.request(
            "TEXT R?ponse pr?te."
        )

        self.assertEqual(
            connection.writes,
            [
                (
                    "@TEXT R?ponse pr?te.\n"
                ).encode("utf-8"),
            ],
        )

    def test_request_rejects_protocol_control_characters(self) -> None:
        connection = FakeSerialConnection()
        transport = self._transport(
            connection
        )

        for command in (
            "TEXT ligne 1\nligne 2",
            "TEXT test@adresse",
        ):
            with (
                self.subTest(command=command),
                self.assertRaises(ValueError),
            ):
                transport.request(
                    command
                )

        self.assertEqual(
            connection.writes,
            [],
        )

    def test_request_times_out_without_response(self) -> None:
        connection = FakeSerialConnection()
        transport = self._transport(
            connection
        )

        with self.assertRaisesRegex(
            HardwareTransportError,
            "in time",
        ):
            transport.request(
                "PING"
            )

    def test_request_converts_write_failure(self) -> None:
        connection = FakeSerialConnection(
            fail_write=True
        )
        transport = self._transport(
            connection
        )

        with self.assertRaisesRegex(
            HardwareTransportError,
            "could not be written",
        ):
            transport.request(
                "PING"
            )

    def test_request_converts_read_failure(self) -> None:
        connection = FakeSerialConnection(
            fail_read=True
        )
        transport = self._transport(
            connection
        )

        with self.assertRaisesRegex(
            HardwareTransportError,
            "could not be read",
        ):
            transport.request(
                "PING"
            )

    def test_close_closes_connection(self) -> None:
        connection = FakeSerialConnection()
        transport = self._transport(
            connection
        )

        transport.close()

        self.assertTrue(
            connection.closed
        )


if __name__ == "__main__":
    unittest.main()
