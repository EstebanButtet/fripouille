"""Windows serial connection for assistant hardware."""

from __future__ import annotations

import ctypes
import os
import re
import time
from collections.abc import Callable
from typing import Protocol
from ctypes import wintypes

from assistant_ia.hardware.transport import HardwareTransportError

_BAUD_RATE = 115200
_MAX_DWORD = 0xFFFFFFFF

_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_OPEN_EXISTING = 3
_PURGE_RXCLEAR = 0x0008

_NOPARITY = 0
_ONESTOPBIT = 0
_DTR_CONTROL_DISABLE = 0
_RTS_CONTROL_DISABLE = 0

_PORT_PATTERN = re.compile(
    r"COM[1-9][0-9]*",
    re.IGNORECASE,
)


class WindowsSerialConnectionError(HardwareTransportError):
    """Raised when Windows serial communication cannot be used."""


class _DCB(ctypes.Structure):
    """Win32 device-control block."""

    _fields_ = [
        ("DCBlength", wintypes.DWORD),
        ("BaudRate", wintypes.DWORD),
        ("fBinary", wintypes.DWORD, 1),
        ("fParity", wintypes.DWORD, 1),
        ("fOutxCtsFlow", wintypes.DWORD, 1),
        ("fOutxDsrFlow", wintypes.DWORD, 1),
        ("fDtrControl", wintypes.DWORD, 2),
        ("fDsrSensitivity", wintypes.DWORD, 1),
        ("fTXContinueOnXoff", wintypes.DWORD, 1),
        ("fOutX", wintypes.DWORD, 1),
        ("fInX", wintypes.DWORD, 1),
        ("fErrorChar", wintypes.DWORD, 1),
        ("fNull", wintypes.DWORD, 1),
        ("fRtsControl", wintypes.DWORD, 2),
        ("fAbortOnError", wintypes.DWORD, 1),
        ("fDummy2", wintypes.DWORD, 17),
        ("wReserved", wintypes.WORD),
        ("XonLim", wintypes.WORD),
        ("XoffLim", wintypes.WORD),
        ("ByteSize", wintypes.BYTE),
        ("Parity", wintypes.BYTE),
        ("StopBits", wintypes.BYTE),
        ("XonChar", ctypes.c_char),
        ("XoffChar", ctypes.c_char),
        ("ErrorChar", ctypes.c_char),
        ("EofChar", ctypes.c_char),
        ("EvtChar", ctypes.c_char),
        ("wReserved1", wintypes.WORD),
    ]


class _COMMTIMEOUTS(ctypes.Structure):
    """Win32 serial read and write timeout configuration."""

    _fields_ = [
        ("ReadIntervalTimeout", wintypes.DWORD),
        ("ReadTotalTimeoutMultiplier", wintypes.DWORD),
        ("ReadTotalTimeoutConstant", wintypes.DWORD),
        ("WriteTotalTimeoutMultiplier", wintypes.DWORD),
        ("WriteTotalTimeoutConstant", wintypes.DWORD),
    ]


class _WindowsSerialApi(Protocol):
    """Provide the Win32 operations required by the connection."""

    def open_port(self, device_path: str) -> object:
        """Open one serial device."""

    def configure_port(
        self,
        handle: object,
        baud_rate: int,
    ) -> None:
        """Configure one serial device."""

    def configure_timeouts(
        self,
        handle: object,
    ) -> None:
        """Configure serial read and write timeouts."""

    def purge_input(
        self,
        handle: object,
    ) -> None:
        """Discard stale input bytes."""

    def read(
        self,
        handle: object,
        max_bytes: int,
    ) -> bytes:
        """Read currently available bytes."""

    def write(
        self,
        handle: object,
        data: bytes,
    ) -> None:
        """Write all supplied bytes."""

    def close(
        self,
        handle: object,
    ) -> None:
        """Close the Windows handle."""


class _CtypesWindowsSerialApi:
    """Call the Win32 serial API through the Python standard library."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError(
                "Windows serial access requires Windows."
            )

        kernel32 = ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        )

        self._create_file = kernel32.CreateFileW
        self._create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self._create_file.restype = wintypes.HANDLE

        self._get_comm_state = kernel32.GetCommState
        self._get_comm_state.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_DCB),
        ]
        self._get_comm_state.restype = wintypes.BOOL

        self._set_comm_state = kernel32.SetCommState
        self._set_comm_state.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_DCB),
        ]
        self._set_comm_state.restype = wintypes.BOOL

        self._set_comm_timeouts = kernel32.SetCommTimeouts
        self._set_comm_timeouts.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_COMMTIMEOUTS),
        ]
        self._set_comm_timeouts.restype = wintypes.BOOL

        self._purge_comm = kernel32.PurgeComm
        self._purge_comm.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        self._purge_comm.restype = wintypes.BOOL

        self._read_file = kernel32.ReadFile
        self._read_file.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        self._read_file.restype = wintypes.BOOL

        self._write_file = kernel32.WriteFile
        self._write_file.argtypes = [
            wintypes.HANDLE,
            wintypes.LPCVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        self._write_file.restype = wintypes.BOOL

        self._close_handle = kernel32.CloseHandle
        self._close_handle.argtypes = [
            wintypes.HANDLE,
        ]
        self._close_handle.restype = wintypes.BOOL

    def open_port(self, device_path: str) -> object:
        """Open one COM device for exclusive read/write access."""
        handle = self._create_file(
            device_path,
            _GENERIC_READ | _GENERIC_WRITE,
            0,
            None,
            _OPEN_EXISTING,
            0,
            None,
        )

        invalid_handle = ctypes.c_void_p(-1).value

        if handle == invalid_handle:
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

        return handle

    def configure_port(
        self,
        handle: object,
        baud_rate: int,
    ) -> None:
        """Configure 115200 baud, 8 data bits, no parity, one stop bit."""
        dcb = _DCB()
        dcb.DCBlength = ctypes.sizeof(_DCB)

        if not self._get_comm_state(
            handle,
            ctypes.byref(dcb),
        ):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

        dcb.BaudRate = baud_rate
        dcb.ByteSize = 8
        dcb.Parity = _NOPARITY
        dcb.StopBits = _ONESTOPBIT

        dcb.fBinary = 1
        dcb.fParity = 0
        dcb.fOutxCtsFlow = 0
        dcb.fOutxDsrFlow = 0
        dcb.fDtrControl = _DTR_CONTROL_DISABLE
        dcb.fDsrSensitivity = 0
        dcb.fOutX = 0
        dcb.fInX = 0
        dcb.fRtsControl = _RTS_CONTROL_DISABLE
        dcb.fAbortOnError = 0

        if not self._set_comm_state(
            handle,
            ctypes.byref(dcb),
        ):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

    def configure_timeouts(
        self,
        handle: object,
    ) -> None:
        """Use immediate reads and a bounded write timeout."""
        timeouts = _COMMTIMEOUTS()
        timeouts.ReadIntervalTimeout = _MAX_DWORD
        timeouts.ReadTotalTimeoutMultiplier = 0
        timeouts.ReadTotalTimeoutConstant = 0
        timeouts.WriteTotalTimeoutMultiplier = 0
        timeouts.WriteTotalTimeoutConstant = 1000

        if not self._set_comm_timeouts(
            handle,
            ctypes.byref(timeouts),
        ):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

    def purge_input(
        self,
        handle: object,
    ) -> None:
        """Discard stale boot output accumulated after opening."""
        if not self._purge_comm(
            handle,
            _PURGE_RXCLEAR,
        ):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

    def read(
        self,
        handle: object,
        max_bytes: int,
    ) -> bytes:
        """Return currently available serial bytes."""
        buffer = ctypes.create_string_buffer(
            max_bytes
        )
        bytes_read = wintypes.DWORD()

        if not self._read_file(
            handle,
            buffer,
            max_bytes,
            ctypes.byref(bytes_read),
            None,
        ):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

        return buffer.raw[:bytes_read.value]

    def write(
        self,
        handle: object,
        data: bytes,
    ) -> None:
        """Write all bytes to the serial device."""
        if not data:
            return

        buffer = (
            ctypes.c_char * len(data)
        ).from_buffer_copy(data)

        bytes_written = wintypes.DWORD()

        if not self._write_file(
            handle,
            buffer,
            len(data),
            ctypes.byref(bytes_written),
            None,
        ):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )

        if bytes_written.value != len(data):
            raise OSError(
                "Windows serial write was incomplete."
            )

    def close(
        self,
        handle: object,
    ) -> None:
        """Close the serial device handle."""
        if not self._close_handle(handle):
            raise ctypes.WinError(
                ctypes.get_last_error()
            )


class WindowsSerialConnection:
    """Provide raw serial access to Fripouille hardware on Windows."""

    def __init__(
        self,
        port_name: str,
        *,
        startup_delay: float = 2.0,
        api: _WindowsSerialApi | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Open and configure one Windows COM port."""
        normalized_port = _normalize_port_name(
            port_name
        )

        if (
            isinstance(startup_delay, bool)
            or not isinstance(
                startup_delay,
                (int, float),
            )
        ):
            raise TypeError(
                "Serial startup delay must be a number."
            )

        if startup_delay < 0:
            raise ValueError(
                "Serial startup delay cannot be negative."
            )

        if not callable(sleep):
            raise TypeError(
                "Serial sleep function must be callable."
            )

        self._api = (
            api
            if api is not None
            else _CtypesWindowsSerialApi()
        )
        self._handle: object | None = None
        self._closed = True

        device_path = (
            rf"\\.\{normalized_port}"
        )

        try:
            handle = self._api.open_port(
                device_path
            )
            self._handle = handle

            self._api.configure_port(
                handle,
                _BAUD_RATE,
            )
            self._api.configure_timeouts(
                handle
            )

            if startup_delay:
                sleep(
                    float(startup_delay)
                )

            self._api.purge_input(
                handle
            )
        except OSError as error:
            self._close_after_initialization_failure()

            raise WindowsSerialConnectionError(
                "Windows serial port could not be initialized."
            ) from error

        self._closed = False

    def read(
        self,
        max_bytes: int,
    ) -> bytes:
        """Read currently available raw serial bytes."""
        self._require_open()

        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
        ):
            raise TypeError(
                "Serial read size must be an integer."
            )

        if max_bytes <= 0:
            raise ValueError(
                "Serial read size must be positive."
            )

        try:
            return self._api.read(
                self._handle,
                max_bytes,
            )
        except OSError as error:
            raise WindowsSerialConnectionError(
                "Windows serial data could not be read."
            ) from error

    def write(
        self,
        data: bytes,
    ) -> None:
        """Write raw bytes to the serial device."""
        self._require_open()

        if not isinstance(data, bytes):
            raise TypeError(
                "Serial output must be bytes."
            )

        try:
            self._api.write(
                self._handle,
                data,
            )
        except OSError as error:
            raise WindowsSerialConnectionError(
                "Windows serial data could not be written."
            ) from error

    def close(self) -> None:
        """Close the serial connection once."""
        if self._closed:
            return

        handle = self._handle
        self._closed = True
        self._handle = None

        try:
            self._api.close(
                handle
            )
        except OSError as error:
            raise WindowsSerialConnectionError(
                "Windows serial port could not be closed."
            ) from error

    def _require_open(self) -> None:
        """Reject operations after the connection is closed."""
        if self._closed:
            raise WindowsSerialConnectionError(
                "Windows serial connection is closed."
            )

    def _close_after_initialization_failure(
        self,
    ) -> None:
        """Best-effort cleanup after partial initialization."""
        if self._handle is None:
            return

        try:
            self._api.close(
                self._handle
            )
        except OSError:
            pass

        self._handle = None


def _normalize_port_name(
    port_name: str,
) -> str:
    """Validate and normalize one Windows COM port name."""
    if not isinstance(port_name, str):
        raise TypeError(
            "Windows serial port name must be a string."
        )

    normalized = port_name.strip().upper()

    if not _PORT_PATTERN.fullmatch(
        normalized
    ):
        raise ValueError(
            "Windows serial port name must use the COM<number> form."
        )

    return normalized
