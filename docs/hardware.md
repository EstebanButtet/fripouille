# Hardware boundary

Fripouille's cognitive architecture is designed to remain independent of a
particular body. This repository contains a limited display prototype, not the
full mobile robot.

## Hardware represented here

The tracked prototype targets:

- a Windows PC running the Python assistant;
- a Waveshare ESP32-S3-Touch-LCD-4 board, identified by the ESP-IDF component
  `waveshare/esp32_s3_touch_lcd_4`;
- the board display through LVGL 9.3.0;
- USB Serial/JTAG for the firmware console and framed commands;
- a simple on-screen face, text label, and bundled French-capable LVGL font.

The ESP-IDF defaults keep the board's 16 MB flash and octal PSRAM settings and
enable LVGL vector graphics. `sdkconfig.defaults` is intentionally versioned.

## PC-to-display path

The Python side separates each responsibility:

```text
AssistantRuntime final response
    -> DisplayResponsePresenter
    -> DisplayController
    -> FramedSerialTransport
    -> WindowsSerialConnection
    -> Windows COM port
    -> ESP32 firmware
```

`WindowsSerialConnection` validates `COM<number>` names and configures the
port for 115200 baud, 8 data bits, no parity, and one stop bit. The transport
frames UTF-8 lines as `@<payload>\n`, limits command size, ignores leading
noise, and waits for a framed response.

The high-level display protocol currently supports:

- `PING`, answered with `PONG` by the firmware;
- `TEXT <content>`, acknowledged as `OK TEXT` and rendered on the display.

The controller validates text before transport. It rejects protocol control
characters, line breaks, and oversized UTF-8 commands. The firmware also
rejects unknown or overlong commands.

## What software tests verify

Unit tests cover:

- serial framing, partial reads, invalid UTF-8, size limits, and timeouts;
- high-level display command validation and acknowledgements;
- response formatting and safe UTF-8 truncation for the display;
- Windows COM-name validation, connection lifecycle, reads, and writes using
  test doubles;
- composition and closure of the Windows display presenter.

These tests validate the software contracts without requiring a real COM port,
board, screen, or Ollama instance.

## Prototype status

Firmware exists for ESP-IDF and initializes the Waveshare board display, draws
a simple face, receives framed commands, and updates a text label. The Python
transport and presenter layers also exist.

The default terminal and tkinter startup paths do not construct the physical
display presenter. This repository's unit tests do not prove flashing,
electrical behavior, display output, or a complete end-to-end assistant-to-
board session on physical hardware. The display path must therefore be treated
as a prototype rather than a supported robot integration.

Experimental firmware variants are kept separate from the validated `main`
baseline. They should not be merged into IA documentation or commits without
explicit hardware validation.

## Outside this repository's current IA scope

The complete mobile robot direction includes locomotion, an expressive head,
vision, arms, sensors, tools, and physical interaction. Its ROB/CAO work is
developed in separate branches of the broader project and is not fully
specified by this IA repository.

No motor, GPIO, PWM, actuator, or safety-critical device is controlled by the
LLM. Future hardware capabilities must remain behind validated high-level
application commands, permissions, and confirmations.

Bioelectrical interfaces are also outside the current implementation. There is
no EGM, EMG, or EEG hardware or signal-processing integration here.
