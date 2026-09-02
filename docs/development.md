# Developing Fripouille

This document describes the human-facing development workflow for the current
repository. `AGENTS.md` remains the operational instruction file for coding
agents and tools; it is not replaced by this guide.

## Environment

The supported development baseline is:

- Windows 10;
- Python 3.14 or newer;
- PowerShell;
- Ollama running locally;
- `qwen3.5:9b`, the default configured model.

The Python package uses a `src` layout and currently has no declared
third-party runtime dependencies. Tests should not require a running Ollama
service, physical hardware, or a real display.

## Installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --editable .
```

Prepare the model once if it is not already installed:

```powershell
ollama pull qwen3.5:9b
```

Ollama's local service must be available at `http://localhost:11434` for real
conversation turns. The test suite replaces external services with controlled
doubles.

## Running the assistant

```powershell
python -m assistant_ia
```

Available options:

```powershell
python -m assistant_ia --gui
python -m assistant_ia --debug
python -m assistant_ia --gui --debug
```

The terminal is the historical interface and supports interactive
confirmations. The tkinter GUI is provisional; sensitive actions are denied by
default there because it does not yet provide a confirmation handler.

## Validation

Run a focused module while developing:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime
```

Before committing a behavioral change, run the complete suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
git diff --check
git status --short
```

The current baseline contains 561 passing tests. The number is descriptive,
not a fixed target: new behavior should add or adapt focused tests.

## Design principles

- Preserve the flow `interface -> AssistantRuntime -> AssistantCore ->
  intelligence / actions`.
- Let the LLM interpret and propose; keep validation and authority in the
  application.
- Never expose raw GPIO, PWM, motor, shell, SQL, or serial control to the LLM.
- Keep stable identity separate from memory, people, relationships, learning,
  internal state, and roles.
- Reuse existing domain services and repositories instead of duplicating
  behavior.
- Prefer dependency injection and deterministic unit-test doubles at external
  boundaries.
- Add or update focused tests for every behavior change.
- Keep the terminal useful as the historical interface and a diagnostic tool.

## Git and milestone conventions

Use focused commits and preserve `FRP-IA` milestone names in documentation,
tests, and milestone commits. Do not rewrite milestone history merely to make
old commit messages uniform.

Hardware and firmware changes require an explicit hardware scope. Experimental
firmware work must not be folded into IA commits accidentally. Before a
commit, inspect the staged diff and exclude SQLite databases, captures, local
calibration, models, logs, and temporary files.

## Documentation map

- [architecture.md](architecture.md) explains system responsibilities.
- [hardware.md](hardware.md) defines the current hardware boundary.
- [privacy-and-security.md](privacy-and-security.md) records data and authority
  guarantees.
- [READING_GUIDE.md](READING_GUIDE.md) provides the detailed pedagogical code
  tour.
- [CODEX_MAP.md](CODEX_MAP.md) gives a short file-routing map.
