# Privacy and security

Fripouille is local-first, but local execution alone is not a complete
security guarantee. This document describes the controls implemented in the
current repository and the limits users should understand.

## Local model and data

The default model client sends requests to Ollama at
`http://localhost:11434` and uses `qwen3.5:9b`. The default application does
not configure a cloud model endpoint.

Tasks, journal entries, and memories are stored in a SQLite database at:

```text
%LOCALAPPDATA%\assistant-ia\assistant_ia.db
```

This location is outside the repository. Database files, environment files,
logs, temporary files, local data, captures, calibration data, and model
directories are excluded by `.gitignore`. No SQLite database is tracked in the
current repository.

The SQLite file is not encrypted by this application. Operating-system account
security, disk encryption, backups, and access to the local Ollama service
remain the user's responsibility.

## Memory policy

Manual memory actions are explicit registered actions. Automatic analysis does
not write a model suggestion directly to SQLite:

1. the analyzer creates a non-persistent candidate;
2. the candidate retains source text and a confidence value;
3. the application compares it with existing memories;
4. a promotion proposal identifies creation, duplication, update, or conflict;
5. the core requests confirmation when required and rechecks the proposal
   before persistence.

Contextual recall is local, read-only, lexically scored, and bounded before it
enters a prompt. Recalled text is non-authoritative context, not an instruction
or an independent truth source.

The active-person context is session-local and minimal. Confirmed profile
facts are persisted locally with an explicit person foreign key, provenance
and confirmation; candidates are not treated as truth. Profile facts are not
yet injected into prompts, and relationships are not implemented.

Memory/person associations are explicit application data, never name matches
or model-selected identifiers. Historical memories remain unassigned. During
automatic contextual recall, the active person can receive their linked
memories and relevant unassigned memories; memories linked only to another
person are excluded in repository-backed retrieval before any prompt is built.

## Action authority

The language model can propose only a name from a closed intent set and
structured text parameters. It cannot call repositories, processes, or
hardware directly.

Execution crosses these application-owned controls:

```text
model proposal
    -> intent schema
    -> registered action
    -> exact parameter validation
    -> business rules
    -> permission policy
    -> confirmation when required
    -> controlled adapter
```

The default permission policy denies unconfigured actions. Windows application
launching requires explicit confirmation, resolves only a built-in allowlist,
rejects shell and script executables, and starts the selected command with
`shell=False`. User text cannot supply an arbitrary executable or free-form
command line.

The provisional tkinter interface has no sensitive-action confirmation handler,
so such actions are denied rather than silently approved.

## Hardware boundary

The LLM has no direct API for GPIO, PWM, motors, serial bytes, or raw screen
operations. Application code constructs and validates high-level display text
commands before a framed transport sends them to prototype firmware.

Any future actuator or bioelectrical interface must preserve the same rule:
model output is a proposal, while deterministic application code owns
validation, permission, confirmation, limits, and the final command.

There is currently no EGM, EMG, or EEG integration in this repository.

## Secrets and publication hygiene

`.env`, `.env.*` except `.env.example`, SQLite files, logs, dumps, local model
directories, captures, and local calibration data are ignored. Secrets should
never be placed in source files, tests, documentation examples, or commits.

Before publication or contribution, inspect both the worktree and staged diff:

```powershell
git status --short
git diff --cached
```

Ignore rules reduce accidental inclusion; they do not revoke a secret that has
already entered Git history. Any exposed credential must be rotated and the
history assessed separately.
