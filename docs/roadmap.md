# Fripouille roadmap

The FRP-IA roadmap tracks the cognitive and application architecture. It does
not imply that the complete robot, mechanical design, or every future
interface belongs to this repository.

Statuses below are based on code and Git history in `main`. Early foundation
commits predate consistent `FRP-IA-*` commit prefixes, so FRP-IA-00 and
FRP-IA-01 are grouped here by their delivered scope rather than retroactively
renaming history.

## Completed baseline

| Milestone | Scope evidenced in `main` | Status |
| --- | --- | --- |
| FRP-IA-00 | Python project foundation, package structure, terminal, core, and local Ollama connection | Completed |
| FRP-IA-01 | Structured intents, SQLite persistence, tasks, journal, manual memories, secure Windows launch, stable identity, and conversational foundations | Completed |
| FRP-IA-02 | `AssistantRuntime`, interface/core separation, terminal routing, and tested presenter boundaries | Completed |
| FRP-IA-02B | Lightweight provisional tkinter chat interface and project map | Completed prototype |
| FRP-IA-03 | Contextual memory series: provenance and schema v3, retrieval, bounded history, prompt injection, automatic candidates, controlled promotion and correction | Completed |
| FRP-IA-04 | Persistent people, controlled profile facts, person-scoped memories, bounded relationships, observations and private social context | Completed |
| FRP-IA-05 | Inspectable behavioral experiences, sourced lesson candidates and application-owned person scope; no automatic consolidation | Completed |

The hardware display commits in the same history establish a prototype track,
not proof that a complete robot has been delivered.

## Current position

The current branch includes FRP-IA-05A and 05B. Behavioral experiences retain
context, objective, strategy, result, minimal evaluation and structured
provenance. Lesson candidates remain explicitly non-confirmed and retain their
exact source-experience links. Neither is created automatically from a turn,
used as a permission, or injected into model prompts.

## Planned milestones

| Milestone | Direction | Status |
| --- | --- | --- |
| FRP-IA-06 | Experience feedback and explicit evaluation of outcomes | Future |
| FRP-IA-07 | Consolidation of observations and learned material | Future |
| FRP-IA-08 | Internal state modeled separately from identity and conversation | Future |
| FRP-IA-09 | Voice input and output interfaces | Future |
| FRP-IA-10 | Face and expressive behavior beyond the current provisional visuals | Future |
| FRP-IA-11 | Social vision and perception with explicit privacy boundaries | Future |
| FRP-IA-12 | Contextual roles and professions | Future |
| FRP-IA-13 | Integration of the preceding cognitive domains | Future |

These future entries describe intended research directions, not implemented
features or fixed delivery promises. Each milestone should preserve the
authority rule that a language model proposes while application code validates
and authorizes effects.

## Parallel embodiment directions

The complete mobile robot is developed as a separate robotics direction,
including ROB/CAO branches in the broader project. This repository currently
contains only the IA-side display and transport prototype plus associated
ESP32 firmware.

Bioelectrical signals are a separate experimental direction. EGM, EMG, EEG,
and related interfaces are not integrated here and have no completed FRP-IA
milestone.
