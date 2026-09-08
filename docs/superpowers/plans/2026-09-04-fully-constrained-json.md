# Fully Constrained JSON Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended) to implement this plan task-by-task.

**Goal:** Replace the provisional call generator with a complete `{`-to-`}` token-constrained JSON state machine.

**Architecture:** Dedicated structural state constructors drive one shared logits loop. Typed value grammars are registered handlers so future types do not alter root control flow.

**Tech Stack:** Python 3.10+, numpy, Pydantic, public `llm_sdk.Small_LLM_Model` API, unittest/pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-04-fully-constrained-json-design.md`

## Global Constraints

- Do not edit `LLM_SDK`.
- Do not import torch or transformers and do not access SDK private attributes.
- Every emitted token must pass the current state transition validator.
- Preserve regex semantic handling separately from JSON syntax.
- Run `uv run -m src` after implementation changes.

### Task 1: Define state and handler interfaces

**Files:** Modify `src/constrained_decoder.py`; Test `tests/test_constrained_decoder.py`.

- [ ] Add failing tests for structural state consumption, handler registration, and unsupported types.
- [ ] Run the focused tests and confirm the new tests fail for missing interfaces.
- [ ] Add immutable state objects, `ValueHandler` protocol/base class, registry, and dedicated decoder errors.
- [ ] Run focused tests and confirm they pass.

### Task 2: Implement token transition engine

**Files:** Modify `src/constrained_decoder.py`; Test `tests/test_constrained_decoder.py`.

- [ ] Add a failing test where one token crosses a literal-state boundary and only fully consumable candidates survive.
- [ ] Implement candidate simulation, masked logits selection, budget checks, and exact token/output synchronization.
- [ ] Run focused tests and confirm they pass.

### Task 3: Implement complete root JSON flow

**Files:** Modify `src/constrained_decoder.py`, `src/main.py`; Test `tests/test_main.py`, `tests/test_constrained_decoder.py`.

- [ ] Add failing tests for exact prompt, function trie prefixes, ordered parameter keys, separators, and final brace.
- [ ] Implement `gen_open_object`, key/literal states, exact prompt state, function-name state, parameter object states, and terminal state.
- [ ] Run all unit tests and confirm they pass.

### Task 4: Implement extensible typed handlers

**Files:** Modify `src/constrained_decoder.py`, `src/model.py`; Test `tests/test_constrained_decoder.py`.

- [ ] Add failing tests for string escapes, complete/incomplete numbers, booleans, and custom handler registration.
- [ ] Implement built-in handlers and dispatch through the registry without changing structural flow.
- [ ] Run all unit tests and confirm they pass.

### Task 5: Integration and cleanup

**Files:** Modify `src/prompt.py`, `src/json_to_file.py` only if required; Test `tests/test_main.py`.

- [ ] Add an integration assertion that output parses exactly and contains no trailing text.
- [ ] Run `uv run python -m unittest discover -s tests -v`.
- [ ] Run `uv run -m src` with the project inputs and inspect the output JSON.
- [ ] Run `git diff --check` and confirm no prohibited SDK/torch/private API changes.
