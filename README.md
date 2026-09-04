*This project has been created as part of the 42 curriculum by hfujisad.*

# call me maybe

## Description

`call me maybe` translates natural-language requests into JSON function calls containing the original prompt, selected function name, and typed parameters. It uses Qwen/Qwen3-0.6B through the supplied `llm_sdk` package.

The decoder constrains generation token by token instead of relying on prompting alone, so output follows `functions_definition.json` and contains no extra prose.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m src
```

Custom files:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Common Makefile commands are `make install`, `make run`, `make debug`, `make lint`, and `make clean`.

## Input and output

The input test file contains objects with a `prompt` string. The function-definition file contains function names, descriptions, parameter schemas, and return types.

Output is a JSON array with exactly `prompt`, `name`, and `parameters`:

```json
[{"prompt":"What is the sum of 2 and 3?","name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}]
```

## Algorithm: constrained decoding

The tokenizer vocabulary is loaded once through the SDK. Dedicated states generate `{`, JSON keys, separators, values, and the final `}`. Each candidate token is simulated against the current state; invalid candidates are masked out before selecting the highest logit. Fixed fragments use a validated tokenization fast path.

Function names are selected by an LLM-driven token-ID trie. Trie terminal markers coexist with child nodes, so names sharing prefixes are supported. The selected schema dispatches to registered value handlers for strings, JSON numbers, and booleans. Strings enforce escaping and closure; regex semantic handling is kept separate from JSON syntax. The complete response is parsed with `json.loads` and validated with Pydantic.

## Design decisions

- Structural states and typed value handlers are independent.
- `ValueHandlerRegistry` allows future types without changing the root state machine.
- Function selection uses constrained model logits, not keyword heuristics.
- The original prompt is emitted as an exact JSON-escaped value.
- Only public `llm_sdk` methods are used; `LLM_SDK` is not modified.

## Performance and reliability

Token masking prevents malformed JSON, extra keys, missing required parameters, and trailing prose. Deterministic structural token paths avoid needless model calls. The target is 90%+ function/argument accuracy, 100% parseable schema-compliant JSON, and completion within five minutes on standard hardware.

## Challenges faced

Token boundaries, shared function-name prefixes, and regex termination were the main challenges. Token-ID tries and literal-state simulation address boundary and prefix cases; the string handler validates escaping and stops only at a valid closing quote.

## Testing strategy

Tests cover trie terminal/continuation behavior, literal states, token candidates, string escaping, handler registration, unsupported types, JSON writing, and command error handling.

```bash
uv run python -m unittest discover -s tests -v
make lint
uv run -m src
```

## Resources and AI usage

- [Python `json` documentation](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)
- `en.subject.pdf` (project brief)
- The supplied `llm_sdk` public API

AI assistance was used for architecture discussion, token-boundary and termination debugging, test drafting, and documentation. Suggestions were reviewed, adapted to the project constraints, and verified with local tests. AI did not modify `LLM_SDK`.
