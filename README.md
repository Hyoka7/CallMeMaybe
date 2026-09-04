*This project has been created as part of the 42 curriculum by hfujisad.*

# call me maybe

## Description

`call me maybe` translates natural-language requests into JSON function calls containing the original prompt, selected function name, and typed parameters. It uses Qwen/Qwen3-0.6B through the supplied `llm_sdk` package.

The decoder constrains generation token by token instead of relying on prompting alone, so output follows `functions_definition.json` and contains no extra prose.

The repository is deliberately split into a model-facing layer and a deterministic validation layer. `src/loader.py` validates input files, `src/prompt.py` builds the compiler-style Qwen prompt, `src/constrained_decoder.py` performs decoding, and `src/json_to_file.py` writes the result array. The SDK is treated as an external dependency: the application calls only its public encoding, decoding, vocabulary-path, and logits methods.

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

The first run may download Qwen/Qwen3-0.6B. A network or model-cache failure is reported as a readable command error; it does not produce a partial output file.

## Input and output

The input test file contains objects with a `prompt` string. The function-definition file contains function names, descriptions, parameter schemas, and return types.

Output is a JSON array with exactly `prompt`, `name`, and `parameters`:

```json
[{"prompt":"What is the sum of 2 and 3?","name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}]
```

## Algorithm: constrained decoding

### 1. Vocabulary preparation

`Vocabulary.from_sdk()` reads the tokenizer JSON returned by `get_path_to_tokenizer_file()`. It records decoded text for each token and derives masks for safe string fragments, leading-space fragments, quote/close fragments, numeric fragments, and special escaped fragments. This work is done once per process rather than once per prompt.

### 2. State-driven output

The response is generated through this explicit path:

```text
root `{`
  -> exact key `"prompt"` and exact prompt value
  -> exact key `"name"`
  -> function-name trie
  -> exact key `"parameters"` and parameters `{`
  -> parameter key -> typed value -> separator (repeat)
  -> parameters `}` -> root `}` -> terminal
```

`LiteralState` represents a fixed fragment that still has characters to consume. For every candidate vocabulary token, the decoder simulates its decoded text against a copy of the state. A token is valid only if it consumes a prefix without violating the fragment. The selected token is then appended to both the model context and output buffer, keeping them synchronized.

When a fixed fragment has a single valid tokenizer path, that path is validated once and appended without repeated model calls. If tokenization is ambiguous, the decoder obtains logits, masks every invalid ID to negative infinity, and chooses the highest remaining score. Thus the fast path is an optimization of the same constraint, not a separate unvalidated output path.

### 3. Function selection

Each supplied function name is tokenized into a `TrieNode` tree. A terminal marker is stored at the end of every name. At a shared node, both the terminal marker and child token remain candidates; this handles names such as `fn_add` and `fn_add_numbers` without guessing or prefix truncation. The LLM therefore chooses the function, while the trie guarantees that the choice belongs to the supplied schema.

### 4. Typed values

`ParameterValueState` resolves a schema type through `ValueHandlerRegistry`. The built-in handlers enforce JSON string, number, and boolean grammars. String generation rejects unsafe quotes and backslashes unless they form valid JSON escapes and always requires a closing quote. Number generation follows JSON's sign/integer/fraction/exponent rules and cannot terminate after an incomplete prefix such as `-`, `1.`, or `1e`. Boolean generation follows the `true`/`false` literal trie.

The registry is intentionally open for extension:

```python
decoder.register_value_handler("date", DateHandler())
```

A handler owns only value syntax and conversion. It does not emit commas, keys, or braces; those remain controlled by the structural state machine.

### 5. Final validation

Generation ends only after the terminal root brace. The complete text is parsed with `json.loads`; the result model rejects missing or extra top-level keys, and the selected function schema checks parameter names and types. Any empty candidate set, unsupported type, malformed tokenizer metadata, or incomplete value raises a specific decoder error instead of silently repairing output.

## Design decisions

- Structural states and typed value handlers are independent.
- `ValueHandlerRegistry` allows future types without changing the root state machine.
- Function selection uses constrained model logits, not keyword heuristics.
- The original prompt is emitted as an exact JSON-escaped value.
- Only public `llm_sdk` methods are used; `LLM_SDK` is not modified.
- Fixed literals are deterministic but still pass through the same state validator.
- Semantic interpretation (for example, whether a string is a regex) is kept outside JSON grammar so syntax guarantees remain reusable.

## Performance and reliability

Token masking prevents malformed JSON, extra keys, missing required parameters, and trailing prose. Deterministic structural token paths avoid needless model calls, while value decisions still use model logits. The dominant cost is the number of value-generation steps and vocabulary scans; vocabulary masks are cached and fixed fragments use the fast path. The target is 90%+ function/argument accuracy, 100% parseable schema-compliant JSON, and completion within five minutes on standard hardware. Accuracy depends on the model and prompt quality; constraints guarantee structure and types, not the truth of a semantically incorrect answer.

## Challenges faced

Token boundaries, shared function-name prefixes, regex termination, and reliable stopping were the main challenges. A token may contain punctuation and part of a key at once, so character-by-character assumptions were replaced with whole-token state simulation. Trie terminal markers solve prefix collisions. The string handler separates safe content, escaped content, and closing-quote candidates; regex completion is checked without allowing matched source text or replacement text to leak into the pattern. Finally, explicit terminal states prevent the model from continuing after the final `}`.

## Testing strategy

Tests cover trie terminal/continuation behavior, literal states, token candidates, string escaping, handler registration, unsupported types, JSON writing, and command error handling.

The tests use small deterministic model doubles for state-level behavior, then exercise the real CLI separately. This keeps edge cases reproducible while preserving an end-to-end path for file loading, model construction, decoding, Pydantic validation, and output writing. Tests intentionally include empty strings, quotes, backslashes, common function-name prefixes, and future/custom value types.

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
