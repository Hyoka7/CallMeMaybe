*This project has been created as part of the 42 curriculum by hfujisad.*

# call me maybe

## Description

`call me maybe` translates natural-language requests into JSON function calls containing the original prompt, selected function name, and typed parameters. It uses Qwen/Qwen3-0.6B through the supplied `llm_sdk` package.

The decoder constrains generation token by token instead of relying on prompting alone, so output follows `functions_definition.json` and contains no extra prose.

The repository is deliberately split into a model-facing layer and a deterministic validation layer. `src/loader.py` validates input files, `src/prompt.py` builds the compiler-style Qwen prompt, `src/constrained_decoder.py` performs decoding, and `src/json_to_file.py` writes the result array. The SDK is treated as an external dependency: the application calls only its public encoding, decoding, vocabulary-path, and logits methods.

## Source layout

- `generation_engine.py`: function-call orchestration and parameter ordering.
- `token_generation.py`: shared model context, fixed JSON fragments and trie selection.
- `regex_generation.py`: regex argument identification, intent and completion rules.
- `value_generation.py`: JSON string, number and boolean generation.
- `value_handlers.py`: handler protocol, registry and built-in adapters.
- `states.py`: immutable generation states and token trie.
- `vocabulary.py`: tokenizer vocabulary and token masks.
- `decoder_errors.py`: decoding exceptions.
- `constrained_decoder.py` and `decoder_core.py`: stable compatibility exports.

The internal generation classes build on token selection, regex support and
value grammars in that order. `ConstrainedDecoder` owns call orchestration.
State and handler modules use type-only decoder imports to avoid circular imports.

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

The tests use small deterministic model doubles for state-level behavior, then exercise the real CLI separately. This keeps edge cases reproducible while preserving an end-to-end path for file loading, model construction, decoding, Pydantic validation, and output writing. Tests intentionally include emptyex3内で、super()関数が利用されていたため、該当箇所以外はokということでこのような結果とさせていただきます。コード自体は動作は問題なく、またそれぞれの実装方法などに理由があって良かったと思います。課題文の書かれ方が正直曖昧な点があるのも理解できますが、super()が「組み込み関数」セクションに含まれていることなどを踏まえ、ここでは使用禁止であると判断させていただきました。int(temp)が2回書かれている点だけ修正お忘れなく。
 strings, quotes, backslashes, common function-name prefixes, and future/custom value types.

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

## Repository layout

```text
.
├── data/input/
│   ├── function_calling_tests.json       # natural-language requests
│   └── functions_definition.json         # callable functions and schemas
├── data/output/
│   └── function_calling_results.json     # generated result array
├── llm_sdk/                              # supplied SDK workspace
├── src/
│   ├── cli.py                            # command-line arguments/default paths
│   ├── constrained_decoder.py            # vocabulary, states, tries, handlers
│   ├── json_to_file.py                   # atomic result serialization
│   ├── loader.py                          # Pydantic-backed input loading
│   ├── main.py                            # application orchestration
│   ├── model.py                           # input and output models
│   └── prompt.py                          # Qwen chat/compiler prompt
├── tests/                                 # deterministic unit tests
├── Makefile
└── pyproject.toml
```

The command-line layer intentionally contains little generation logic. This makes the decoder testable with a small model double and keeps file errors separate from model errors.

## End-to-end walkthrough

For each item in the input array, `src.main.run()` performs these operations:

1. Parse command-line paths and construct `Small_LLM_Model`.
2. Load and validate function definitions and prompts with Pydantic.
3. Build one immutable vocabulary classification from the SDK tokenizer file.
4. Build a compiler-style prompt containing all available function descriptions and the user request.
5. Encode the prompt into the SDK's input-ID representation.
6. Generate the complete JSON call through the constrained decoder.
7. Decode and parse the output, then verify that the selected function's parameter set matches exactly.
8. Append a validated `JsonResult` and write all results as one JSON array.

There is no function execution in this project. The result is a description of the call that a later application could dispatch.

## State transition reference

| State | Allowed content | Next state | Failure condition |
| --- | --- | --- | --- |
| Root open | `{` | Prompt key | no valid token for `{` |
| Prompt key | exact escaped key and colon | Prompt value | token is not a literal prefix |
| Prompt value | exact user prompt, JSON escaped | Name key | altered or unterminated prompt |
| Name key | exact key and colon | Function trie | wrong key or separator |
| Function trie | tokenized supplied names | Parameters key | trie has no valid child |
| Parameters key | exact key and colon | Parameter object | wrong key |
| Parameter object | `{` or `}` for empty schema | Parameter key/value | invalid schema order |
| Parameter key | next schema key only | Value handler | unknown or duplicate key |
| Value handler | type-specific grammar | Separator | incomplete value |
| Separator | `,` except after last value, otherwise `}` | Next key/root close | trailing comma |
| Root close | final `}` | Terminal | any token after terminal |

This table describes the contract independently of the model. The model supplies preferences among valid candidates; it never expands the valid language.

## Token masking in detail

At a decision point, the decoder conceptually executes the following operation:

```python
logits = model.get_logits_from_input_ids(context_ids)
allowed = state.valid_token_ids(vocabulary)
masked = np.full_like(logits, -np.inf)
masked[allowed] = logits[allowed]
token_id = int(np.argmax(masked))
next_state = state.consume(vocabulary.text(token_id))
```

The implementation uses state copies for simulation, so rejecting a candidate never mutates the real context. Both the selected token ID and its decoded text are tracked. This is important for BPE/SentencePiece-style tokens that contain leading whitespace, punctuation, or several characters at once.

The decoder treats an empty candidate set as a hard error. It does not append a quote, brace, or other “repair” token after the model has failed to produce one. This makes failures visible during development and prevents invalid output from reaching the result file.

## Function-definition contract

Each function definition has this shape:

```json
{
  "name": "fn_add_numbers",
  "description": "Add two numbers together and return their sum.",
  "parameters": {
    "a": {"type": "number"},
    "b": {"type": "number"}
  },
  "returns": {"type": "number"}
}
```

The decoder preserves parameter insertion order. Every declared parameter is emitted exactly once. The return schema documents the eventual function result but is not emitted in the call object. Unsupported or malformed definitions are rejected before model generation begins.

## Adding a new value type

New value types should implement the handler contract and register an instance after constructing the decoder:

```python
class DateHandler:
    def generate(self, decoder, prompt, user_input, parameter_name, function):
        # Consume only tokens representing an ISO-8601 date.
        return generate_date_value(decoder, prompt)

decoder.register_value_handler("date", DateHandler())
```

A production handler should also have focused tests for incomplete prefixes, valid termination, escaped content where applicable, and empty candidate sets. The handler must append model-context token IDs consistently with the value it returns. Structural punctuation remains outside the handler, so adding `date` cannot accidentally change comma or brace rules.

## Error handling

The CLI catches interruptions, memory exhaustion, and unexpected exceptions and returns a non-zero exit code with a short message on stderr. Typical actionable errors include:

- `FileNotFoundError`: an input, definition, or tokenizer file is missing;
- `json.JSONDecodeError`: an input file is not valid JSON;
- Pydantic validation errors: required keys or schema types are invalid;
- `UnsupportedTypeError`: a parameter type has no registered handler;
- `NoValidTokenError`: the model vocabulary cannot continue the requested state;
- model download/configuration errors: the configured Qwen model is unavailable.

Output is written only after all prompts have been decoded and validated. A failed run therefore cannot be mistaken for a successful partial result.

## Troubleshooting

**Model download fails.** Confirm network access to Hugging Face or pre-populate the local model cache, then rerun `uv run -m src`. The decoder itself does not download or modify model files.

**Generation is slow.** The first model load is usually the largest fixed cost. Subsequent costs come from logits calls for value and trie decisions. Keep the vocabulary object alive for the whole batch; do not reconstruct it per prompt. Fixed JSON fragments already use a validated deterministic fast path.

**An unsupported type is reported.** Add a handler with `register_value_handler()` before processing definitions, or change the input schema to one of the built-in types. Do not bypass validation by inserting arbitrary values after decoding.

**A regex value is semantically wrong.** Constrained decoding guarantees JSON syntax and the declared string type, not that a natural-language interpretation is correct. Adjust the semantic prompt/classifier policy and add a focused regression test; leave the JSON state machine unchanged.

**The output file is rejected.** Inspect the first reported JSON/Pydantic error and run the unit tests. The final validator is deliberately strict about exact top-level keys, parameter names, and types.

## Accuracy versus structural guarantees

There are two separate quality dimensions:

- Structural correctness is deterministic: invalid JSON tokens, unknown function names, extra keys, missing keys, wrong primitive types, and trailing text are excluded by the decoder.
- Semantic correctness is probabilistic: the model must understand which function and argument value the user intended. Prompt wording, function descriptions, and semantic policies influence this dimension.

This separation is intentional. It lets the project make a strong 100% structural guarantee without claiming that a 0.6B model understands every ambiguous request perfectly.

## Security and robustness considerations

User prompts and function descriptions are data, not executable instructions. The program never evaluates generated text and never calls a selected function. JSON escaping prevents a prompt containing quotes or backslashes from breaking the surrounding object. Pydantic rejects extra fields in input models, while the output validator rejects extra fields in generated calls.

## Reproducibility checklist

Before submitting or reviewing a change:

1. Use a clean virtual environment created by `uv sync`.
2. Run `uv run python -m unittest discover -s tests -v`.
3. Run `make lint` with the required flake8 and mypy flags.
4. Run `uv run -m src` using the default input files.
5. Inspect `data/output/function_calling_results.json` with a JSON parser.
6. Confirm that `LLM_SDK` and its private attributes were not changed or accessed.

## Glossary

- **Token:** A vocabulary item consumed by the model; it may represent multiple characters.
- **Logit:** The model score for one possible next token.
- **Mask:** The set of token IDs permitted by the current state.
- **Trie:** A prefix tree used for constrained function-name selection.
- **Literal state:** A state that consumes one fixed JSON fragment.
- **Value handler:** A registered grammar for one schema type.
- **Terminal state:** The state reached immediately after the final root brace.
