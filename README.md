*This project has been created as part of the 42 curriculum by hfujisad.*

# Call Me Maybe

## Description

Call Me Maybe converts natural-language requests into structured JSON function calls. Given a list of available functions and a user prompt, it selects one function and generates values for every parameter declared by that function.

The program does not execute the selected function or answer the request directly. For example, a request to add two numbers produces a function-call description:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

The project uses the supplied `llm_sdk` package with Qwen/Qwen3-0.6B. Because a small language model cannot reliably produce valid JSON by prompting alone, the decoder restricts the model token by token. The model chooses among valid functions and values, while the application controls the JSON structure and validates the completed result.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- Enough memory to load Qwen/Qwen3-0.6B
- Network access on the first run, unless the model is already cached

### Installation

Install the project and its development dependencies:

```bash
make install
```

The equivalent direct command is:

```bash
uv sync
```

The project treats `llm_sdk` as a workspace dependency. The SDK is not modified by this implementation.

### Running the program

Run with the default files:

```bash
make run
```

or:

```bash
uv run python -m src
```

The default paths are:

```text
Function definitions: data/input/functions_definition.json
Prompts:              data/input/function_calling_tests.json
Results:              data/output/function_calling_results.json
```

Use custom paths with command-line options:

```bash
uv run python -m src \
  --functions_definition path/to/functions.json \
  --input path/to/prompts.json \
  --output path/to/results.json
```

During generation, `tqdm` displays the number of completed prompts, elapsed time, estimated remaining time, and average time per prompt. Individual generated calls are not printed to the terminal. After every prompt succeeds, the results are written to the output file and a save-confirmation message is printed.

### Development commands

```bash
make lint
make debug
make clean
```

`make lint` runs both flake8 and mypy.

## Example Usage

Run the bundled example data with the default paths:

```bash
make run
```

This reads function definitions from `data/input/functions_definition.json`,
processes the prompts in `data/input/function_calling_tests.json`, and writes
the generated calls to `data/output/function_calling_results.json`.

To use your own files, provide all three paths explicitly:

```bash
uv run python -m src \
  --functions_definition examples/functions.json \
  --input examples/prompts.json \
  --output examples/results.json
```

For example, a prompt asking for an addition can produce:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

## Input and Output

### Prompt input

The prompt file is a JSON array. Every entry must contain exactly one `prompt` string:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
  {
    "prompt": "Greet shrek"
  }
]
```

### Function definitions

The function-definition file is a JSON array. Each function contains a name, description, parameter schema, and return schema:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

The built-in schema types are:

| Type | Generated JSON value |
| --- | --- |
| `string` | A JSON string with required escaping |
| `number` | A JSON number with a decimal point; an exponent is optional |
| `integer` | A signed JSON integer without a fraction or exponent |
| `boolean` | `true` or `false` |

Every parameter definition and return definition must contain only its `type` field. The return schema documents the eventual function result; it is not included in the generated call.

### Output format

The output is one JSON array containing an entry for every input prompt:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2,
      "b": 3
    }
  }
]
```

Each result contains exactly `prompt`, `name`, and `parameters`. The program writes the file only after the full batch succeeds, so a failed run does not produce a partial new result.

Input JSON objects are checked for duplicate keys while loading. A duplicate key is rejected instead of silently accepting the last value.

## Algorithm Explanation

### Generation pipeline

For each user request, the program performs the following steps:

1. Validate the function definitions and input prompts with Pydantic.
2. Build a reusable classification of the tokenizer vocabulary.
3. Construct a compiler-style prompt containing the available functions and the request.
4. Emit the fixed `prompt` field and its JSON-escaped input value.
5. Select one function name through a token-ID Trie.
6. Emit every schema parameter in definition order.
7. Generate each value with its type-specific grammar.
8. Close the parameter object and root object.
9. Decode the generated token IDs and parse them with `json.loads`.
10. Verify that the generated parameter keys exactly match the selected schema.

The generated object follows this state sequence:

```text
root `{`
  -> fixed `"prompt"` key and exact escaped prompt value
  -> fixed `"name"` key
  -> constrained function-name Trie
  -> fixed `"parameters"` key and object start
  -> parameter key -> typed value -> separator, repeated in schema order
  -> parameter object close
  -> root object close
  -> terminal
```

### Fixed JSON fragments

Keys, colons, commas, braces, and the original prompt are not freely generated by the model. The decoder encodes each complete fixed fragment with the SDK tokenizer and appends those token IDs directly to both buffers. Fixed fragments do not require model logits or a fallback tokenization path.

Two token buffers are maintained:

- The context buffer contains the original model prompt plus everything generated so far.
- The output buffer contains only the JSON call that will be decoded and saved.

Both buffers receive the same generated token IDs, keeping the model context synchronized with the final output.

### Function-name selection

Every available function name is encoded into token IDs and inserted into a `TrieNode` tree. An `END` marker is stored after every complete name.

The marker is necessary when one function name prefixes another. For example, after consuming `fn_add`, the Trie may allow either:

```text
END       -> select fn_add
next ID   -> continue toward fn_add_numbers
```

At each branch, the model supplies logits, but only Trie children and the valid terminal choice remain selectable. A name not present in the function-definition file cannot be generated.

### Typed value generation

`ParameterState` resolves each schema type through the built-in `ValueHandlerRegistry`. `generate_parameters()` delegates value generation to the resolved handler, and each handler calls the shared string, number, integer, or boolean generator.

All string parameters use the same generator. Vocabulary masks allow printable JSON string fragments and closing quotes while rejecting invalid fragments. Leading-space tokens are allowed, including in the first generated fragment. Tokens containing JSON escapes are decoded into their semantic string value for tracking, while the model-selected token itself remains in the context.

Number generation maintains the text produced so far. Vocabulary construction provides separate `num_mask` and `int_mask` arrays, so candidate tokens are first restricted to numeric or integer characters before grammar checks run. A token is valid only if appending it still matches a possible JSON-number prefix. A `number` may terminate only as a complete value containing a decimal point; an `integer` excludes decimal points and exponents.

Boolean generation limits the choice to the token sequences for `true` and `false`.

### String generation

Every string parameter follows the same token-mask and closing-quote logic. Regex and replacement behavior comes from the compiler prompt and the model's logits; the decoder does not parse, normalize, or rewrite their values after generation.

String and numeric generators each allow at most 256 generated tokens. Exceeding the limit raises `RuntimeError` instead of saving a truncated value.

### Final validation

After the root brace is emitted, the output token IDs are decoded and parsed with `json.loads`. The decoder rejects a result if `parameters` is not an object or if its key set differs from the selected function schema. Input and output Pydantic models reject extra fields.

Constraints guarantee syntax, available function names, parameter names, and primitive value types. They cannot guarantee that a small model always understands an ambiguous request correctly.

## Design Decisions

### Separate structure from semantics

Deterministic JSON structure is generated independently from semantic values. This prevents model preferences from altering keys, punctuation, or object boundaries while preserving model-based function and argument selection.

### Split the decoder by responsibility

The decoder is divided into focused modules:

| Module | Responsibility |
| --- | --- |
| `generation_engine.py` | Complete call orchestration and parameter ordering |
| `token_generation.py` | Fixed literal emission and Trie traversal |
| `states.py` | Parameter metadata and function-name Trie nodes |
| `vocabulary.py` | Token text, masks, numeric fragments, and quote metadata |
| `value_handlers.py` | Type-handler protocol and registry |
| `value_generation.py` | String, number, integer, and boolean grammars |


### Keep the SDK unchanged

The application uses only the SDK's documented model construction, encode, decode, tokenizer-path, and logits interfaces. The supplied SDK is not modified, and no internal SDK attributes are accessed.

### Encode fixed literals directly

Calling the model for every brace, key, and separator caused the standard batch to exceed practical limits. Deterministic fragments are encoded and appended directly. Model logits remain responsible for function and value decisions.

### Report failures clearly

The command layer converts `TypeError`, `ValueError`, `OSError`, and `RuntimeError` into readable messages and exit status `1`. `KeyboardInterrupt` returns `130`, and `MemoryError` returns `1` with a focused message. Other exceptions are caught at the CLI boundary and reported with their exception type and message so the program does not terminate with an unhandled traceback.

## Performance Analysis

### Speed

The main costs are model loading, logits computation for semantic choices, and string-value generation. The tokenizer vocabulary is classified once per process and reused for every prompt. Fixed JSON fragments avoid logits calls by being encoded directly.

With the cached Qwen/Qwen3-0.6B model in offline mode, the distributed data batch completed locally in approximately 1 minute 50 seconds. Execution time depends on hardware, model cache state, prompt length, and generated value length. The validation job has a ten-minute job-level timeout.

### Accuracy

Structural correctness and semantic accuracy are separate:

- Structural correctness is deterministic within the supported grammar: the decoder restricts function names, JSON structure, parameter names, and primitive types.
- Semantic accuracy is probabilistic: the 0.6B model may still choose the wrong function or infer an unintended value.

Function descriptions, argument names, the request, and the compiler prompt can influence semantic results.

### Reliability

Malformed input is rejected before generation. User requests are limited to 256 tokens; function names and parameter names to 64 tokens each; descriptions to 256 tokens. Empty token candidate sets and incomplete values raise explicit decoder errors. Output is parsed and checked again before it is accepted. Files are opened through context managers, and a failed batch is not written as a successful partial result.

The first run may require downloading model data. A cached model can be used with offline environment settings, but cache availability is outside the decoder's control.

## Challenges Faced

### Token boundaries

A tokenizer token may contain several characters, leading whitespace, punctuation, JSON escapes, or a closing quote together with preceding text. The vocabulary therefore stores decoded token text, semantic JSON string values, and quote metadata per complete token instead of assuming character-sized tokens.

### Function-name prefixes

Function names can share token prefixes. A simple greedy string comparison could select a shorter name too early. Explicit terminal nodes in the Trie allow completion and continuation to compete at the same prefix.

### Number termination

The decoder must distinguish incomplete prefixes such as `-`, `1.`, and `1e` from complete JSON numbers. Separate prefix and completion expressions allow valid continuation while preventing premature termination.

### JSON string escaping

Quotes and backslashes can make an otherwise correct model value invalid JSON. Vocabulary construction accepts tokens that parse as valid JSON string fragments and stores their semantic values separately. This permits escaped quote and backslash tokens without replacing the token chosen by the model.

### Regex and replacement values

Regex and replacement parameters use ordinary string generation. Short examples in the compiler prompt guide common number, vowel, exact-word, and replacement requests. Accuracy remains model-dependent, and values are not corrected after generation.

### Runtime cost

Obtaining logits for every fixed JSON token made generation too slow because the SDK recomputes the growing context. Encoding fixed literals directly avoids those model calls without modifying the SDK.

Run static checks:

```bash
make lint
```

For validation, run the default batch with the real model and parse `data/output/function_calling_results.json`. The real-model run evaluates semantic accuracy and execution time.

## Testing Strategy

Automated validation is intentionally maintained in GitHub Actions rather than in repository test files. The workflow runs `uv sync --dev`, the required flake8 and mypy checks, a command-line smoke check, and the default Qwen/Qwen3-0.6B batch.

The generated file is then validated as JSON and checked against the input definitions. The validation confirms the prompt count, the exact output keys, the selected function name, the exact parameter names, and the JSON type of every generated value. The job has a ten-minute timeout. Semantic accuracy is evaluated from the generated calls against the prompts when reviewing a workflow run; structural validity is checked automatically.

## Repository Layout

```text
.
├── data/
│   ├── input/
│   │   ├── function_calling_tests.json
│   │   └── functions_definition.json
│   └── output/
│       └── function_calling_results.json
├── llm_sdk/                       # supplied SDK workspace package
├── src/
│   ├── __main__.py                # python -m src entry point
│   ├── cli.py                     # command-line paths
│   ├── generation_engine.py       # complete function-call orchestration
│   ├── json_to_file.py            # JSON result serialization
│   ├── loader.py                  # input loading and validation
│   ├── main.py                    # application lifecycle and progress
│   ├── model.py                   # Pydantic input/output models
│   ├── prompt.py                  # Qwen compiler prompt
│   ├── states.py                  # parameter metadata and Trie nodes
│   ├── token_generation.py        # fixed literals and Trie choices
│   ├── value_generation.py        # primitive JSON value grammars
│   ├── value_handlers.py          # built-in type handlers
│   └── vocabulary.py              # tokenizer-derived token classes
├── Makefile
├── pyproject.toml
├── tokenizer.json                # local tokenizer reference copy
└── README.md
```

### Local tokenizer reference

`tokenizer.json` is a local copy of the Qwen3-0.6B tokenizer definition. It contains the normal vocabulary, token IDs, merge rules, and added special tokens such as `<|im_start|>` and `<|im_end|>`. It is kept for inspecting token boundaries and IDs; the application still obtains the active tokenizer path from `llm_sdk` at runtime.

## Resources

### References

- The project brief: `en.subject.pdf`
- [Python `json` documentation](https://docs.python.org/3/library/json.html)
- [Python `re` documentation](https://docs.python.org/3/library/re.html)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [NumPy documentation](https://numpy.org/doc/)
- [tqdm documentation](https://tqdm.github.io/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Hugging Face Chat templates](https://huggingface.co/docs/transformers/chat_templating)
- The public interface and tokenizer data exposed by the supplied `llm_sdk`

### AI usage

AI assistance was used for:

- discussing the decoder architecture and separation of responsibilities;
- identifying token-boundary, JSON escaping, number termination, regex, and replacement edge cases;
- analyzing performance trade-offs between per-token logits calls and validated fixed-literal output;
- reorganizing modules and drafting documentation.
