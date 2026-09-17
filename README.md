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
| `number` | A JSON number, including fractional and exponent forms |
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

Keys, colons, commas, braces, and the original prompt are not freely generated by the model. They are passed to `LiteralState`, which stores the remaining text of a required fragment.

For the normal fast path, the decoder:

1. Encodes the complete literal with the SDK tokenizer.
2. Decodes and checks each resulting token against `LiteralState`.
3. Appends the token sequence only if it consumes the exact fragment.

This path avoids an expensive model call for deterministic JSON text. If the encoded sequence cannot be validated, the fallback path examines vocabulary tokens, keeps only tokens whose decoded text advances the literal, obtains model logits, and chooses the highest-scoring valid token. The fallback can choose a tokenization path but cannot change the required text.

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

`ParameterState` resolves each schema type through `ValueHandlerRegistry`. `generate_parameters()` delegates value generation to the resolved handler, and each built-in handler calls the shared string, number, integer, or boolean generator. Custom types use the same handler-based path after registration.

Before generating a string, the model classifies the parameter as `source`, `regex`, `replacement`, or `ordinary`, using the function purpose, all string argument names, the current argument name, and the request. The selected role dispatches to its dedicated path. String values use vocabulary masks for printable content, leading whitespace, closing quotes, and tokens containing quotes or backslashes. Replacement values use the normal string generator and are not rewritten after generation.

Number generation maintains the text produced so far. Vocabulary construction provides separate `num_mask` and `int_mask` arrays, so candidate tokens are first restricted to numeric or integer characters before grammar checks run. A token is valid only if appending it still matches a possible JSON-number prefix. The value may terminate only when it matches a complete number. Integer generation uses the integer mask and an additional integer-only expression, excluding decimal points and exponents.

Boolean generation limits the choice to the token sequences for `true` and `false`.

### String role handling

String role classification is model-driven rather than based on a hard-coded source/input/text argument-name rule. The four roles are `source`, `regex`, `replacement`, and `ordinary`. Source selection uses a semantic Trie choice; quoted source spans are used as bounded candidates when the request provides them, while unquoted requests use contiguous token spans. Regex selection additionally classifies regex kind:

- `characters` for a set of individual characters;
- `exact` for one literal word or text value;
- `general` for repeated categories or other regular-expression structures.

The decoder checks regex syntax with Python's `re` module and stops at a completed reusable pattern. Alternative expressions cannot stop after a trailing `|`. Generated regex and replacement values are not shape-transformed after generation.

Replacement values use the normal string generation path. The prompt asks for the exact text to insert, and the selected value is copied without post-generation rewriting.

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
| `states.py` | Literal, parameter, separator, and Trie states |
| `vocabulary.py` | Token text, masks, numeric fragments, and quote metadata |
| `value_handlers.py` | Type-handler protocol and registry |
| `value_generation.py` | String, number, integer, and boolean grammars |
| `regex_generation.py` | Regex argument roles, intent, and completion |

`constrained_decoder.py` and `decoder_core.py` preserve stable import paths while the implementation remains split by responsibility.

### Keep the SDK unchanged

The application uses only the SDK's public model construction, encode, decode, tokenizer-path, and logits interfaces. The supplied SDK is not modified, and no private SDK attributes are accessed.

### Use a validated fast path for literals

Calling the model for every brace, key, and separator caused the standard batch to exceed practical limits. Deterministic fragments therefore use a tokenizer path that is fully checked against the literal state before being appended. Model logits remain responsible for function and value decisions.

### Report failures clearly

The command layer converts `ValueError`, `OSError`, and `RuntimeError` into readable messages and exit status `1`. `KeyboardInterrupt` returns `130`, and `MemoryError` returns `1` with a focused message. Other exceptions are caught at the CLI boundary and reported with their exception type and message so the program does not terminate with an unhandled traceback.

## Performance Analysis

### Speed

The main costs are model loading, logits computation for semantic choices, and string-value generation. The tokenizer vocabulary is classified once per process and reused for every prompt. Fixed JSON fragments normally avoid logits calls through the validated fast path.

With the cached Qwen/Qwen3-0.6B model in offline mode, the standard 15-prompt batch completed locally in approximately 151 seconds. This is below the project's five-minute target on that machine, but execution time depends on hardware, model cache state, prompt length, and generated value length.

### Accuracy

Structural correctness and semantic accuracy are separate:

- Structural correctness is deterministic within the supported grammar: the decoder restricts function names, JSON structure, parameter names, and primitive types.
- Semantic accuracy is probabilistic: the 0.6B model may still choose the wrong function or infer an unintended value.

Function descriptions, argument names, the role-classification prompt, and the compiler prompt can influence semantic results. The role classifier is cached per function schema, argument, and request.

### Reliability

Malformed input is rejected before generation. Empty token candidate sets and incomplete values raise explicit decoder errors. Output is parsed and checked again before it is accepted. Files are opened through context managers, and a failed batch is not written as a successful partial result.

The first run may require downloading model data. A cached model can be used with offline environment settings, but cache availability is outside the decoder's control.

## Challenges Faced

### Token boundaries

A tokenizer token may contain several characters, leading whitespace, punctuation, or a closing quote together with preceding text. Character-by-character assumptions therefore failed. The solution was to classify decoded vocabulary text and let each state consume complete token strings.

### Function-name prefixes

Function names can share token prefixes. A simple greedy string comparison could select a shorter name too early. Explicit terminal nodes in the Trie allow completion and continuation to compete at the same prefix.

### Number termination

The decoder must distinguish incomplete prefixes such as `-`, `1.`, and `1e` from complete JSON numbers. Separate prefix and completion expressions allow valid continuation while preventing premature termination.

### JSON string escaping

Quotes and backslashes can make an otherwise correct model value invalid JSON. The string generator tracks safe, special, and closing tokens separately and re-encodes escaped fragments when required.

### Regex completion

The model sometimes continued a useful regex with source text, replacement text, or a broad `.*` suffix. Regex intent classification, compile checks, completed-prefix detection, and explicit alternative-boundary handling constrain generation, but the generated value is not rewritten after the fact. Source extraction similarly prefers tokenizer-validated quoted spans so trailing instruction text is not copied into the source argument.

### Replacement values

Replacement is one of the four AI-classified string roles. Candidate selection excludes quoted source text and instruction fragments. Descriptive symbol names are constrained before generation, and the selected replacement value is preserved without shape-changing post-processing.

### Five-minute execution target

Obtaining logits for every fixed JSON token made generation too slow because the SDK recomputes the growing context. Restoring the validated literal fast path reduced the standard batch to approximately two and a half minutes without modifying the SDK.

Run static checks:

```bash
make lint
```

For validation, run the default batch with the real model and parse `data/output/function_calling_results.json`. The real-model run evaluates semantic accuracy and execution time.

## Testing Strategy

Automated validation is intentionally maintained in GitHub Actions rather than in repository test files. The workflow runs `uv sync --dev`, the required flake8 and mypy checks, a command-line smoke check, and the default Qwen/Qwen3-0.6B batch.

The generated file is then validated as JSON and checked against the input definitions. The validation confirms the prompt count, the exact output keys, the selected function name, the exact parameter names, and the JSON type of every generated value. The generation step has a five-minute timeout. Semantic accuracy is evaluated from the generated calls against the prompts when reviewing a workflow run; structural validity is checked automatically.

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
│   ├── constrained_decoder.py     # public compatibility exports
│   ├── decoder_core.py            # internal compatibility exports
│   ├── generation_engine.py       # complete function-call orchestration
│   ├── json_to_file.py            # JSON result serialization
│   ├── loader.py                  # input loading and validation
│   ├── main.py                    # application lifecycle and progress
│   ├── model.py                   # Pydantic input/output models
│   ├── prompt.py                  # Qwen compiler prompt
│   ├── regex_generation.py        # regex and replacement semantics
│   ├── states.py                  # generation states and Trie nodes
│   ├── token_generation.py        # fixed literals and Trie choices
│   ├── value_generation.py        # primitive JSON value grammars
│   ├── value_handlers.py          # extensible type registry
│   └── vocabulary.py              # tokenizer-derived token classes
├── Makefile
├── pyproject.toml
└── README.md
```

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
