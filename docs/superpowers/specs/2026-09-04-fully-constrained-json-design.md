# Fully Constrained JSON Decoder Design

## Goal

Generate the complete result object from the first `{` through the final `}` using a token-validating state machine. Every emitted SDK token must be valid for the current JSON/schema state.

## Architecture

The decoder is a small state machine. Dedicated transition constructors cover the root object, fixed keys/literals, exact prompt, function-name trie, parameter keys, separators, and terminal braces. Value generation is delegated through a registry of type handlers. A candidate token is accepted only when its decoded text can be consumed completely by a copy of the current state.

The registry keeps syntax constraints independent from semantic policies such as regex intent classification. Adding a type means implementing and registering one handler; root-object control flow remains unchanged.

## Contracts

- Output schema is exactly `{ "prompt": string, "name": string, "parameters": object }` with no extra keys.
- `prompt` is the JSON-escaped input prompt and is generated through an exact trie.
- `name` is one of the supplied function names; common prefixes allow both trie terminal and continuation.
- Parameter keys and required order come from the selected function schema.
- Supported built-in value types are `string`, `number`, and `boolean`.
- Unknown types fail before generation with a clear error.
- No post-hoc JSON repair or trailing text is allowed.
- The SDK remains the only model interface; no SDK internals, torch, transformers, or beam search are used.

## Extensibility

`ValueHandler` exposes `initial_state`, `transitions`, `consume`, and `finish` operations. `ValueHandlerRegistry` maps schema type names to handlers. Handlers own only their value grammar and conversion to a Python value; structural states own quotes, commas, keys, and braces.

## Error handling and validation

Zero valid token candidates, an incomplete value at a structural boundary, budget exhaustion, malformed tokenizer metadata, and unsupported schema types raise dedicated decoder errors. On success, the complete token text is parsed with `json.loads` and validated against the result Pydantic model.

## Testing

Tests cover every state transition, exact prompt preservation, function-name prefixes, token chunks crossing state boundaries, string escapes, JSON number grammar, booleans, dynamic parameter schemas, unsupported types, terminal completion, and integration execution with `uv run -m src`.
