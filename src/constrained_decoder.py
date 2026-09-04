from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from llm_sdk import Small_LLM_Model
from src.model import JsonFunction

END = -1
NUMBER_END_MARGIN = 3.0
NUMBER_PREFIX = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?"
)
NUMBER_COMPLETE = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)


class DecoderError(RuntimeError):
    """Base error for an invalid constrained-generation transition."""


class NoValidTokenError(DecoderError):
    """Raised when no vocabulary token can continue the current state."""


class UnsupportedTypeError(DecoderError):
    """Raised when a schema type has no registered value handler."""


@dataclass(frozen=True)
class LiteralResult:
    """Result of consuming one token's decoded text in a literal state."""

    valid: bool
    remaining: str
    finished: bool


@dataclass(frozen=True)
class LiteralState:
    """State for a fixed JSON fragment, including token-boundary crossing."""

    remaining: str

    def consume(self, token_text: str) -> LiteralResult:
        if not self.remaining.startswith(token_text):
            return LiteralResult(False, self.remaining, False)
        remainder = self.remaining[len(token_text):]
        return LiteralResult(True, remainder, not remainder)


@dataclass(frozen=True)
class FunctionNameState:
    """Token-trie state for selecting one function name."""

    choices: tuple[str, ...]

    def build(self, decoder: ConstrainedDecoder) -> TrieNode:
        root = TrieNode()
        for choice in self.choices:
            root.insert(decoder.model.encode(choice)[0].tolist(), choice)
        return root


@dataclass(frozen=True)
class ParameterKeyState:
    """State describing the next schema parameter key to emit."""

    name: str

    @property
    def literal(self) -> str:
        return json.dumps(self.name, ensure_ascii=False) + ": "


@dataclass(frozen=True)
class ParameterSeparatorState:
    """State selecting the only valid separator after a parameter value."""

    is_last: bool

    @property
    def literal(self) -> str:
        return "}" if self.is_last else ","


@dataclass(frozen=True)
class ParameterValueState:
    """Dispatch state for one schema-declared parameter value type."""

    type_name: str

    def handler(self, registry: ValueHandlerRegistry) -> ValueHandler:
        return registry.get(self.type_name)


class ValueHandler(Protocol):
    """Interface implemented by one schema value-type grammar."""

    def generate(
        self,
        decoder: ConstrainedDecoder,
        prompt: list[int],
        user_input: str,
        parameter_name: str,
        function: JsonFunction,
    ) -> Any:
        """Generate one JSON value and append its token IDs to prompt."""


class ValueHandlerRegistry:
    """Extensible mapping from schema type names to value generators."""

    def __init__(self) -> None:
        self._handlers: dict[str, ValueHandler] = {}

    def register(self, type_name: str, handler: ValueHandler) -> None:
        if not type_name or not type_name.strip():
            raise ValueError("Type name must not be empty")
        self._handlers[type_name] = handler

    def get(self, type_name: str) -> ValueHandler:
        try:
            return self._handlers[type_name]
        except KeyError as exc:
            raise UnsupportedTypeError(
                f"No value handler registered for type {type_name!r}"
            ) from exc


class _StringHandler:
    def generate(
        self,
        decoder: ConstrainedDecoder,
        prompt: list[int],
        user_input: str,
        parameter_name: str,
        function: JsonFunction,
    ) -> Any:
        regex_kind = None
        if decoder._is_regex_argument(function, parameter_name):
            regex_kind = decoder._regex_kind(function, parameter_name, user_input)
        decoder._append(prompt, [], '"')
        return decoder._string(prompt, regex_kind, user_input)


class _NumberHandler:
    def generate(
        self,
        decoder: ConstrainedDecoder,
        prompt: list[int],
        user_input: str,
        parameter_name: str,
        function: JsonFunction,
    ) -> Any:
        del user_input, parameter_name, function
        return decoder._number(prompt, "}")


class _BooleanHandler:
    def generate(
        self,
        decoder: ConstrainedDecoder,
        prompt: list[int],
        user_input: str,
        parameter_name: str,
        function: JsonFunction,
    ) -> Any:
        del user_input, parameter_name, function
        return decoder._boolean(prompt)


class TrieNode(BaseModel):
    """One token-ID trie node; END marks a complete candidate."""

    children: dict[int, TrieNode] = Field(default_factory=dict)
    value: str | None = None

    def insert(self, token_ids: list[int], value: str) -> None:
        """Insert one tokenized candidate and its terminal value."""
        node = self
        for token_id in token_ids:
            node = node.children.setdefault(token_id, TrieNode())
        node.children.setdefault(END, TrieNode()).value = value


class Vocabulary(BaseModel):
    """Token classes derived once from the SDK vocabulary file."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    strs: tuple[str, ...]
    str_mask: NDArray[np.bool_]
    lead_space: NDArray[np.bool_]
    close_mask: NDArray[np.bool_]
    close_prefix: tuple[str | None, ...]
    quote: int
    number_tokens: dict[int, str]
    special_tokens: dict[int, str]

    @classmethod
    def from_sdk(cls, model: Small_LLM_Model) -> Vocabulary:
        """Build constrained token classes from the SDK tokenizer file."""
        path = Path(model.get_path_to_tokenizer_file())
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        raw_vocab = data.get("model", {}).get("vocab")
        if not isinstance(raw_vocab, dict):
            raise TypeError("Tokenizer file has no model.vocab mapping")
        token_ids = [
            token_id for token_id in raw_vocab.values()
            if isinstance(token_id, int)
        ]
        token_ids.extend(
            item["id"] for item in data.get("added_tokens", [])
            if isinstance(item, dict) and isinstance(item.get("id"), int)
        )
        vocab_size = max(token_ids) + 1
        strings = [""] * vocab_size
        string_mask = np.zeros(vocab_size, dtype=bool)
        lead_space = np.zeros(vocab_size, dtype=bool)
        close_mask = np.zeros(vocab_size, dtype=bool)
        close_prefix: list[str | None] = [None] * vocab_size
        number_ids: dict[int, str] = {}
        special_ids: dict[int, str] = {}
        for token_id in raw_vocab.values():
            if not isinstance(token_id, int):
                continue
            text = model.decode([token_id])
            strings[token_id] = text
            lead_space[token_id] = bool(text and text[0].isspace())
            if text and all(
                char.isprintable() and char not in {'"', "\\", "\ufffd"}
                for char in text
            ):
                string_mask[token_id] = True
            elif (
                text
                and any(char in text for char in ('"', "\\"))
                and all(char.isprintable() and char != "\ufffd" for char in text)
            ):
                special_ids[token_id] = text
            if text.endswith('"'):
                prefix = text[:-1]
                if all(
                    char.isprintable()
                    and char not in {'"', "\\", "\ufffd"}
                    for char in prefix
                ):
                    close_mask[token_id] = True
                    close_prefix[token_id] = prefix
            if text and all(char in "-+.eE0123456789" for char in text):
                number_ids[token_id] = text
        quote_ids = model.encode('"')[0].tolist()
        if len(quote_ids) != 1:
            raise ValueError("Closing quote must be one token")
        if not string_mask.any() or not close_mask.any() or not number_ids:
            raise ValueError("Could not derive token classes from vocabulary")
        return cls(
            strs=tuple(strings),
            str_mask=string_mask,
            lead_space=lead_space,
            close_mask=close_mask,
            close_prefix=tuple(close_prefix),
            quote=quote_ids[0],
            number_tokens=number_ids,
            special_tokens=special_ids,
        )


class ConstrainedDecoder(BaseModel):
    """Greedy LLM decoding restricted by tries and JSON value types."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    vocabulary: Vocabulary
    _regex_roles: dict[tuple[str, tuple[str, ...]], str | None] = PrivateAttr(
        default_factory=dict
    )
    _value_handlers: ValueHandlerRegistry = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Install built-in handlers while keeping the registry extensible."""
        del __context
        registry = ValueHandlerRegistry()
        registry.register("string", _StringHandler())
        registry.register("number", _NumberHandler())
        registry.register("boolean", _BooleanHandler())
        self._value_handlers = registry

    def register_value_handler(self, type_name: str, handler: ValueHandler) -> None:
        """Register a schema value handler for future/custom types."""
        self._ensure_value_handlers().register(type_name, handler)

    def _ensure_value_handlers(self) -> ValueHandlerRegistry:
        """Support lightweight model_construct() instances used by tests."""
        try:
            return self._value_handlers
        except AttributeError:
            self.model_post_init(None)
            return self._value_handlers

    def _append(self, prompt: list[int], output: list[int], text: str) -> None:
        """Tokenize text and append it to prompt and generated output."""
        token_ids = self.model.encode(text)[0].tolist()
        prompt.extend(token_ids)
        output.extend(token_ids)

    def literal_candidates(self, state: LiteralState) -> dict[int, LiteralResult]:
        """Return vocabulary tokens that fully advance a fixed-fragment state."""
        candidates: dict[int, LiteralResult] = {}
        for token_id, token_text in enumerate(self.vocabulary.strs):
            if not token_text:
                continue
            result = state.consume(token_text)
            if result.valid:
                candidates[token_id] = result
        return candidates

    def _emit_literal_constrained(
        self, prompt: list[int], output: list[int], literal: str
    ) -> None:
        """Emit a fixed fragment using only tokens valid for its literal state."""
        state = LiteralState(literal)
        while not state.finished:
            candidates = self.literal_candidates(state)
            if not candidates:
                raise NoValidTokenError(
                    f"No token can continue fixed JSON fragment {state.remaining!r}"
                )
            logits = np.asarray(self.model.get_logits_from_input_ids(prompt))
            scored = {
                token_id: float(logits[token_id])
                for token_id in candidates
                if token_id < len(logits)
            }
            if not scored:
                raise NoValidTokenError("Model logits contain no valid vocabulary token")
            chosen = max(scored, key=scored.__getitem__)
            token_text = self.vocabulary.strs[chosen]
            prompt.append(chosen)
            output.append(chosen)
            state = LiteralState(candidates[chosen].remaining)

    def _trie_choice(self, prompt: str, choices: list[str]) -> str:
        """Choose one complete string, allowing terminal prefix nodes."""
        return self._trie_choice_ids(
            self.model.encode(prompt)[0].tolist(), choices
        )

    def _function_name(
        self,
        prompt_ids: list[int],
        function_names: list[str],
        output_ids: list[int],
    ) -> str:
        """Generate a function name through its dedicated trie state."""
        state = FunctionNameState(tuple(function_names))
        state.build(self)
        return self._trie_choice_ids(prompt_ids, list(state.choices), output_ids)

    def _trie_choice_ids(
        self,
        prompt_ids: list[int],
        choices: list[str],
        output_ids: list[int] | None = None,
    ) -> str:
        """Choose a trie value while continuing an existing generation."""
        root = TrieNode()
        for choice in choices:
            root.insert(self.model.encode(choice)[0].tolist(), choice)
        end_token = self.model.encode('"')[0].tolist()[0]
        node = root
        while True:
            candidates = {
                end_token if token_id == END else token_id: token_id
                for token_id in node.children
            }
            if len(candidates) == 1:
                chosen = next(iter(candidates.values()))
            else:
                logits = self.model.get_logits_from_input_ids(prompt_ids)
                chosen = candidates[max(candidates, key=logits.__getitem__)]
            node = node.children[chosen]
            if chosen == END:
                if node.value is None:
                    raise RuntimeError("Trie ended without a value")
                return node.value
            prompt_ids.append(chosen)
            if output_ids is not None:
                output_ids.append(chosen)

    def _is_regex_argument(
        self, function: JsonFunction, parameter_name: str
    ) -> bool:
        """Choose the pattern argument by comparing the complete schema."""
        string_names = tuple(
            name for name, definition in function.parameters.items()
            if definition["type"] == "string"
        )
        cache_key = (function.description, string_names)
        description = function.description.lower()
        if not any(
            marker in description for marker in ("regex", "regular expression")
        ):
            self._regex_roles[cache_key] = None
            return False
        if cache_key in self._regex_roles:
            return self._regex_roles[cache_key] == parameter_name
        prompt = (
            "Choose which string argument itself stores the reusable regular "
            "expression used for matching. Do not choose source text, "
            "replacement text, names, or other direct values. Choose NONE if "
            "this function has no regex-pattern argument.\n"
            f"Function purpose: {function.description}\n"
            f"String arguments: {', '.join(string_names)}\n"
            "Pattern argument: \""
        )
        selected = self._trie_choice(prompt, list(string_names) + ["NONE"])
        self._regex_roles[cache_key] = None if selected == "NONE" else selected
        return selected == parameter_name

    def _regex_kind(
        self,
        function: JsonFunction,
        parameter_name: str,
        user_input: str,
    ) -> str:
        """Classify the requested pattern as character, exact, or general."""
        choices = ("characters", "exact", "general")
        choice_ids = {
            choice: self.model.encode(choice)[0].tolist()[0]
            for choice in choices
        }

        def scores(request: str) -> dict[str, float]:
            """Return baseline-adjustable intent logits for one request."""
            prompt = (
                "Classify regex matching intent as characters for alternative "
                "individual characters, exact for one exact literal word or "
                "text, or general for a repeated category or other "
                "structure.\n"
                "Examples: individual vowels = characters; exact word bird = "
                "exact; numeric sequences = general.\n"
                f"Function purpose: {function.description}\n"
                f"Target regex argument: {parameter_name}\n"
                f"Request: {request}\nIntent: "
            )
            ids = self.model.encode(prompt)[0].tolist()
            logits = self.model.get_logits_from_input_ids(ids)
            return {
                choice: float(logits[token_id])
                for choice, token_id in choice_ids.items()
            }

        actual = scores(user_input)
        baseline = scores("unspecified matching intent")
        return max(
            choices,
            key=lambda choice: actual[choice] - baseline[choice],
        )

    @staticmethod
    def _regex_complete(pattern: str) -> bool:
        """Return whether a minimal reusable regex has reached a safe end."""
        if not pattern or pattern.endswith(("\\", "|", "(", "[", "{")):
            return False
        try:
            re.compile(pattern)
        except re.error:
            return False
        if not any(character in pattern for character in "[](){}+*?\\.^$"):
            return True
        return pattern.endswith(("]", ")", "}", "+", "*", "?", "$"))

    @classmethod
    def _completed_regex_prefix(cls, pattern: str) -> str | None:
        """Find a completed structural regex inside a multi-text token."""
        for length in range(1, len(pattern) + 1):
            prefix = pattern[:length]
            structural = any(char in prefix for char in "[](){}+*?\\.^$")
            if structural and cls._regex_complete(prefix):
                return prefix
        return None

    def _refine_regex(self, pattern: str) -> str:
        """Enforce the prompt's shortest-pattern invariant."""
        if not pattern.endswith(".*") or len(pattern) <= 2:
            return pattern
        return pattern[:-2]

    def _string(
        self,
        prompt: list[int],
        regex_kind: str | None = None,
        user_input: str = "",
        limit: int = 48,
    ) -> str:
        """Generate safe content and always close its JSON quote."""
        content = ""
        if regex_kind == "characters":
            content = "["
            prompt.extend(self.model.encode("[")[0].tolist())
        for _ in range(limit):
            logits = np.asarray(
                self.model.get_logits_from_input_ids(prompt),
                dtype=np.float64,
            )
            known_size = len(self.vocabulary.str_mask)
            copy_size = min(known_size, len(logits))
            mask = np.zeros(len(logits), dtype=bool)
            mask[:copy_size] = self.vocabulary.str_mask[:copy_size]
            for token_id, token_text in self.vocabulary.special_tokens.items():
                if token_id < len(mask) and self._literal_prefix(
                    content + token_text, user_input
                ):
                    mask[token_id] = True
            close_mask = np.zeros(len(logits), dtype=bool)
            close_mask[:copy_size] = self.vocabulary.close_mask[:copy_size]
            if not content:
                lead_space = np.zeros(len(logits), dtype=bool)
                lead_space[:copy_size] = self.vocabulary.lead_space[:copy_size]
                mask &= ~lead_space
            if regex_kind is not None or not self._literal_incomplete(
                content, user_input
            ):
                mask |= close_mask
            chosen = int(np.argmax(np.where(mask, logits, -np.inf)))
            prefix = self.vocabulary.close_prefix[chosen]
            if prefix is not None:
                token_text = self.vocabulary.strs[chosen]
                if token_text and self._literal_prefix(
                    content + token_text, user_input
                ):
                    self._append_string_fragment(prompt, token_text, chosen)
                    content += token_text
                    continue
                proposed_close = content + prefix
                if (
                    regex_kind is None
                    and prefix
                    and self._literal_incomplete(proposed_close, user_input)
                ):
                    prompt.extend(self.model.encode(prefix)[0].tolist())
                    content = proposed_close
                    continue
                content = proposed_close
                if regex_kind == "characters" and not content.endswith("]"):
                    content += "]"
                    prompt.extend(self.model.encode("]")[0].tolist())
                prompt.append(self.vocabulary.quote)
                return content
            proposed = content + self.vocabulary.strs[chosen]
            complete: str | None = None
            if regex_kind == "exact":
                match = re.search(r"[\[\](){}+*?\\.^$|]", proposed)
                complete = proposed[:match.start()] if match else proposed
            elif regex_kind in {"characters", "general"}:
                complete = self._completed_regex_prefix(proposed)
            if complete is not None:
                suffix = complete[len(content):]
                prompt.extend(self.model.encode(suffix)[0].tolist())
                prompt.append(self.vocabulary.quote)
                return complete
            fragment = self.vocabulary.strs[chosen]
            self._append_string_fragment(prompt, fragment, chosen)
            content = proposed
        if regex_kind == "characters" and not content.endswith("]"):
            content += "]"
            prompt.extend(self.model.encode("]")[0].tolist())
        prompt.append(self.vocabulary.quote)
        return content

    @staticmethod
    def _literal_candidates(user_input: str) -> list[str]:
        """Extract likely literal argument values from a user request."""
        quoted = [
            match[0] or match[1]
            for match in re.findall(
                r"'([^']*)'|\"([^\"]*)\"", user_input
            )
        ]
        if quoted:
            return quoted
        return re.findall(r"[A-Za-z0-9_]+", user_input)

    @classmethod
    def _literal_prefix(cls, content: str, user_input: str) -> bool:
        """Check whether content prefixes a requested literal value."""
        return any(
            candidate.startswith(content)
            for candidate in cls._literal_candidates(user_input)
        )

    @classmethod
    def _literal_incomplete(
        cls, content: str, user_input: str
    ) -> bool:
        """Check if content is a strict prefix of a requested text span."""
        return any(
            candidate.startswith(content) and candidate != content
            for candidate in cls._literal_candidates(user_input)
        )

    def _append_string_fragment(
        self, prompt: list[int], fragment: str, token_id: int
    ) -> None:
        """Append one semantic string fragment using JSON escaping."""
        escaped = json.dumps(fragment, ensure_ascii=False)[1:-1]
        if escaped == fragment:
            prompt.append(token_id)
        else:
            prompt.extend(self.model.encode(escaped)[0].tolist())

    def _number(
        self, prompt: list[int], end_text: str, limit: int = 24
    ) -> list[int]:
        """Generate a terminating JSON number token by token."""
        output: list[int] = []
        text = ""
        end_token = self.model.encode(end_text)[0].tolist()[0]
        for _ in range(limit):
            logits = self.model.get_logits_from_input_ids(prompt)
            valid = {
                token_id
                for token_id, token_text
                in self.vocabulary.number_tokens.items()
                if NUMBER_PREFIX.fullmatch(text + token_text)
            }
            if not valid:
                raise RuntimeError("No valid number token")
            candidates = set(valid)
            if NUMBER_COMPLETE.fullmatch(text):
                candidates.add(END)
                best_number = max(valid, key=logits.__getitem__)
                if (
                    logits[best_number] - logits[end_token]
                    <= NUMBER_END_MARGIN
                ):
                    return output
            chosen = max(
                candidates,
                key=lambda token_id: (
                    logits[end_token] if token_id == END
                    else logits[token_id]
                ),
            )
            if chosen == END:
                return output
            output.append(chosen)
            prompt.append(chosen)
            text += self.vocabulary.number_tokens[chosen]
        raise RuntimeError("Number value did not terminate")

    def _boolean(self, prompt: list[int]) -> list[int]:
        """Choose one JSON boolean literal from model logits."""
        choices = {
            value: self.model.encode(value)[0].tolist()
            for value in ("true", "false")
        }
        first_logits = self.model.get_logits_from_input_ids(prompt)
        value = max(choices, key=lambda item: first_logits[choices[item][0]])
        token_ids = cast(list[int], choices[value])
        prompt.extend(token_ids)
        return token_ids

    def _generate_parameters(
        self,
        structure_prompt: list[int],
        output: list[int],
        function: JsonFunction,
        user_input: str,
    ) -> None:
        """Generate one schema-constrained argument object."""
        self._emit_literal_constrained(structure_prompt, output, "{")
        for index, (name, definition) in enumerate(
            function.parameters.items()
        ):
            key_state = ParameterKeyState(name)
            self._emit_literal_constrained(
                structure_prompt, output, key_state.literal
            )
            value_type = definition["type"]
            value_state = ParameterValueState(value_type)
            value_state.handler(self._ensure_value_handlers())
            if value_type == "string":
                self._emit_literal_constrained(structure_prompt, output, '"')
                regex_kind = None
                if self._is_regex_argument(function, name):
                    regex_kind = self._regex_kind(
                        function, name, user_input
                    )
                value_start = len(structure_prompt)
                value = self._string(
                    structure_prompt, regex_kind, user_input
                )
                if regex_kind is not None:
                    refined = self._refine_regex(value)
                    if refined != value:
                        del structure_prompt[value_start:]
                        structure_prompt.extend(
                            self.model.encode(refined)[0].tolist()
                        )
                        structure_prompt.append(self.vocabulary.quote)
                        value = refined
                escaped = json.dumps(value, ensure_ascii=False)[1:-1]
                output.extend(self.model.encode(escaped)[0].tolist())
                output.append(self.vocabulary.quote)
            elif value_type == "number":
                end_text = (
                    "}" if index + 1 == len(function.parameters) else ","
                )
                output.extend(self._number(structure_prompt, end_text))
            elif value_type == "boolean":
                output.extend(self._boolean(structure_prompt))
            separator = ParameterSeparatorState(
                is_last=index + 1 == len(function.parameters)
            )
            self._emit_literal_constrained(
                structure_prompt, output, separator.literal
            )
        if not function.parameters:
            self._emit_literal_constrained(structure_prompt, output, "}")

    def generate_call(
        self,
        prompt: str,
        functions: list[JsonFunction],
        user_input: str,
    ) -> tuple[JsonFunction, dict[str, Any]]:
        """Generate a function name and its arguments on one token stream."""
        if not functions:
            raise ValueError("No functions available")
        by_name = {function.name: function for function in functions}
        prompt_ids = self.model.encode(prompt)[0].tolist()
        prompt_value = json.dumps(user_input, ensure_ascii=False)
        output: list[int] = self.model.encode(
            '{"prompt": ' + prompt_value + ', "name": "'
        )[0].tolist()
        # The chat prompt already contains the opening object/key/quote.
        self._append(
            prompt_ids,
            [],
            prompt_value[1:] + ', "name": "',
        )
        name = self._function_name(prompt_ids, list(by_name), output)
        selected = by_name[name]
        self._emit_literal_constrained(
            prompt_ids, output, '", "parameters": '
        )
        self._generate_parameters(
            prompt_ids, output, selected, user_input
        )
        self._emit_literal_constrained(prompt_ids, output, "}")
        call = json.loads(self.model.decode(output))
        parameters = call.get("parameters")
        if not isinstance(parameters, dict):
            raise TypeError("Generated parameters are not an object")
        if set(parameters) != set(selected.parameters):
            raise RuntimeError("Generated arguments do not match schema")
        return selected, parameters
