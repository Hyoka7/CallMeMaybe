"""Orchestrate schema-constrained function-call generation."""
from __future__ import annotations

import json
from typing import Any

from pydantic import PrivateAttr

from src.decoder_errors import DecoderError
from src.model import JsonFunction
from src.states import (
    ParameterKeyState,
    ParameterSeparatorState,
    ParameterValueState,
)
from src.value_generation import ValueGeneration
from src.value_handlers import (
    ValueHandler,
    ValueHandlerRegistry,
)


class ConstrainedDecoder(ValueGeneration):
    """Coordinate function selection, parameter values and JSON output."""

    _value_handlers: ValueHandlerRegistry = PrivateAttr()

    def model_post_init(self, _context: Any) -> None:
        """Install built-in handlers while keeping the registry extensible."""
        del _context
        self._value_handlers = ValueHandlerRegistry.default()

    def register_value_handler(
        self, type_name: str, handler: ValueHandler
    ) -> None:
        """Register a schema value handler for future/custom types."""
        self.value_handlers().register(type_name, handler)

    def value_handlers(self) -> ValueHandlerRegistry:
        """Support lightweight model_construct() instances used by tests."""
        try:
            return self._value_handlers
        except AttributeError:
            self.model_post_init(None)
            return self._value_handlers

    def generate_parameters(
        self,
        structure_prompt: list[int],
        output: list[int],
        function: JsonFunction,
        user_input: str,
    ) -> None:
        """Generate one schema-constrained argument object."""
        self.emit_literal(structure_prompt, output, "{")
        for index, (name, definition) in enumerate(
            function.parameters.items()
        ):
            key_state = ParameterKeyState(name=name)
            self.emit_literal(
                structure_prompt, output, key_state.literal
            )
            value_type = definition["type"]
            value_state = ParameterValueState(type_name=value_type)
            value_state.handler(self.value_handlers())
            if value_type == "string":
                self.emit_literal(structure_prompt, output, " ")
                self.emit_literal(structure_prompt, output, '"')
                regex_kind = None
                if self.is_regex_argument(function, name):
                    regex_kind = self.regex_kind(function, name, user_input)
                value_start = len(structure_prompt)
                if self.is_source_argument(name) and regex_kind is None:
                    value = self.generate_source(structure_prompt, user_input)
                else:
                    value = self.generate_string(
                        structure_prompt, regex_kind, user_input
                    )
                if self.is_replacement_argument(function, name):
                    replacement = self.refine_replacement(
                        value, self.extract_literal_candidates(user_input)
                    )
                    if replacement != value:
                        del structure_prompt[value_start:]
                        structure_prompt.extend(
                            self.model.encode(replacement)[0].tolist()
                        )
                        structure_prompt.append(self.vocabulary.quote)
                        value = replacement
                if regex_kind is not None:
                    refined = self.refine_regex(value)
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
            elif value_type in {"number", "integer"}:
                end_text = (
                    "}" if index + 1 == len(function.parameters) else ","
                )
                output.extend(
                    self.generate_number(
                        structure_prompt,
                        end_text,
                        integer=value_type == "integer",
                    )
                )
            elif value_type == "boolean":
                output.extend(self.generate_boolean(structure_prompt))
            separator = ParameterSeparatorState(
                is_last=index + 1 == len(function.parameters)
            )
            self.emit_literal(
                structure_prompt, output, separator.literal
            )
        if not function.parameters:
            self.emit_literal(structure_prompt, output, "}")

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
        output: list[int] = []
        self.emit_literal(prompt_ids, output, '{"prompt": "')
        prompt_value = json.dumps(user_input, ensure_ascii=False)
        self.emit_literal(prompt_ids, output, prompt_value[1:-1])
        self.emit_literal(prompt_ids, output, '", "name": "')
        name = self.choose_function_name(prompt_ids, list(by_name), output)
        selected = by_name[name]
        self.emit_literal(
            prompt_ids, output, '", "parameters": '
        )
        self.generate_parameters(
            prompt_ids, output, selected, user_input
        )
        self.emit_literal(prompt_ids, output, "}")
        call = json.loads(self.model.decode(output))
        parameters = call.get("parameters")
        if not isinstance(parameters, dict):
            raise DecoderError("Generated parameters are not an object")
        if set(parameters) != set(selected.parameters):
            raise DecoderError("Generated arguments do not match schema")
        return selected, parameters
