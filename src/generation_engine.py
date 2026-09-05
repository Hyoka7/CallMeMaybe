"""Orchestrate schema-constrained function-call generation."""
from __future__ import annotations

import json
from typing import Any

from pydantic import PrivateAttr

from src.model import JsonFunction
from src.states import (
    ParameterKeyState, ParameterSeparatorState, ParameterValueState,
)
from src.value_generation import ValueGeneration
from src.value_handlers import (
    ValueHandler, ValueHandlerRegistry,
    _StringHandler, _NumberHandler, _BooleanHandler,
)


class ConstrainedDecoder(ValueGeneration):
    """Coordinate function selection, parameter values and JSON output."""

    _value_handlers: ValueHandlerRegistry = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Install built-in handlers while keeping the registry extensible."""
        del __context
        registry = ValueHandlerRegistry()
        registry.register("string", _StringHandler())
        registry.register("number", _NumberHandler())
        registry.register("boolean", _BooleanHandler())
        self._value_handlers = registry

    def register_value_handler(
        self, type_name: str, handler: ValueHandler
    ) -> None:
        """Register a schema value handler for future/custom types."""
        self._ensure_value_handlers().register(type_name, handler)

    def _ensure_value_handlers(self) -> ValueHandlerRegistry:
        """Support lightweight model_construct() instances used by tests."""
        try:
            return self._value_handlers
        except AttributeError:
            self.model_post_init(None)
            return self._value_handlers

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
        output: list[int] = []
        self._emit_literal_constrained(prompt_ids, output, '{"prompt": "')
        prompt_value = json.dumps(user_input, ensure_ascii=False)
        self._emit_literal_constrained(prompt_ids, output, prompt_value[1:-1])
        self._emit_literal_constrained(prompt_ids, output, '", "name": "')
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
