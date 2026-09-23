"""Orchestrate schema-constrained function-call generation."""
from __future__ import annotations

import json
from typing import Any

from pydantic import PrivateAttr

from src.model import JsonFunction
from src.states import (
    ParameterState,
)
from src.value_generation import ValueGeneration
from src.value_handlers import (
    ValueHandlerRegistry,
)


class ConstrainedDecoder(ValueGeneration):
    """Coordinate function selection, parameter values and JSON output."""

    _value_handlers: ValueHandlerRegistry = PrivateAttr()

    def model_post_init(self, _context: Any) -> None:
        """Install built-in handlers while keeping the registry extensible."""
        del _context
        self._value_handlers = ValueHandlerRegistry.default()

    def value_handlers(self) -> ValueHandlerRegistry:
        """Support lightweight instances created with model_construct()."""
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
            parameter_state = ParameterState(
                name=name,
                type_name=definition["type"],
                is_last=index + 1 == len(function.parameters),
            )
            self.emit_literal(
                structure_prompt, output, parameter_state.key_literal
            )
            parameter_state.handler(self.value_handlers()).generate(
                self,
                structure_prompt,
                output,
                user_input,
                name,
                function,
                parameter_state.is_last,
            )
            self.emit_literal(
                structure_prompt, output, parameter_state.separator_literal
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
            raise TypeError("Generated parameters are not an object")
        if set(parameters) != set(selected.parameters):
            raise RuntimeError("Generated arguments do not match schema")
        return selected, parameters
