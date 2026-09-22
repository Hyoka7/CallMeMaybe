"""Schema value-handler interface, registry and built-in adapters."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, PrivateAttr

from src.model import JsonFunction

if TYPE_CHECKING:
    from src.generation_engine import ConstrainedDecoder


class ValueHandler(Protocol):
    """Interface implemented by one schema value-type grammar."""

    def generate(
        self,
        decoder: ConstrainedDecoder,
        prompt: list[int],
        output: list[int],
        user_input: str,
        parameter_name: str,
        function: JsonFunction,
        is_last: bool,
    ) -> None:
        """Generate one value and append tokens to prompt and output."""


class ValueHandlerRegistry(BaseModel):
    """Extensible mapping from schema type names to value generators."""

    _handlers: dict[str, ValueHandler] = PrivateAttr(default_factory=dict)

    @classmethod
    def default(cls) -> ValueHandlerRegistry:
        """Create a registry populated with the built-in JSON handlers."""
        registry = cls()
        registry.register("string", StringHandler())
        registry.register("number", NumberHandler())
        registry.register("integer", IntegerHandler())
        registry.register("boolean", BooleanHandler())
        return registry

    def register(self, type_name: str, handler: ValueHandler) -> None:
        """Associate a non-empty schema type name with its generator."""
        if not type_name or not type_name.strip():
            raise ValueError("Type name must not be empty")
        self._handlers[type_name] = handler

    def get(self, type_name: str) -> ValueHandler:
        """Return the generator registered for a schema type."""
        try:
            return self._handlers[type_name]
        except KeyError as exc:
            raise RuntimeError(
                f"No value handler registered for type {type_name!r}"
            ) from exc


class StringHandler(BaseModel):
    """Adapter for constrained JSON string generation."""

    def generate(
        self,
        decoder: ConstrainedDecoder,
        prompt: list[int],
        output: list[int],
        user_input: str,
        parameter_name: str,
        function: JsonFunction,
        is_last: bool,
    ) -> None:
        """Generate a string and append its JSON representation."""
        del user_input, parameter_name, function
        decoder.emit_literal(prompt, output, " ")
        decoder.emit_literal(prompt, output, '"')
        end_text = "}" if is_last else ","
        value = decoder.generate_string(prompt, end_text=end_text)
        escaped = json.dumps(value, ensure_ascii=False)[1:-1]
        output.extend(decoder.model.encode(escaped)[0].tolist())
        output.append(decoder.vocabulary.quote)


class NumberHandler(BaseModel):
    """Adapter for constrained JSON number generation."""

    def generate(
        self,
        decoder: ConstrainedDecoder,
        prompt: list[int],
        output: list[int],
        user_input: str,
        parameter_name: str,
        function: JsonFunction,
        is_last: bool,
    ) -> None:
        """Generate a JSON number value."""
        del user_input, parameter_name, function
        end_text = "}" if is_last else ","
        output.extend(decoder.generate_number(prompt, end_text))


class IntegerHandler(BaseModel):
    """Adapter for constrained JSON integer generation."""

    def generate(
        self,
        decoder: ConstrainedDecoder,
        prompt: list[int],
        output: list[int],
        user_input: str,
        parameter_name: str,
        function: JsonFunction,
        is_last: bool,
    ) -> None:
        """Generate a JSON integer value."""
        del user_input, parameter_name, function
        end_text = "}" if is_last else ","
        output.extend(
            decoder.generate_number(prompt, end_text, integer=True)
        )


class BooleanHandler(BaseModel):
    """Adapter for constrained JSON boolean generation."""

    def generate(
        self,
        decoder: ConstrainedDecoder,
        prompt: list[int],
        output: list[int],
        user_input: str,
        parameter_name: str,
        function: JsonFunction,
        is_last: bool,
    ) -> None:
        """Generate a JSON boolean value."""
        del user_input, parameter_name, function, is_last
        output.extend(decoder.generate_boolean(prompt))
