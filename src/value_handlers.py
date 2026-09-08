"""Schema value-handler interface, registry and built-in adapters."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, PrivateAttr

from src.decoder_errors import UnsupportedTypeError
from src.model import JsonFunction

if TYPE_CHECKING:
    from src.generation_engine import ConstrainedDecoder


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


class ValueHandlerRegistry(BaseModel):
    """Extensible mapping from schema type names to value generators."""

    _handlers: dict[str, ValueHandler] = PrivateAttr(default_factory=dict)

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
            raise UnsupportedTypeError(
                f"No value handler registered for type {type_name!r}"
            ) from exc


class _StringHandler(BaseModel):
    """Adapter for constrained JSON string generation."""

    def generate(
        self,
        decoder: ConstrainedDecoder,
        prompt: list[int],
        user_input: str,
        parameter_name: str,
        function: JsonFunction,
    ) -> Any:
        """Generate a string, including regex handling when applicable."""
        regex_kind = None
        if decoder._is_regex_argument(function, parameter_name):
            regex_kind = decoder._regex_kind(
                function, parameter_name, user_input
            )
        decoder._append(prompt, [], '"')
        return decoder._string(prompt, regex_kind, user_input)


class _NumberHandler(BaseModel):
    """Adapter for constrained JSON number generation."""

    def generate(
        self,
        decoder: ConstrainedDecoder,
        prompt: list[int],
        user_input: str,
        parameter_name: str,
        function: JsonFunction,
    ) -> Any:
        """Generate a JSON number value."""
        del user_input, parameter_name, function
        return decoder._number(prompt, "}")


class _IntegerHandler(BaseModel):
    """Adapter for constrained JSON integer generation."""

    def generate(
        self,
        decoder: ConstrainedDecoder,
        prompt: list[int],
        user_input: str,
        parameter_name: str,
        function: JsonFunction,
    ) -> Any:
        """Generate a JSON integer value."""
        del user_input, parameter_name, function
        return decoder._number(prompt, "}", integer=True)


class _BooleanHandler(BaseModel):
    """Adapter for constrained JSON boolean generation."""

    def generate(
        self,
        decoder: ConstrainedDecoder,
        prompt: list[int],
        user_input: str,
        parameter_name: str,
        function: JsonFunction,
    ) -> Any:
        """Generate a JSON boolean value."""
        del user_input, parameter_name, function
        return decoder._boolean(prompt)
