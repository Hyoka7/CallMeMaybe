"""Schema value-handler interface, registry and built-in adapters."""
from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING

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
            regex_kind = decoder._regex_kind(
                function, parameter_name, user_input
            )
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
