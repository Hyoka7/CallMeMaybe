"""JSON fragment states and the function-name token trie."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.token_generation import TokenGeneration
    from src.value_handlers import ValueHandler, ValueHandlerRegistry

END = -1


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

    @property
    def finished(self) -> bool:
        """Whether the fixed fragment has been completely consumed."""
        return not self.remaining

    def consume(self, token_text: str) -> LiteralResult:
        if not self.remaining.startswith(token_text):
            return LiteralResult(False, self.remaining, False)
        remainder = self.remaining[len(token_text):]
        return LiteralResult(True, remainder, not remainder)


@dataclass(frozen=True)
class FunctionNameState:
    """Token-trie state for selecting one function name."""

    choices: tuple[str, ...]

    def build(self, decoder: TokenGeneration) -> TrieNode:
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
