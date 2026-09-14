"""JSON fragment states and the function-name token trie."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.value_handlers import ValueHandler, ValueHandlerRegistry

END = -1


class LiteralResult(BaseModel):
    """Result of consuming one token's decoded text in a literal state."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    remaining: str
    finished: bool


class LiteralState(BaseModel):
    """State for a fixed JSON fragment, including token-boundary crossing."""

    model_config = ConfigDict(frozen=True)

    remaining: str

    @property
    def finished(self) -> bool:
        """Whether the fixed fragment has been completely consumed."""
        return not self.remaining

    def consume(self, token_text: str) -> LiteralResult:
        """Consume token text when it prefixes the remaining literal."""
        if not self.remaining.startswith(token_text):
            return LiteralResult(
                valid=False,
                remaining=self.remaining,
                finished=False,
            )
        remainder = self.remaining[len(token_text):]
        return LiteralResult(
            valid=True,
            remaining=remainder,
            finished=not remainder,
        )


class ParameterState(BaseModel):
    """State for one parameter's key, type validation and separator."""

    model_config = ConfigDict(frozen=True)

    name: str
    type_name: str
    is_last: bool

    @property
    def key_literal(self) -> str:
        """Return the JSON-encoded key and colon."""
        return json.dumps(self.name, ensure_ascii=False) + ":"

    @property
    def separator_literal(self) -> str:
        """Return an object close or comma for the current position."""
        return "}" if self.is_last else ","

    def handler(self, registry: ValueHandlerRegistry) -> ValueHandler:
        """Resolve the registered generator for this value type."""
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
