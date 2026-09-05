"""Build token classes and masks from the SDK tokenizer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model


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
                and all(
                    char.isprintable() and char != "\ufffd"
                    for char in text
                )
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
