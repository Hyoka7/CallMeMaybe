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
    str_values: tuple[str | None, ...]
    str_mask: NDArray[np.bool_]
    int_mask: NDArray[np.bool_]
    num_mask: NDArray[np.bool_]
    lead_space: NDArray[np.bool_]
    close_mask: NDArray[np.bool_]
    close_prefix: tuple[str | None, ...]
    close_suffix: tuple[str | None, ...]
    quote: int

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
        string_values: list[str | None] = [None] * vocab_size
        string_mask = np.zeros(vocab_size, dtype=bool)
        int_mask = np.zeros(vocab_size, dtype=bool)
        num_mask = np.zeros(vocab_size, dtype=bool)
        lead_space = np.zeros(vocab_size, dtype=bool)
        close_mask = np.zeros(vocab_size, dtype=bool)
        close_prefix: list[str | None] = [None] * vocab_size
        close_suffix: list[str | None] = [None] * vocab_size
        for token_id in raw_vocab.values():
            if not isinstance(token_id, int):
                continue
            text = model.decode([token_id])
            strings[token_id] = text
            lead_space[token_id] = bool(text and text[0].isspace())
            try:
                value = json.loads(f'"{text}"')
            except json.JSONDecodeError:
                value = None
            if (
                text
                and isinstance(value, str)
                and all(
                    char.isprintable() and char != "\ufffd"
                    for char in value
                )
            ):
                string_mask[token_id] = True
                string_values[token_id] = value
            if text.endswith('"'):
                prefix = text[:-1]
                if all(
                    char.isprintable()
                    and char not in {'"', "\\", "\ufffd"}
                    for char in prefix
                ):
                    close_mask[token_id] = True
                    close_prefix[token_id] = prefix
            if text.startswith('"'):
                suffix = text[1:]
                if all(
                    char.isprintable() and char != "\ufffd"
                    for char in suffix
                ):
                    close_suffix[token_id] = suffix
            number_text = text
            if number_text and all(
                char in "-+.eE0123456789" for char in number_text
            ):
                num_mask[token_id] = True
            if number_text and all(
                char in "-0123456789" for char in number_text
            ):
                int_mask[token_id] = True
        quote_ids = model.encode('"')[0].tolist()
        if len(quote_ids) != 1:
            raise RuntimeError("Closing quote must be one token")
        if (
            not string_mask.any()
            or not close_mask.any()
            or not int_mask.any()
            or not num_mask.any()
        ):
            raise RuntimeError(
                "Could not derive token classes from vocabulary"
            )
        return cls(
            strs=tuple(strings),
            str_values=tuple(string_values),
            str_mask=string_mask,
            int_mask=int_mask,
            num_mask=num_mask,
            lead_space=lead_space,
            close_mask=close_mask,
            close_prefix=tuple(close_prefix),
            close_suffix=tuple(close_suffix),
            quote=quote_ids[0],
        )
