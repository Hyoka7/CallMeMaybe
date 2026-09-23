"""Fixed-fragment emission and token-trie selection."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model
from src.states import (
    END,
    TrieNode,
)
from src.vocabulary import Vocabulary


class TokenGeneration(BaseModel):
    """Shared model context and constrained structural token operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    vocabulary: Vocabulary

    def emit_literal(
        self,
        prompt: list[int],
        output: list[int],
        literal: str,
    ) -> None:
        """Encode and append one fixed JSON fragment."""
        encoded = self.model.encode(literal)[0].tolist()
        prompt.extend(encoded)
        output.extend(encoded)

    def choose_function_name(
        self,
        prompt_ids: list[int],
        function_names: list[str],
        output_ids: list[int],
    ) -> str:
        """Generate a function name through the candidate token trie."""
        return self.choose_trie_token_ids(
            prompt_ids, function_names, output_ids
        )

    def choose_trie_token_ids(
        self,
        prompt_ids: list[int],
        choices: list[str],
        output_ids: list[int] | None = None,
    ) -> str:
        """Choose a trie value while continuing an existing generation."""
        root = TrieNode()
        for choice in choices:
            root.insert(self.model.encode(choice)[0].tolist(), choice)
        end_token = self.model.encode('"')[0].tolist()[0]
        node = root
        while True:
            candidates = {
                end_token if token_id == END else token_id: token_id
                for token_id in node.children
            }
            if len(candidates) == 1:
                chosen = next(iter(candidates.values()))
            else:
                logits = self.model.get_logits_from_input_ids(prompt_ids)
                chosen = candidates[max(candidates, key=logits.__getitem__)]
            node = node.children[chosen]
            if chosen == END:
                if node.value is None:
                    raise RuntimeError("Trie ended without a value")
                return node.value
            prompt_ids.append(chosen)
            if output_ids is not None:
                output_ids.append(chosen)
