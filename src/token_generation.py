"""Fixed-fragment emission and token-trie selection."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model
from src.decoder_errors import NoValidTokenError
from src.states import (
    END, FunctionNameState, LiteralResult, LiteralState, TrieNode,
)
from src.vocabulary import Vocabulary


class TokenGeneration(BaseModel):
    """Shared model context and constrained structural token operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    vocabulary: Vocabulary

    def _append(self, prompt: list[int], output: list[int], text: str) -> None:
        """Tokenize text and append it to prompt and generated output."""
        token_ids = self.model.encode(text)[0].tolist()
        prompt.extend(token_ids)
        output.extend(token_ids)

    def literal_candidates(
        self, state: LiteralState
    ) -> dict[int, LiteralResult]:
        """Return tokens that advance a fixed-fragment state."""
        candidates: dict[int, LiteralResult] = {}
        for token_id, token_text in enumerate(self.vocabulary.strs):
            if not token_text:
                continue
            result = state.consume(token_text)
            if result.valid:
                candidates[token_id] = result
        return candidates

    def _emit_literal_constrained(
        self, prompt: list[int], output: list[int], literal: str
    ) -> None:
        """Emit a fixed fragment using tokens valid for its state."""
        state = LiteralState(literal)
        # Fixed schema fragments normally have one deterministic token path.
        # Validate that path once, avoiding an expensive model call per token.
        encoded = self.model.encode(literal)[0].tolist()
        fast_state = state
        fast_valid = True
        for token_id in encoded:
            if token_id >= len(self.vocabulary.strs):
                fast_valid = False
                break
            result = fast_state.consume(self.vocabulary.strs[token_id])
            if not result.valid:
                fast_valid = False
                break
            fast_state = LiteralState(result.remaining)
        if fast_valid and fast_state.finished:
            prompt.extend(encoded)
            output.extend(encoded)
            return
        while not state.finished:
            candidates = self.literal_candidates(state)
            if not candidates:
                raise NoValidTokenError(
                    "No token can continue fixed JSON fragment "
                    f"{state.remaining!r}"
                )
            logits = np.asarray(self.model.get_logits_from_input_ids(prompt))
            scored = {
                token_id: float(logits[token_id])
                for token_id in candidates
                if token_id < len(logits)
            }
            if not scored:
                raise NoValidTokenError(
                    "Model logits contain no valid vocabulary token"
                )
            chosen = max(scored, key=scored.__getitem__)
            prompt.append(chosen)
            output.append(chosen)
            state = LiteralState(candidates[chosen].remaining)

    def _trie_choice(self, prompt: str, choices: list[str]) -> str:
        """Choose one complete string, allowing terminal prefix nodes."""
        return self._trie_choice_ids(
            self.model.encode(prompt)[0].tolist(), choices
        )

    def _function_name(
        self,
        prompt_ids: list[int],
        function_names: list[str],
        output_ids: list[int],
    ) -> str:
        """Generate a function name through its dedicated trie state."""
        state = FunctionNameState(tuple(function_names))
        state.build(self)
        return self._trie_choice_ids(
            prompt_ids, list(state.choices), output_ids
        )

    def _trie_choice_ids(
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
