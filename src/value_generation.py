"""JSON string, number and boolean generation grammars."""
from __future__ import annotations

import re
from typing import cast

import numpy as np

from src.states import END
from src.token_generation import TokenGeneration

NUMBER_PREFIX = re.compile(
    r"-?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?)?"
)
INTEGER_PREFIX = re.compile(r"-?(?:(?:0|[1-9][0-9]*))?")
NUMBER_COMPLETE = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)


class ValueGeneration(TokenGeneration):
    """Generate typed values on the shared token stream."""

    def generate_string(
        self,
        prompt: list[int],
        user_input: str = "",
        end_text: str = ",",
        limit: int = 48,
    ) -> str:
        """Generate safe content and always close its JSON quote."""
        content = ""
        virtual_close_ids = [
            token_id
            for token_id, suffix in enumerate(
                self.vocabulary.close_suffix
            )
            if suffix is not None
            and (not suffix or suffix.startswith(end_text))
        ]
        for _ in range(limit):
            logits = np.asarray(
                self.model.get_logits_from_input_ids(prompt),
                dtype=np.float64,
            )
            known_size = len(self.vocabulary.str_mask)
            copy_size = min(known_size, len(logits))
            mask = np.zeros(len(logits), dtype=bool)
            mask[:copy_size] = self.vocabulary.str_mask[:copy_size]
            close_mask = np.zeros(len(logits), dtype=bool)
            close_mask[:copy_size] = self.vocabulary.close_mask[:copy_size]
            if not content:
                lead_space = np.zeros(len(logits), dtype=bool)
                lead_space[:copy_size] = self.vocabulary.lead_space[:copy_size]
                mask &= ~lead_space
            can_close = not self.literal_incomplete(content, user_input)
            if can_close:
                mask |= close_mask
            if not mask.any():
                raise RuntimeError(
                    "No token can continue the requested string"
                )
            chosen = int(np.argmax(np.where(mask, logits, -np.inf)))
            if can_close:
                close_ids = [
                    token_id for token_id in virtual_close_ids
                    if token_id < len(logits)
                ]
                if close_ids:
                    close_id = max(
                        close_ids, key=logits.__getitem__
                    )
                    if logits[close_id] > logits[chosen]:
                        prompt.append(self.vocabulary.quote)
                        return content
            prefix = self.vocabulary.close_prefix[chosen]
            if prefix is not None:
                proposed_close = content + prefix
                if (
                    prefix
                    and self.literal_incomplete(proposed_close, user_input)
                ):
                    prompt.extend(self.model.encode(prefix)[0].tolist())
                    content = proposed_close
                    continue
                content = proposed_close
                prompt.append(self.vocabulary.quote)
                return content
            fragment = self.vocabulary.str_values[chosen]
            if fragment is None:
                raise RuntimeError("Invalid JSON string token")
            prompt.append(chosen)
            content += fragment
        prompt.append(self.vocabulary.quote)
        return content

    @staticmethod
    def extract_literal_candidates(user_input: str) -> list[str]:
        """Extract likely literal argument values from a user request."""
        quoted = [
            match[0] or match[1]
            for match in re.findall(
                r"'([^']*)'|\"([^\"]*)\"", user_input
            )
        ]
        if quoted:
            return quoted
        return re.findall(r"[A-Za-z0-9_]+", user_input)

    @classmethod
    def literal_prefix(cls, content: str, user_input: str) -> bool:
        """Check whether content prefixes a requested literal value."""
        return any(
            candidate.startswith(content)
            for candidate in cls.extract_literal_candidates(user_input)
        )

    @classmethod
    def literal_incomplete(
        cls, content: str, user_input: str
    ) -> bool:
        """Check if content is a strict prefix of a requested text span."""
        return any(
            candidate.startswith(content) and candidate != content
            for candidate in cls.extract_literal_candidates(user_input)
        )

    def generate_number(
        self, prompt: list[int], end_text: str, limit: int = 24,
        integer: bool = False,
    ) -> list[int]:
        """Generate a terminating JSON number token by token."""
        output: list[int] = []
        text = ""
        end_token = self.model.encode(end_text)[0].tolist()[0]
        for _ in range(limit):
            logits = self.model.get_logits_from_input_ids(prompt)
            base_mask = (
                self.vocabulary.int_mask
                if integer
                else self.vocabulary.num_mask
            )
            mask = np.zeros(len(logits), dtype=bool)
            copy_size = min(len(mask), len(base_mask))
            mask[:copy_size] = base_mask[:copy_size]
            valid = {
                int(token_id)
                for token_id in np.flatnonzero(mask)
                for token_text in (self.vocabulary.strs[int(token_id)],)
                if NUMBER_PREFIX.fullmatch(
                    text + (token_text.lstrip() if not text else token_text)
                )
                and (
                    not integer
                    or INTEGER_PREFIX.fullmatch(
                        text + (
                            token_text.lstrip() if not text else token_text
                        )
                    )
                )
            }
            if not valid:
                raise RuntimeError("No valid number token")
            candidates = set(valid)
            if self.number_can_end(text, integer):
                candidates.add(END)
            chosen = max(
                candidates,
                key=lambda token_id: (
                    logits[end_token] if token_id == END
                    else logits[token_id],
                    token_id == END,
                ),
            )
            if chosen == END:
                return output
            output.append(chosen)
            prompt.append(chosen)
            token_text = self.vocabulary.strs[chosen]
            text += token_text.lstrip() if not text else token_text
        raise RuntimeError("Number value did not terminate")

    @staticmethod
    def number_can_end(text: str, integer: bool) -> bool:
        """Check whether a generated numeric value has its required form."""
        if integer:
            return re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text) is not None
        if NUMBER_COMPLETE.fullmatch(text) is None:
            return False
        mantissa = re.split(r"[eE]", text, maxsplit=1)[0]
        return "." in mantissa

    def generate_boolean(self, prompt: list[int]) -> list[int]:
        """Choose one JSON boolean literal from model logits."""
        choices = {
            value: self.model.encode(f" {value}")[0].tolist()
            for value in ("true", "false")
        }
        first_logits = self.model.get_logits_from_input_ids(prompt)
        value = max(choices, key=lambda item: first_logits[choices[item][0]])
        token_ids = cast(list[int], choices[value])
        prompt.extend(token_ids)
        return token_ids
