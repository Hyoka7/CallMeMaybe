"""JSON string, number and boolean generation grammars."""
from __future__ import annotations

import json
import re
from typing import cast

import numpy as np

from src.regex_generation import RegexGeneration
from src.states import END

NUMBER_PREFIX = re.compile(
    r"-?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?)?"
)
INTEGER_PREFIX = re.compile(r"-?(?:(?:0|[1-9][0-9]*))?")
NUMBER_COMPLETE = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)


class ValueGeneration(RegexGeneration):
    """Generate typed values on the shared token stream."""

    def generate_string(
        self,
        prompt: list[int],
        regex_kind: str | None = None,
        user_input: str = "",
        limit: int = 48,
    ) -> str:
        """Generate safe content and always close its JSON quote."""
        content = ""
        if regex_kind == "characters":
            content = "["
            prompt.extend(self.model.encode("[")[0].tolist())
        for _ in range(limit):
            logits = np.asarray(
                self.model.get_logits_from_input_ids(prompt),
                dtype=np.float64,
            )
            known_size = len(self.vocabulary.str_mask)
            copy_size = min(known_size, len(logits))
            mask = np.zeros(len(logits), dtype=bool)
            mask[:copy_size] = self.vocabulary.str_mask[:copy_size]
            for token_id, token_text in self.vocabulary.special_tokens.items():
                if token_id < len(mask) and self.literal_prefix(
                    content + token_text, user_input
                ):
                    mask[token_id] = True
            close_mask = np.zeros(len(logits), dtype=bool)
            close_mask[:copy_size] = self.vocabulary.close_mask[:copy_size]
            if not content:
                lead_space = np.zeros(len(logits), dtype=bool)
                lead_space[:copy_size] = self.vocabulary.lead_space[:copy_size]
                mask &= ~lead_space
            if regex_kind is not None or not self.literal_incomplete(
                content, user_input
            ):
                mask |= close_mask
            if regex_kind is not None:
                for token_id in np.flatnonzero(mask):
                    prefix = self.vocabulary.close_prefix[token_id]
                    token_text = (
                        prefix
                        if prefix is not None
                        else self.vocabulary.strs[token_id]
                    )
                    if regex_kind == "exact":
                        proposed = content + token_text
                        candidates = self.extract_literal_candidates(
                            user_input
                        )
                        allowed = (
                            any(value == proposed for value in candidates)
                            if prefix is not None
                            else self.literal_prefix(proposed, user_input)
                        )
                    else:
                        allowed = self.regex_token_allowed(
                            content,
                            token_text,
                            regex_kind,
                            closing=prefix is not None,
                        )
                    if not allowed:
                        mask[token_id] = False
            if not mask.any():
                raise RuntimeError(
                    "No token can continue the requested string"
                )
            chosen = int(np.argmax(np.where(mask, logits, -np.inf)))
            prefix = self.vocabulary.close_prefix[chosen]
            if prefix is not None:
                token_text = self.vocabulary.strs[chosen]
                if token_text and self.literal_prefix(
                    content + token_text, user_input
                ):
                    self.append_string_fragment(prompt, token_text, chosen)
                    content += token_text
                    continue
                proposed_close = content + prefix
                if (
                    regex_kind is None
                    and prefix
                    and self.literal_incomplete(proposed_close, user_input)
                ):
                    prompt.extend(self.model.encode(prefix)[0].tolist())
                    content = proposed_close
                    continue
                content = proposed_close
                prompt.append(self.vocabulary.quote)
                return content
            proposed = content + self.vocabulary.strs[chosen]
            fragment = self.vocabulary.strs[chosen]
            self.append_string_fragment(prompt, fragment, chosen)
            content = proposed
        prompt.append(self.vocabulary.quote)
        return content

    def generate_source(
        self,
        prompt: list[int],
        user_input: str,
        limit: int = 48,
    ) -> str:
        """Select and copy a source span through semantic Trie selection."""
        source_ids = self.model.encode(user_input)[0].tolist()
        if not source_ids:
            prompt.append(self.vocabulary.quote)
            return ""
        units = [
            match.span()
            for match in re.finditer(r"\w+|['\"]|[^\w\s'\"]+", user_input)
        ]
        quote_counts = {'"': 0, "'": 0}
        quote_kinds: list[str | None] = [None] * len(units)
        for index, (start, stop) in enumerate(units):
            quote = user_input[start:stop]
            if quote not in {'"', "'"}:
                continue
            backslashes = 0
            escaped_index = start - 1
            while escaped_index >= 0 and user_input[escaped_index] == "\\":
                backslashes += 1
                escaped_index -= 1
            if backslashes % 2:
                continue
            internal_apostrophe = (
                quote == "'"
                and start > 0
                and stop < len(user_input)
                and user_input[start - 1].isalnum()
                and user_input[stop].isalnum()
            )
            if internal_apostrophe:
                continue
            quote_kinds[index] = quote
            quote_counts[quote] += 1
        for quote, count in quote_counts.items():
            if count % 2:
                raise ValueError(f"Unmatched quote: {quote}")
        choices: set[str] = set()
        for start in range(len(units)):
            for stop in range(start + 1, min(len(units), start + limit) + 1):
                candidate_quotes = [
                    quote
                    for quote in quote_kinds[start:stop]
                    if quote is not None
                ]
                if any(
                    candidate_quotes.count(quote) % 2
                    for quote in ('"', "'")
                ):
                    continue
                choices.add(user_input[units[start][0]:units[stop - 1][1]])
        selection_prompt = (
            "Select the exact source text for the source argument. Return "
            "only the exact substring being operated on, without "
            "instructions, "
            "operation names, replacement text, or explanation.\n"
            f"Request: {user_input}\nSource text:"
        )
        value = self.choose_trie_value(selection_prompt, sorted(choices))
        prompt.extend(self.model.encode(value)[0].tolist())
        prompt.append(self.vocabulary.quote)
        return value

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

    def append_string_fragment(
        self, prompt: list[int], fragment: str, token_id: int
    ) -> None:
        """Append one semantic string fragment using JSON escaping."""
        escaped = json.dumps(fragment, ensure_ascii=False)[1:-1]
        if escaped == fragment:
            prompt.append(token_id)
        else:
            prompt.extend(self.model.encode(escaped)[0].tolist())

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
            if NUMBER_COMPLETE.fullmatch(text) and (
                not integer or re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text)
            ):
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
