"""JSON string, number and boolean generation grammars."""
from __future__ import annotations

import json
import re
from typing import cast

import numpy as np

from src.regex_generation import RegexGeneration
from src.states import END

NUMBER_END_MARGIN = 3.0
NUMBER_PREFIX = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?"
)
NUMBER_COMPLETE = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)


class ValueGeneration(RegexGeneration):
    """Generate typed values on the shared token stream."""

    def _string(
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
                if token_id < len(mask) and self._literal_prefix(
                    content + token_text, user_input
                ):
                    mask[token_id] = True
            close_mask = np.zeros(len(logits), dtype=bool)
            close_mask[:copy_size] = self.vocabulary.close_mask[:copy_size]
            if not content:
                lead_space = np.zeros(len(logits), dtype=bool)
                lead_space[:copy_size] = self.vocabulary.lead_space[:copy_size]
                mask &= ~lead_space
            if regex_kind is not None or not self._literal_incomplete(
                content, user_input
            ):
                mask |= close_mask
            chosen = int(np.argmax(np.where(mask, logits, -np.inf)))
            prefix = self.vocabulary.close_prefix[chosen]
            if prefix is not None:
                token_text = self.vocabulary.strs[chosen]
                if token_text and self._literal_prefix(
                    content + token_text, user_input
                ):
                    self._append_string_fragment(prompt, token_text, chosen)
                    content += token_text
                    continue
                proposed_close = content + prefix
                if (
                    regex_kind is None
                    and prefix
                    and self._literal_incomplete(proposed_close, user_input)
                ):
                    prompt.extend(self.model.encode(prefix)[0].tolist())
                    content = proposed_close
                    continue
                content = proposed_close
                if regex_kind == "characters" and not content.endswith("]"):
                    content += "]"
                    prompt.extend(self.model.encode("]")[0].tolist())
                prompt.append(self.vocabulary.quote)
                return content
            proposed = content + self.vocabulary.strs[chosen]
            complete: str | None = None
            if regex_kind == "exact":
                match = re.search(r"[\[\](){}+*?\\.^$|]", proposed)
                complete = proposed[:match.start()] if match else proposed
            elif regex_kind in {"characters", "general"}:
                complete = self._completed_regex_prefix(proposed)
            if complete is not None:
                suffix = complete[len(content):]
                prompt.extend(self.model.encode(suffix)[0].tolist())
                prompt.append(self.vocabulary.quote)
                return complete
            fragment = self.vocabulary.strs[chosen]
            self._append_string_fragment(prompt, fragment, chosen)
            content = proposed
        if regex_kind == "characters" and not content.endswith("]"):
            content += "]"
            prompt.extend(self.model.encode("]")[0].tolist())
        prompt.append(self.vocabulary.quote)
        return content

    @staticmethod
    def _literal_candidates(user_input: str) -> list[str]:
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
    def _literal_prefix(cls, content: str, user_input: str) -> bool:
        """Check whether content prefixes a requested literal value."""
        return any(
            candidate.startswith(content)
            for candidate in cls._literal_candidates(user_input)
        )

    @classmethod
    def _literal_incomplete(
        cls, content: str, user_input: str
    ) -> bool:
        """Check if content is a strict prefix of a requested text span."""
        return any(
            candidate.startswith(content) and candidate != content
            for candidate in cls._literal_candidates(user_input)
        )

    def _append_string_fragment(
        self, prompt: list[int], fragment: str, token_id: int
    ) -> None:
        """Append one semantic string fragment using JSON escaping."""
        escaped = json.dumps(fragment, ensure_ascii=False)[1:-1]
        if escaped == fragment:
            prompt.append(token_id)
        else:
            prompt.extend(self.model.encode(escaped)[0].tolist())

    def _number(
        self, prompt: list[int], end_text: str, limit: int = 24
    ) -> list[int]:
        """Generate a terminating JSON number token by token."""
        output: list[int] = []
        text = ""
        end_token = self.model.encode(end_text)[0].tolist()[0]
        for _ in range(limit):
            logits = self.model.get_logits_from_input_ids(prompt)
            valid = {
                token_id
                for token_id, token_text
                in self.vocabulary.number_tokens.items()
                if NUMBER_PREFIX.fullmatch(text + token_text)
            }
            if not valid:
                raise RuntimeError("No valid number token")
            candidates = set(valid)
            if NUMBER_COMPLETE.fullmatch(text):
                candidates.add(END)
                best_number = max(valid, key=logits.__getitem__)
                if (
                    logits[best_number] - logits[end_token]
                    <= NUMBER_END_MARGIN
                ):
                    return output
            chosen = max(
                candidates,
                key=lambda token_id: (
                    logits[end_token] if token_id == END
                    else logits[token_id]
                ),
            )
            if chosen == END:
                return output
            output.append(chosen)
            prompt.append(chosen)
            text += self.vocabulary.number_tokens[chosen]
        raise RuntimeError("Number value did not terminate")

    def _boolean(self, prompt: list[int]) -> list[int]:
        """Choose one JSON boolean literal from model logits."""
        choices = {
            value: self.model.encode(value)[0].tolist()
            for value in ("true", "false")
        }
        first_logits = self.model.get_logits_from_input_ids(prompt)
        value = max(choices, key=lambda item: first_logits[choices[item][0]])
        token_ids = cast(list[int], choices[value])
        prompt.extend(token_ids)
        return token_ids
