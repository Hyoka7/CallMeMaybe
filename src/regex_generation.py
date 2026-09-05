"""Regex argument interpretation and pattern completion rules."""
from __future__ import annotations

import re

from pydantic import PrivateAttr

from src.model import JsonFunction
from src.token_generation import TokenGeneration


class RegexGeneration(TokenGeneration):
    """Interpret regex intent using the structural token selector."""

    _regex_roles: dict[tuple[str, tuple[str, ...]], str | None] = PrivateAttr(
        default_factory=dict
    )

    def _is_regex_argument(
        self, function: JsonFunction, parameter_name: str
    ) -> bool:
        """Choose the pattern argument by comparing the complete schema."""
        string_names = tuple(
            name for name, definition in function.parameters.items()
            if definition["type"] == "string"
        )
        cache_key = (function.description, string_names)
        description = function.description.lower()
        if not any(
            marker in description for marker in ("regex", "regular expression")
        ):
            self._regex_roles[cache_key] = None
            return False
        if cache_key in self._regex_roles:
            return self._regex_roles[cache_key] == parameter_name
        prompt = (
            "Choose which string argument itself stores the reusable regular "
            "expression used for matching. Do not choose source text, "
            "replacement text, names, or other direct values. Choose NONE if "
            "this function has no regex-pattern argument.\n"
            f"Function purpose: {function.description}\n"
            f"String arguments: {', '.join(string_names)}\n"
            "Pattern argument: \""
        )
        selected = self._trie_choice(prompt, list(string_names) + ["NONE"])
        self._regex_roles[cache_key] = None if selected == "NONE" else selected
        return selected == parameter_name

    def _regex_kind(
        self,
        function: JsonFunction,
        parameter_name: str,
        user_input: str,
    ) -> str:
        """Classify the requested pattern as character, exact, or general."""
        choices = ("characters", "exact", "general")
        choice_ids = {
            choice: self.model.encode(choice)[0].tolist()[0]
            for choice in choices
        }

        def scores(request: str) -> dict[str, float]:
            """Return baseline-adjustable intent logits for one request."""
            prompt = (
                "Classify regex matching intent as characters for alternative "
                "individual characters, exact for one exact literal word or "
                "text, or general for a repeated category or other "
                "structure.\n"
                "Examples: individual vowels = characters; exact word bird = "
                "exact; numeric sequences = general.\n"
                f"Function purpose: {function.description}\n"
                f"Target regex argument: {parameter_name}\n"
                f"Request: {request}\nIntent: "
            )
            ids = self.model.encode(prompt)[0].tolist()
            logits = self.model.get_logits_from_input_ids(ids)
            return {
                choice: float(logits[token_id])
                for choice, token_id in choice_ids.items()
            }

        actual = scores(user_input)
        baseline = scores("unspecified matching intent")
        return max(
            choices,
            key=lambda choice: actual[choice] - baseline[choice],
        )

    @staticmethod
    def _regex_complete(pattern: str) -> bool:
        """Return whether a minimal reusable regex has reached a safe end."""
        if not pattern or pattern.endswith(("\\", "|", "(", "[", "{")):
            return False
        try:
            re.compile(pattern)
        except re.error:
            return False
        if not any(character in pattern for character in "[](){}+*?\\.^$"):
            return True
        return pattern.endswith(("]", ")", "}", "+", "*", "?", "$"))

    @classmethod
    def _completed_regex_prefix(cls, pattern: str) -> str | None:
        """Find a completed structural regex inside a multi-text token."""
        for length in range(1, len(pattern) + 1):
            prefix = pattern[:length]
            structural = any(char in prefix for char in "[](){}+*?\\.^$")
            if structural and cls._regex_complete(prefix):
                return prefix
        return None

    def _refine_regex(self, pattern: str) -> str:
        """Enforce the prompt's shortest-pattern invariant."""
        if not pattern.endswith(".*") or len(pattern) <= 2:
            return pattern
        return pattern[:-2]
