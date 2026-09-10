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
    _replacement_roles: dict[
        tuple[str, tuple[str, ...]], str | None
    ] = PrivateAttr(default_factory=dict)

    @staticmethod
    def is_source_argument(parameter_name: str) -> bool:
        """Identify a parameter intended to hold source/input text."""
        name = parameter_name.lower()
        return (
            any(marker in name for marker in ("source", "input", "text"))
            and "regex" not in name
            and "pattern" not in name
            and "replacement" not in name
        )

    def is_replacement_argument(
        self, function: JsonFunction, parameter_name: str
    ) -> bool:
        """Choose the argument that stores replacement text."""
        string_names = tuple(
            name for name, definition in function.parameters.items()
            if definition["type"] == "string"
        )
        cache_key = (function.description, string_names)
        if cache_key in self._replacement_roles:
            return self._replacement_roles[cache_key] == parameter_name
        prompt = (
            "Choose which string argument stores the replacement value "
            "inserted for every match. Do not choose source text, matching "
            "patterns, names, or other values. Choose NONE if there is no "
            "replacement argument. The value must be the exact text inserted "
            "into the source, not the name or description of that text. For "
            "example, asterisks means the symbol '*', not the word "
            "'asterisk' or 'asterisks'.\n"
            f"Function purpose: {function.description}\n"
            f"String arguments: {', '.join(string_names)}\n"
            'Replacement argument: "'
        )
        selected = self.choose_trie_value(
            prompt, list(string_names) + ["NONE"]
        )
        self._replacement_roles[cache_key] = (
            None if selected == "NONE" else selected
        )
        return selected == parameter_name

    @staticmethod
    def refine_replacement(
        value: str, explicit_literals: list[str]
    ) -> str:
        """Reduce an inferred single-symbol replacement to one unit."""
        if value in explicit_literals:
            return value
        if (
            len(value) > 1
            and len(set(value)) == 1
            and not value[0].isalnum()
            and not value[0].isspace()
        ):
            return value[0]
        pairs = {"(": ")", "[": "]", "{": "}"}
        if (
            len(value) == 3
            and pairs.get(value[0]) == value[2]
            and not value[1].isalnum()
            and not value[1].isspace()
        ):
            return value[1]
        return value

    def is_regex_argument(
        self,
        function: JsonFunction,
        parameter_name: str,
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
        selected = self.choose_trie_value(
            prompt, list(string_names) + ["NONE"]
        )
        self._regex_roles[cache_key] = None if selected == "NONE" else selected
        return selected == parameter_name

    def regex_kind(
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

    def refine_regex(self, pattern: str) -> str:
        """Enforce the prompt's shortest-pattern invariant."""
        if not pattern.endswith(".*") or len(pattern) <= 2:
            return pattern
        return pattern[:-2]
