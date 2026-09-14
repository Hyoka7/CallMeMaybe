"""Regex argument interpretation and pattern completion rules."""
from __future__ import annotations

import re

from pydantic import PrivateAttr

from src.model import JsonFunction
from src.token_generation import TokenGeneration


class RegexGeneration(TokenGeneration):
    """Interpret regex intent using the structural token selector."""

    _string_roles: dict[
        tuple[str, tuple[str, ...], str, str], str
    ] = PrivateAttr(default_factory=dict)

    def string_role(
        self,
        function: JsonFunction,
        parameter_name: str,
        user_input: str,
    ) -> str:
        """Classify a string parameter by semantic role."""
        string_names = tuple(
            name for name, definition in function.parameters.items()
            if definition["type"] == "string"
        )
        cache_key = (
            function.description,
            string_names,
            parameter_name,
            user_input,
        )
        if cache_key in self._string_roles:
            return self._string_roles[cache_key]
        prompt = (
            "Classify the CURRENT STRING ARGUMENT, not the overall operation, "
            "as exactly one role. Choose source when it contains the original "
            "text being operated on; choose regex when it contains the "
            "matching pattern; choose replacement when it contains text "
            "inserted for matches; choose ordinary for any other string. "
            "The word 'replace' in the function purpose does not make every "
            "argument a replacement. Compare the current argument with the "
            "other string arguments and the request. Typical distinctions "
            "are source_text or source_string -> source, pattern or regex -> "
            "regex, replacement_text or replacement -> replacement, and "
            "name -> ordinary. In a function with source_string, regex, and "
            "replacement arguments, classify those three arguments as source, "
            "regex, and replacement respectively. In particular, "
            "source_string means the original input text, not an ordinary "
            "string, when it appears alongside regex and replacement. "
            "For the exact argument set source_string, regex, replacement, "
            "the roles are source, regex, replacement; do not choose "
            "ordinary for any of those three arguments. "
            "Return only one of: source, regex, replacement, ordinary.\n"
            f"Function purpose: {function.description}\n"
            f"String arguments: {', '.join(string_names)}\n"
            f"Current argument: {parameter_name}\n"
            f"Request: {user_input}\nRole: \""
        )
        role = self.choose_trie_value(
            prompt, ["source", "regex", "replacement", "ordinary"]
        )
        self._string_roles[cache_key] = role
        return role

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
                "Examples: all vowels = characters; the word cat = exact; "
                "exact word bird = exact; numeric sequences = general.\n"
                "Example request: Replace all vowels in 'Programming is fun' "
                "with asterisks. Example intent: characters.\n"
                "Example request: Substitute the word 'cat' with 'dog' in "
                "'The cat sat'. Example intent: exact.\n"
                "Classify the meaning of the request, not punctuation copied "
                "from the surrounding sentence.\n"
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
        if not any(character in pattern for character in "[](){}+*?|\\.^$"):
            return True
        return not pattern.endswith(("\\", "|", "(", "[", "{"))

    @classmethod
    def _completed_regex_prefix(cls, pattern: str) -> str | None:
        """Find a completed structural regex inside a multi-text token."""
        for length in range(1, len(pattern) + 1):
            prefix = pattern[:length]
            structural = any(char in prefix for char in "[](){}+*?\\.^$")
            if structural and cls._regex_complete(prefix):
                return prefix
        return None
