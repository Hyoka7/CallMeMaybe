import unittest

from src.model import JsonFunction, JsonInput
from src.prompt import build_call_prompt


class PromptConstructionTests(unittest.TestCase):
    """Keep semantic boundary instructions explicit in the compiler prompt."""

    def test_regex_prompt_disambiguates_exact_and_character_patterns(
        self,
    ) -> None:
        functions = JsonInput(
            func=[
                JsonFunction(
                    name="fn_substitute_string_with_regex",
                    description=(
                        "Replace all occurrences matching a regex pattern "
                        "in a string."
                    ),
                    parameters={
                        "source_string": {"type": "string"},
                        "regex": {"type": "string"},
                        "replacement": {"type": "string"},
                    },
                    returns={"type": "string"},
                )
            ]
        )

        prompt = build_call_prompt(
            functions,
            "Replace all vowels in 'Programming is fun' with asterisks",
        )

        self.assertIn(
            "all vowels means exactly [aeiouAEIOU] with no trailing text",
            prompt,
        )
        self.assertIn(
            "The word 'cat' means the exact pattern cat, not cat.",
            prompt,
        )
        self.assertIn("asterisks should be emitted as '**'", prompt)
        self.assertIn("the regex value is [aeiouAEIOU]", prompt)


if __name__ == "__main__":
    unittest.main()
