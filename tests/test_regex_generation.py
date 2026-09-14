import unittest
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.constrained_decoder import ConstrainedDecoder, Vocabulary
from src.model import JsonFunction
from src.regex_generation import RegexGeneration


class ChoiceModel:
    """Provide one-token choices and scripted logit arrays."""

    TOKEN_IDS = {  # noqa: RUF012
        '"': 0,
        "regex": 1,
        "replacement": 2,
        "source_string": 3,
        "NONE": 4,
        "characters": 5,
        "exact": 6,
        "general": 7,
        "source": 8,
        "ordinary": 9,
    }

    def __init__(self, selected_ids: list[int]) -> None:
        self.selected_ids = iter(selected_ids)

    def encode(self, text: str) -> NDArray[np.int_]:
        return np.array([[self.TOKEN_IDS.get(text, 10)]])

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        del input_ids
        logits = [0.0] * 11
        logits[next(self.selected_ids)] = 10.0
        return logits


def semantic_decoder(selected_ids: list[int]) -> ConstrainedDecoder:
    """Construct a decoder for semantic choice methods."""
    return ConstrainedDecoder.model_construct(
        model=ChoiceModel(selected_ids),
        vocabulary=cast(Vocabulary, None),
    )


def regex_function() -> JsonFunction:
    """Return a schema with source, pattern, and replacement roles."""
    return JsonFunction(
        name="fn_substitute",
        description="Replace matches selected by a regex pattern.",
        parameters={
            "source_string": {"type": "string"},
            "regex": {"type": "string"},
            "replacement": {"type": "string"},
        },
        returns={"type": "string"},
    )


class RegexGenerationTests(unittest.TestCase):
    """Verify regex intent and string-argument role selection."""

    def test_regex_alternatives_only_finish_after_an_alternative(self) -> None:
        self.assertTrue(RegexGeneration._regex_complete("2|3"))
        self.assertFalse(RegexGeneration._regex_complete("2|"))

    def test_classifies_string_roles(self) -> None:
        function = regex_function()
        for role in ("source", "regex", "replacement", "ordinary"):
            with self.subTest(role=role):
                decoder = semantic_decoder([ChoiceModel.TOKEN_IDS[role]])
                self.assertEqual(
                    decoder.string_role(function, role, "request"),
                    role,
                )

    def test_classifies_each_regex_kind(self) -> None:
        function = regex_function()
        for kind in ("characters", "exact", "general"):
            with self.subTest(kind=kind):
                token_id = ChoiceModel.TOKEN_IDS[kind]
                decoder = semantic_decoder([token_id, 8])
                self.assertEqual(
                    decoder.regex_kind(function, "regex", "request"), kind
                )


if __name__ == "__main__":
    unittest.main()
