import unittest
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.constrained_decoder import ConstrainedDecoder, Vocabulary
from src.model import JsonFunction


class ChoiceModel:
    """Provide one-token choices and scripted logit arrays."""

    TOKEN_IDS = {
        '"': 0,
        "regex": 1,
        "replacement": 2,
        "source_string": 3,
        "NONE": 4,
        "characters": 5,
        "exact": 6,
        "general": 7,
    }

    def __init__(self, selected_ids: list[int]) -> None:
        self.selected_ids = iter(selected_ids)

    def encode(self, text: str) -> NDArray[np.int_]:
        return np.array([[self.TOKEN_IDS.get(text, 8)]])

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        del input_ids
        logits = [0.0] * 9
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

    def test_identifies_regex_argument(self) -> None:
        decoder = semantic_decoder([ChoiceModel.TOKEN_IDS["regex"]])
        function = regex_function()
        self.assertTrue(decoder._is_regex_argument(function, "regex"))
        self.assertFalse(decoder._is_regex_argument(function, "source_string"))

    def test_identifies_replacement_argument(self) -> None:
        decoder = semantic_decoder([ChoiceModel.TOKEN_IDS["replacement"]])
        function = regex_function()
        self.assertTrue(
            decoder._is_replacement_argument(function, "replacement")
        )
        self.assertFalse(
            decoder._is_replacement_argument(function, "source_string")
        )

    def test_classifies_each_regex_kind(self) -> None:
        function = regex_function()
        for kind in ("characters", "exact", "general"):
            with self.subTest(kind=kind):
                token_id = ChoiceModel.TOKEN_IDS[kind]
                decoder = semantic_decoder([token_id, 8])
                self.assertEqual(
                    decoder._regex_kind(function, "regex", "request"), kind
                )


if __name__ == "__main__":
    unittest.main()
