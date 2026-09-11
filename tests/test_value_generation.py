import unittest

import numpy as np
from numpy.typing import NDArray

from src.constrained_decoder import ConstrainedDecoder, Vocabulary


class ScriptedLogitsModel:
    """Return a predetermined score mapping for each generation step."""

    def __init__(self, scores: list[dict[int, float]]) -> None:
        self.scores = iter(scores)

    def encode(self, text: str) -> NDArray[np.int_]:
        del text
        return np.array([[0]])

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        del input_ids
        logits = [-100.0] * 8
        for token_id, score in next(self.scores).items():
            logits[token_id] = score
        return logits


def numeric_decoder(
    scores: list[dict[int, float]],
) -> ConstrainedDecoder:
    """Build a decoder with a small JSON-number vocabulary."""
    vocabulary = Vocabulary(
        strs=("}", "-", "1", ".", "5", "e", "2"),
        str_mask=np.ones(7, dtype=bool),
        lead_space=np.zeros(7, dtype=bool),
        close_mask=np.zeros(7, dtype=bool),
        close_prefix=(None,) * 7,
        quote=0,
        number_tokens={1: "-", 2: "1", 3: ".", 4: "5", 5: "e", 6: "2"},
        special_tokens={},
    )
    return ConstrainedDecoder.model_construct(
        model=ScriptedLogitsModel(scores), vocabulary=vocabulary
    )


class NumberGenerationTests(unittest.TestCase):
    """Verify JSON number continuation and termination rules."""

    def test_number_generates_fraction(self) -> None:
        decoder = numeric_decoder([{2: 10}, {3: 10}, {4: 10}, {0: 10}])
        self.assertEqual(decoder.generate_number([], "}"), [2, 3, 4])

    def test_number_generates_exponent(self) -> None:
        decoder = numeric_decoder([{2: 10}, {5: 10}, {6: 10}, {0: 10}])
        self.assertEqual(decoder.generate_number([], "}"), [2, 5, 6])

    def test_integer_rejects_decimal_token(self) -> None:
        decoder = numeric_decoder([{2: 10}, {3: 10}])
        self.assertEqual(decoder.generate_number([], "}", integer=True), [2])

    def test_integer_generates_negative_value(self) -> None:
        decoder = numeric_decoder([{1: 10}, {2: 10}, {0: 10}])
        self.assertEqual(
            decoder.generate_number([], "}", integer=True), [1, 2]
        )

    def test_minus_prefix_cannot_terminate(self) -> None:
        decoder = numeric_decoder([
            {1: 10},
            {0: 10, 2: 9},
            {0: 10},
        ])
        self.assertEqual(decoder.generate_number([], "}"), [1, 2])

    def test_number_accepts_leading_space_in_negative_token(self) -> None:
        decoder = ConstrainedDecoder.model_construct(
            model=ScriptedLogitsModel([
                {7: 10},
                {2: 10},
                {0: 10},
            ]),
            vocabulary=Vocabulary(
                strs=("}", "-", "1", ".", "5", "e", "2", " -"),
                str_mask=np.ones(8, dtype=bool),
                lead_space=np.zeros(8, dtype=bool),
                close_mask=np.zeros(8, dtype=bool),
                close_prefix=(None,) * 8,
                quote=0,
                number_tokens={7: " -", 2: "1"},
                special_tokens={},
            ),
        )
        self.assertEqual(decoder.generate_number([], "}"), [7, 2])

    def test_fraction_prefix_cannot_terminate(self) -> None:
        decoder = numeric_decoder([
            {2: 10},
            {3: 10},
            {0: 10, 4: 9},
            {0: 10},
        ])
        self.assertEqual(decoder.generate_number([], "}"), [2, 3, 4])

    def test_exponent_prefix_cannot_terminate(self) -> None:
        decoder = numeric_decoder([
            {2: 10},
            {5: 10},
            {0: 10, 6: 9},
            {0: 10},
        ])
        self.assertEqual(decoder.generate_number([], "}"), [2, 5, 6])


class BooleanModel:
    """Provide token IDs and scores for true and false literals."""

    def __init__(self, selected: str) -> None:
        self.selected = selected

    def encode(self, text: str) -> NDArray[np.int_]:
        return np.array([[1] if text.strip() == "true" else [2]])

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        del input_ids
        return [
            0.0,
            10.0 if self.selected == "true" else 0.0,
            10.0 if self.selected == "false" else 0.0,
        ]


class BooleanGenerationTests(unittest.TestCase):
    """Verify that boolean generation selects only JSON literals."""

    def test_selects_json_boolean_literals(self) -> None:
        for selected, token_id in (("true", 1), ("false", 2)):
            with self.subTest(selected=selected):
                decoder = ConstrainedDecoder.model_construct(
                    model=BooleanModel(selected), vocabulary=None
                )
                prompt: list[int] = []
                self.assertEqual(decoder.generate_boolean(prompt), [token_id])
                self.assertEqual(prompt, [token_id])


if __name__ == "__main__":
    unittest.main()
