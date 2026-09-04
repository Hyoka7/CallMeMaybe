import unittest

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from src.constrained_decoder import (
    END,
    ConstrainedDecoder,
    TrieNode,
    Vocabulary,
    UnsupportedTypeError,
    LiteralState,
    ParameterValueState,
)


class FakeStringModel:
    """Supply deterministic token scores for focused string tests."""

    def __init__(self, ranked_tokens: list[int]) -> None:
        self.ranked_tokens = iter(ranked_tokens)
        self.encoded: dict[str, list[int]] = {
            '"': [0],
            r'\"': [3, 0],
            r'a\"b': [2, 3, 0, 2],
            r"\\": [3, 3],
        }

    def encode(self, text: str) -> NDArray[np.int_]:
        return np.array([self.encoded.get(text, [2])])

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        del input_ids
        selected = next(self.ranked_tokens)
        logits = [0.0] * 4
        logits[selected] = 10.0
        return logits


def string_vocabulary() -> Vocabulary:
    """Create a tiny vocabulary containing quote and backslash tokens."""
    return Vocabulary(
        strs=('"', 'a"b', "x", "\\"),
        str_mask=np.array([False, False, True, False]),
        lead_space=np.zeros(4, dtype=bool),
        close_mask=np.array([True, False, False, False]),
        close_prefix=("", None, None, None),
        quote=0,
        number_tokens={2: "1"},
        special_tokens={0: '"', 1: 'a"b', 3: "\\"},
    )


def string_decoder(model: FakeStringModel) -> ConstrainedDecoder:
    """Construct a decoder around a deliberately minimal SDK test double."""
    return ConstrainedDecoder.model_construct(
        model=model, vocabulary=string_vocabulary()
    )


class TrieNodeTests(unittest.TestCase):
    """Verify the recursive Pydantic trie used during decoding."""

    def test_is_a_pydantic_model(self) -> None:
        self.assertTrue(issubclass(TrieNode, BaseModel))

    def test_prefix_function_allows_child_and_end(self) -> None:
        root = TrieNode()
        root.insert([1, 2], "fn_add")
        root.insert([1, 2, 3], "fn_add_numbers")

        prefix = root.children[1].children[2]

        self.assertEqual(set(prefix.children), {END, 3})
        self.assertEqual(prefix.children[END].value, "fn_add")
        self.assertEqual(
            prefix.children[3].children[END].value,
            "fn_add_numbers",
        )

    def test_decoder_can_register_a_future_value_type(self) -> None:
        decoder = string_decoder(FakeStringModel([0]))

        class DateHandler:
            def generate(self, decoder, prompt, user_input, parameter_name, function):
                return "2026-09-04"

        decoder.register_value_handler("date", DateHandler())
        self.assertIsNotNone(decoder._ensure_value_handlers().get("date"))

    def test_unknown_value_type_has_explicit_error(self) -> None:
        decoder = string_decoder(FakeStringModel([0]))
        with self.assertRaises(UnsupportedTypeError):
            decoder._ensure_value_handlers().get("date")

    def test_parameter_value_state_dispatches_through_registry(self) -> None:
        decoder = string_decoder(FakeStringModel([0]))
        handler = ParameterValueState("string").handler(
            decoder._ensure_value_handlers()
        )
        self.assertIsNotNone(handler)

    def test_literal_state_accepts_only_prefix_tokens(self) -> None:
        state = LiteralState('"prompt": "')
        self.assertTrue(state.consume('"prompt":').remaining == ' "')
        self.assertFalse(state.consume('"name"').valid)

    def test_literal_state_finishes_at_exact_boundary(self) -> None:
        state = LiteralState('{}')
        self.assertTrue(state.consume('{').remaining == '}')
        self.assertTrue(state.consume('{}').finished)

    def test_literal_candidates_are_derived_from_vocabulary(self) -> None:
        decoder = string_decoder(FakeStringModel([0]))
        candidates = decoder.literal_candidates(LiteralState('x'))
        self.assertIn(2, candidates)
        self.assertNotIn(1, candidates)


class StringDecoderTests(unittest.TestCase):
    """Verify JSON strings can represent required edge-case values."""

    def test_allows_empty_string(self) -> None:
        model = FakeStringModel([0])
        decoder = string_decoder(model)
        prompt = [99]

        value = decoder._string(prompt, user_input="Use ''")

        self.assertEqual(value, "")
        self.assertEqual(prompt, [99, 0])

    def test_escapes_quote_in_generated_string(self) -> None:
        model = FakeStringModel([1, 0])
        decoder = string_decoder(model)
        prompt = [99]

        value = decoder._string(prompt, user_input="Use 'a\"b'")

        self.assertEqual(value, 'a"b')
        self.assertEqual(prompt, [99, 2, 3, 0, 2, 0])

    def test_escapes_backslash_in_generated_string(self) -> None:
        model = FakeStringModel([3, 0])
        decoder = string_decoder(model)
        prompt = [99]

        value = decoder._string(prompt, user_input="Use '\\'")

        self.assertEqual(value, "\\")
        self.assertEqual(prompt, [99, 3, 3, 0])


if __name__ == "__main__":
    unittest.main()
