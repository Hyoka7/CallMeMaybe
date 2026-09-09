import unittest

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from src.constrained_decoder import (
    END,
    ConstrainedDecoder,
    FunctionNameState,
    LiteralResult,
    LiteralState,
    ParameterKeyState,
    ParameterSeparatorState,
    ParameterValueState,
    TrieNode,
    UnsupportedTypeError,
    ValueHandlerRegistry,
    Vocabulary,
)
from src.model import JsonFunction, JsonInput
from src.prompt import build_call_prompt
from src.regex_generation import RegexGeneration
from src.value_handlers import (
    BooleanHandler,
    IntegerHandler,
    NumberHandler,
    StringHandler,
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


class FakeLiteralModel:
    """Track logits calls while emitting one fixed literal token."""

    def __init__(self) -> None:
        self.logits_calls = 0

    def encode(self, text: str) -> NDArray[np.int_]:
        del text
        return np.array([[0]])

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        del input_ids
        self.logits_calls += 1
        return [10.0]


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

    def test_concrete_state_and_handler_classes_use_pydantic(self) -> None:
        classes = (
            LiteralResult,
            LiteralState,
            FunctionNameState,
            ParameterKeyState,
            ParameterSeparatorState,
            ParameterValueState,
            ValueHandlerRegistry,
            StringHandler,
            NumberHandler,
            IntegerHandler,
            BooleanHandler,
        )
        for model_class in classes:
            with self.subTest(model_class=model_class.__name__):
                self.assertTrue(issubclass(model_class, BaseModel))

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
            def generate(
                self,
                decoder: ConstrainedDecoder,
                prompt: list[int],
                user_input: str,
                parameter_name: str,
                function: JsonFunction,
            ) -> str:
                return "2026-09-04"

        decoder.register_value_handler("date", DateHandler())
        self.assertIsNotNone(decoder.value_handlers().get("date"))

    def test_default_registry_contains_builtin_handlers(self) -> None:
        registry = ValueHandlerRegistry.default()
        for type_name in ("string", "number", "integer", "boolean"):
            with self.subTest(type_name=type_name):
                self.assertIsNotNone(registry.get(type_name))

    def test_unknown_value_type_has_explicit_error(self) -> None:
        decoder = string_decoder(FakeStringModel([0]))
        with self.assertRaises(UnsupportedTypeError):
            decoder.value_handlers().get("date")

    def test_parameter_value_state_dispatches_through_registry(self) -> None:
        decoder = string_decoder(FakeStringModel([0]))
        handler = ParameterValueState(type_name="string").handler(
            decoder.value_handlers()
        )
        self.assertIsNotNone(handler)

    def test_integer_is_a_supported_schema_type(self) -> None:
        function = JsonFunction(
            name="fn_count",
            description="Count items",
            parameters={"count": {"type": "integer"}},
            returns={"type": "integer"},
        )
        self.assertEqual(function.parameters["count"]["type"], "integer")

    def test_prompt_requires_bare_replacement_symbol(self) -> None:
        prompt = build_call_prompt(
            JsonInput(func=[]), "Replace vowels with asterisks"
        )
        self.assertIn("do not add parentheses", prompt)

    def test_repeated_symbol_replacement_is_reduced_to_one_unit(self) -> None:
        self.assertEqual(
            RegexGeneration.refine_replacement("**", []), "*"
        )

    def test_wrapped_symbol_replacement_is_unwrapped(self) -> None:
        self.assertEqual(
            RegexGeneration.refine_replacement("(*)", []), "*"
        )

    def test_explicit_replacement_literal_is_preserved(self) -> None:
        self.assertEqual(
            RegexGeneration.refine_replacement("**", ["**"]), "**"
        )

    def test_literal_state_accepts_only_prefix_tokens(self) -> None:
        state = LiteralState(remaining='"prompt": "')
        self.assertTrue(state.consume('"prompt":').remaining == ' "')
        self.assertFalse(state.consume('"name"').valid)

    def test_literal_state_finishes_at_exact_boundary(self) -> None:
        state = LiteralState(remaining='{}')
        self.assertTrue(state.consume('{').remaining == '}')
        self.assertTrue(state.consume('{}').finished)

    def test_literal_candidates_are_derived_from_vocabulary(self) -> None:
        decoder = string_decoder(FakeStringModel([0]))
        candidates = decoder.literal_candidates(LiteralState(remaining='x'))
        self.assertIn(2, candidates)
        self.assertNotIn(1, candidates)

    def test_fixed_literal_avoids_model_logits(self) -> None:
        model = FakeLiteralModel()
        vocabulary = Vocabulary(
            strs=("{",),
            str_mask=np.array([True]),
            lead_space=np.array([False]),
            close_mask=np.array([False]),
            close_prefix=(None,),
            quote=0,
            number_tokens={0: "1"},
            special_tokens={},
        )
        decoder = ConstrainedDecoder.model_construct(
            model=model, vocabulary=vocabulary
        )
        prompt: list[int] = []
        output: list[int] = []

        decoder.emit_literal(prompt, output, "{")

        self.assertEqual(model.logits_calls, 0)
        self.assertEqual(output, [0])


class StringDecoderTests(unittest.TestCase):
    """Verify JSON strings can represent required edge-case values."""

    def test_allows_empty_string(self) -> None:
        model = FakeStringModel([0])
        decoder = string_decoder(model)
        prompt = [99]

        value = decoder.generate_string(prompt, user_input="Use ''")

        self.assertEqual(value, "")
        self.assertEqual(prompt, [99, 0])

    def test_escapes_quote_in_generated_string(self) -> None:
        model = FakeStringModel([1, 0])
        decoder = string_decoder(model)
        prompt = [99]

        value = decoder.generate_string(prompt, user_input="Use 'a\"b'")

        self.assertEqual(value, 'a"b')
        self.assertEqual(prompt, [99, 2, 3, 0, 2, 0])

    def test_escapes_backslash_in_generated_string(self) -> None:
        model = FakeStringModel([3, 0])
        decoder = string_decoder(model)
        prompt = [99]

        value = decoder.generate_string(prompt, user_input="Use '\\'")

        self.assertEqual(value, "\\")
        self.assertEqual(prompt, [99, 3, 3, 0])


if __name__ == "__main__":
    unittest.main()
