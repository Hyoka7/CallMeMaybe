import json
import unittest
from collections.abc import Iterator
from unittest.mock import Mock

from src.model import JsonFunction
from src.params_maker import params_maker
from src.params_maker import MAX_STRING_TOKENS


def make_encoded_text(text: str) -> list[Mock]:
    """Return an encode-like value backed by ASCII token IDs."""
    row = Mock()
    row.tolist.return_value = [ord(character) for character in text]
    return [row]


def make_model(generated_text: str) -> Mock:
    """Create an LLM mock that emits the requested text token by token."""
    generated: Iterator[str] = iter(generated_text)
    model = Mock()
    model.encode.side_effect = make_encoded_text
    model.decode.side_effect = lambda token_ids: "".join(
        chr(token_id) for token_id in token_ids
    )

    def get_logits(_: list[int]) -> list[float]:
        character = next(generated)
        logits = [float("-inf")] * 128
        logits[ord(character)] = 1.0
        return logits

    model.get_logits_from_input_ids.side_effect = get_logits
    return model


class ParamsMakerTests(unittest.TestCase):
    """Verify constrained parameter JSON generation."""

    def test_two_number_parameters(self) -> None:
        function = JsonFunction(
            name="fn_add_numbers",
            description="Add two numbers.",
            parameters={
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            returns={"type": "number"},
        )
        model = make_model("2,3}")

        result = params_maker(model, [1], function)

        self.assertEqual(json.loads(result), {"a": 2, "b": 3})

    def test_string_parameter(self) -> None:
        function = JsonFunction(
            name="fn_greet",
            description="Greet a person.",
            parameters={"name": {"type": "string"}},
            returns={"type": "string"},
        )
        model = make_model('shrek"')

        result = params_maker(model, [1], function)

        self.assertEqual(json.loads(result), {"name": "shrek"})

    def test_multi_character_string_token(self) -> None:
        function = JsonFunction(
            name="fn_greet",
            description="Greet a person.",
            parameters={"name": {"type": "string"}},
            returns={"type": "string"},
        )
        generated: Iterator[int] = iter([127, ord('"')])
        model = Mock()
        model.encode.side_effect = make_encoded_text
        model.decode.side_effect = lambda token_ids: "".join(
            "shrek" if token_id == 127 else chr(token_id)
            for token_id in token_ids
        )

        def get_logits(_: list[int]) -> list[float]:
            token_id = next(generated)
            logits = [float("-inf")] * 128
            logits[token_id] = 1.0
            return logits

        model.get_logits_from_input_ids.side_effect = get_logits

        result = params_maker(model, [1], function)

        self.assertEqual(json.loads(result), {"name": "shrek"})

    def test_boolean_parameter(self) -> None:
        function = JsonFunction(
            name="fn_is_enabled",
            description="Check whether a feature is enabled.",
            parameters={"enabled": {"type": "boolean"}},
            returns={"type": "boolean"},
        )
        model = make_model("true")

        result = params_maker(model, [1], function)

        self.assertEqual(json.loads(result), {"enabled": True})

    def test_multiple_digit_number(self) -> None:
        function = JsonFunction(
            name="fn_get_square_root",
            description="Calculate a square root.",
            parameters={"a": {"type": "number"}},
            returns={"type": "number"},
        )
        model = make_model("144}")

        result = params_maker(model, [1], function)

        self.assertEqual(json.loads(result), {"a": 144})

    def test_false_boolean_parameter(self) -> None:
        function = JsonFunction(
            name="fn_is_enabled",
            description="Check whether a feature is enabled.",
            parameters={"enabled": {"type": "boolean"}},
            returns={"type": "boolean"},
        )
        model = make_model("false")

        result = params_maker(model, [1], function)

        self.assertEqual(json.loads(result), {"enabled": False})

    def test_empty_string_parameter(self) -> None:
        function = JsonFunction(
            name="fn_greet",
            description="Greet a person.",
            parameters={"name": {"type": "string"}},
            returns={"type": "string"},
        )
        model = make_model('"')

        result = params_maker(model, [1], function)

        self.assertEqual(json.loads(result), {"name": ""})

    def test_multiple_string_parameters(self) -> None:
        function = JsonFunction(
            name="fn_replace",
            description="Replace text in a string.",
            parameters={
                "source": {"type": "string"},
                "replacement": {"type": "string"},
            },
            returns={"type": "string"},
        )
        model = make_model('cat"dog"')

        result = params_maker(model, [1], function)

        self.assertEqual(
            json.loads(result),
            {"source": "cat", "replacement": "dog"},
        )

    def test_no_parameters(self) -> None:
        function = JsonFunction(
            name="fn_now",
            description="Return the current value.",
            parameters={},
            returns={"type": "string"},
        )
        model = make_model("")

        result = params_maker(model, [1], function)

        self.assertEqual(json.loads(result), {})

    def test_string_generation_has_a_safety_limit(self) -> None:
        function = JsonFunction(
            name="fn_echo",
            description="Echo text.",
            parameters={"text": {"type": "string"}},
            returns={"type": "string"},
        )
        model = make_model("a" * MAX_STRING_TOKENS)

        result = params_maker(model, [1], function)

        self.assertEqual(
            json.loads(result),
            {"text": "a" * MAX_STRING_TOKENS},
        )

    def test_string_beam_keeps_a_path_that_ends_cleanly(self) -> None:
        function = JsonFunction(
            name="fn_echo",
            description="Echo text.",
            parameters={"text": {"type": "string"}},
            returns={"type": "string"},
        )
        model = Mock()
        model.encode.side_effect = make_encoded_text
        model.decode.side_effect = lambda token_ids: "".join(
            chr(token_id) for token_id in token_ids
        )

        def get_logits(prompt_ids: list[int]) -> list[float]:
            logits = [float("-inf")] * 128
            if prompt_ids[-1] == ord("b"):
                logits[ord('"')] = 5.0
            elif prompt_ids[-1] == ord("a"):
                logits[ord("a")] = 5.0
                logits[ord('"')] = -5.0
            else:
                logits[ord("a")] = 5.0
                logits[ord("b")] = 4.0
                logits[ord('"')] = -10.0
            return logits

        model.get_logits_from_input_ids.side_effect = get_logits

        result = params_maker(model, [1], function)

        self.assertEqual(json.loads(result), {"text": "b"})


if __name__ == "__main__":
    unittest.main()
