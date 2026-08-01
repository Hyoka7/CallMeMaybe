import unittest
from unittest.mock import Mock

from src.build_tokenn import encode_options
from src.model import JsonFunction, JsonInput
from src.prompt import build_func_select_prompt


def make_encoded_text(text: str) -> list[Mock]:
    row = Mock()
    row.tolist.return_value = [ord(character) for character in text]
    return [row]


class FunctionOptionTests(unittest.TestCase):
    def test_encodes_numeric_options_in_definition_order(self) -> None:
        model = Mock()
        model.encode.side_effect = make_encoded_text
        self.assertEqual(
            encode_options(model, 3),
            [[ord("0")], [ord("1")], [ord("2")]],
        )

    def test_prompt_assigns_indices_to_mixed_function_names(self) -> None:
        functions = JsonInput(
            func=[
                JsonFunction(
                    name="fn_greet",
                    description="Greet someone.",
                    parameters={"name": {"type": "string"}},
                    returns={"type": "string"},
                ),
                JsonFunction(
                    name="ft_strlen",
                    description="Return string length.",
                    parameters={"s": {"type": "string"}},
                    returns={"type": "number"},
                ),
            ]
        )
        prompt = build_func_select_prompt(functions, "Greet shrek")
        self.assertIn("[0] fn_greet: Greet someone.", prompt)
        self.assertIn("[1] ft_strlen: Return string length.", prompt)
        self.assertTrue(prompt.endswith("Option number: "))


if __name__ == "__main__":
    unittest.main()
