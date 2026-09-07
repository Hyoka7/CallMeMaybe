import string
import unittest
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.constrained_decoder import ConstrainedDecoder, Vocabulary
from src.model import JsonFunction


class CharacterModel:
    """Encode ASCII one character per token for a complete-call test."""

    def __init__(self) -> None:
        self.selected = iter((ord("7"), ord("}")))

    def encode(self, text: str) -> NDArray[np.int_]:
        return np.array([[ord(character) for character in text]])

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        del input_ids
        logits = [-100.0] * 128
        logits[next(self.selected)] = 10.0
        return logits


def character_vocabulary() -> Vocabulary:
    """Build the minimal metadata needed by the ASCII model."""
    texts = tuple(chr(token_id) for token_id in range(128))
    return Vocabulary(
        strs=texts,
        str_mask=np.array([
            text in string.printable and text not in {'"', "\\"}
            for text in texts
        ]),
        lead_space=np.array([text.isspace() for text in texts]),
        close_mask=np.array([text == '"' for text in texts]),
        close_prefix=tuple("" if text == '"' else None for text in texts),
        quote=ord('"'),
        number_tokens={
            ord(character): character for character in "-+.eE0123456789"
        },
        special_tokens={},
    )


class GenerationEngineTests(unittest.TestCase):
    """Verify complete call assembly and input preconditions."""

    def test_generate_call_rejects_empty_function_list(self) -> None:
        decoder = ConstrainedDecoder.model_construct(
            model=cast(object, None),
            vocabulary=cast(Vocabulary, None),
        )
        with self.assertRaisesRegex(ValueError, "No functions available"):
            decoder.generate_call("prompt", [], "request")

    def test_generate_call_builds_complete_json(self) -> None:
        function = JsonFunction(
            name="fn_count",
            description="Return a count",
            parameters={"count": {"type": "integer"}},
            returns={"type": "integer"},
        )
        decoder = ConstrainedDecoder.model_construct(
            model=CharacterModel(), vocabulary=character_vocabulary()
        )

        selected, parameters = decoder.generate_call(
            "compiler context", [function], "count seven"
        )

        self.assertEqual(selected, function)
        self.assertEqual(parameters, {"count": 7})


if __name__ == "__main__":
    unittest.main()
