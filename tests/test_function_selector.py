import unittest
from unittest.mock import Mock

from src.function_selector import function_selector


class FunctionSelectorTests(unittest.TestCase):
    """Verify constrained beam-search function selection."""

    def test_keeps_lower_ranked_first_token_until_completion(self) -> None:
        encoded_prompt = Mock()
        encoded_prompt.tolist.return_value = [99]
        model = Mock()
        model.encode.return_value = [encoded_prompt]

        def get_logits(input_ids: list[int]) -> list[float]:
            logits = [float("-inf")] * 100
            if input_ids[-1] == 99:
                logits[1] = 1.0
                logits[2] = 0.8
            elif input_ids[-1] == 1:
                logits[3] = 0.0
                logits[4] = 0.0
            elif input_ids[-1] == 2:
                logits[5] = 0.0
            return logits

        model.get_logits_from_input_ids.side_effect = get_logits
        candidates = [
            [1, 3],
            [1, 4],
            [2, 5],
        ]

        selected = function_selector(
            model,
            "select a function",
            candidates,
            beam_width=3,
        )

        self.assertEqual(selected, [2, 5])

    def test_rejects_empty_candidates(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "No encoded function candidates"
        ):
            function_selector(Mock(), "prompt", [])

    def test_rejects_non_positive_beam_width(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Beam width must be positive"
        ):
            function_selector(Mock(), "prompt", [[1]], beam_width=0)


if __name__ == "__main__":
    unittest.main()
