import json
import tempfile
import unittest
from pathlib import Path

from src.json_to_file import write_results
from src.model import JsonResult


class JsonToFileTests(unittest.TestCase):
    """Verify the final result file format."""

    def test_writes_result_array(self) -> None:
        results = [
            JsonResult(
                prompt="What is the sum of 2 and 3?",
                name="fn_add_numbers",
                parameters={"a": 2, "b": 3},
            ),
            JsonResult(
                prompt="Greet shrek",
                name="fn_greet",
                parameters={"name": "shrek"},
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "results.json"
            write_results(output, results)

            with output.open(encoding="utf-8") as file:
                saved = json.load(file)

        self.assertEqual(
            saved,
            [
                {
                    "prompt": "What is the sum of 2 and 3?",
                    "name": "fn_add_numbers",
                    "parameters": {"a": 2, "b": 3},
                },
                {
                    "prompt": "Greet shrek",
                    "name": "fn_greet",
                    "parameters": {"name": "shrek"},
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
