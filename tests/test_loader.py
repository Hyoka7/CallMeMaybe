import json
import tempfile
import unittest
from pathlib import Path

from src.loader import load_functions, load_prompts


class LoaderTests(unittest.TestCase):
    """Verify JSON file loading and Pydantic boundary validation."""

    def write_json(self, directory: str, name: str, data: object) -> Path:
        path = Path(directory) / name
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file)
        return path

    def test_loads_valid_function_and_prompt_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            function_path = self.write_json(directory, "functions.json", [{
                "name": "fn_count",
                "description": "Count items",
                "parameters": {"count": {"type": "integer"}},
                "returns": {"type": "integer"},
            }])
            prompt_path = self.write_json(
                directory, "prompts.json", [{"prompt": "Count three items"}]
            )
            functions = load_functions(function_path)
            self.assertEqual(functions.func[0].name, "fn_count")
            self.assertEqual(
                load_prompts(prompt_path).prompts[0].prompt,
                "Count three items",
            )

    def test_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            with self.assertRaisesRegex(
                ValueError, "Could not read function data"
            ):
                load_functions(path)

    def test_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("[", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid Json in prompt"):
                load_prompts(path)

    def test_rejects_duplicate_json_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "functions.json"
            path.write_text(
                '{"name": "fn_first", "name": "fn_second"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, rf"{path}: Duplicate JSON key: 'name'"
            ):
                load_functions(path)

    def test_rejects_invalid_function_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(directory, "functions.json", [{
                "name": "fn_bad",
                "description": "Invalid type",
                "parameters": {"value": {"type": "object"}},
                "returns": {"type": "string"},
            }])
            with self.assertRaisesRegex(
                ValueError, "Error while function validation"
            ):
                load_functions(path)

    def test_rejects_prompt_without_prompt_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(directory, "prompts.json", [{"text": "hi"}])
            with self.assertRaisesRegex(
                ValueError, "Required key 'prompt' is missing"
            ):
                load_prompts(path)


if __name__ == "__main__":
    unittest.main()
