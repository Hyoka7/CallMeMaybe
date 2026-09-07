import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.main import main, run
from src.decoder_errors import DecoderError


class MainTests(unittest.TestCase):
    """Verify process-level error handling and exit statuses."""

    def test_returns_run_exit_status(self) -> None:
        with patch("src.main.run", return_value=0):
            self.assertEqual(main(), 0)

    def test_run_reports_progress_without_printing_results(self) -> None:
        args = SimpleNamespace(
            functions_definition="functions.json",
            input="prompts.json",
            output="results.json",
        )
        function = SimpleNamespace(name="fn_greet")
        functions = SimpleNamespace(func=[function])
        prompt = SimpleNamespace(prompt="Greet shrek")
        prompts = SimpleNamespace(prompts=[prompt])
        decoder = unittest.mock.MagicMock()
        decoder.generate_call.return_value = (function, {"name": "shrek"})

        with (
            patch("src.main.parse_args", return_value=args),
            patch("src.main.Small_LLM_Model"),
            patch("src.main.load_functions", return_value=functions),
            patch("src.main.load_prompts", return_value=prompts),
            patch("src.main.Vocabulary.from_sdk"),
            patch("src.main.ConstrainedDecoder", return_value=decoder),
            patch("src.main.build_call_prompt", return_value="call prompt"),
            patch("src.main.write_results") as write_results,
            patch("src.main.tqdm", create=True) as progress,
            patch("builtins.print") as print_output,
        ):
            progress.return_value = prompts.prompts
            self.assertEqual(run(), 0)

        progress.assert_called_once_with(
            prompts.prompts,
            desc="Generating calls",
            unit="prompt",
        )
        print_output.assert_not_called()
        write_results.assert_called_once()

    def test_keyboard_interrupt_returns_130(self) -> None:
        stderr = io.StringIO()

        with patch("src.main.run", side_effect=KeyboardInterrupt), patch(
            "sys.stderr", stderr
        ):
            self.assertEqual(main(), 130)

        self.assertEqual(stderr.getvalue(), "\nAborted by user.\n")

    def test_memory_error_returns_failure(self) -> None:
        stderr = io.StringIO()

        with patch("src.main.run", side_effect=MemoryError), patch(
            "sys.stderr", stderr
        ):
            self.assertEqual(main(), 1)

        self.assertEqual(stderr.getvalue(), "Aborting: insufficient memory.\n")

    def test_decoder_error_returns_failure(self) -> None:
        stderr = io.StringIO()

        with patch("src.main.run", side_effect=DecoderError("broken")), patch(
            "sys.stderr", stderr
        ):
            self.assertEqual(main(), 1)

        self.assertEqual(stderr.getvalue(), "Aborting: broken\n")

    def test_unexpected_exception_is_not_swallowed(self) -> None:
        with patch("src.main.run", side_effect=RuntimeError("bug")):
            with self.assertRaisesRegex(RuntimeError, "bug"):
                main()


if __name__ == "__main__":
    unittest.main()
