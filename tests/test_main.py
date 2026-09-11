import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.decoder_errors import DecoderError
from src.main import main, run


class MainTests(unittest.TestCase):
    """Verify process-level error handling and exit statuses."""

    def test_run_reports_progress_and_save_message(self) -> None:
        args = SimpleNamespace(
            functions_definition="functions.json",
            input="prompts.json",
            output="results.json",
        )
        function = SimpleNamespace(name="fn_greet")
        functions = SimpleNamespace(func=[function])
        prompt = SimpleNamespace(prompt="Greet shrek")
        prompts = SimpleNamespace(prompts=[prompt])
        decoder = MagicMock()
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
        print_output.assert_called_once_with(
            "Result successfully saved to 'results.json'."
        )
        write_results.assert_called_once()

    def test_run_does_not_write_partial_results_after_decoder_error(
        self,
    ) -> None:
        args = SimpleNamespace(
            functions_definition="functions.json",
            input="prompts.json",
            output="results.json",
        )
        functions = SimpleNamespace(func=[SimpleNamespace(name="fn_one")])
        prompts = SimpleNamespace(
            prompts=[SimpleNamespace(prompt="first")]
        )
        decoder = MagicMock()
        decoder.generate_call.side_effect = DecoderError("broken")

        with (  # noqa: SIM117
            patch("src.main.parse_args", return_value=args),
            patch("src.main.Small_LLM_Model"),
            patch("src.main.load_functions", return_value=functions),
            patch("src.main.load_prompts", return_value=prompts),
            patch("src.main.Vocabulary.from_sdk"),
            patch("src.main.ConstrainedDecoder", return_value=decoder),
            patch("src.main.build_call_prompt", return_value="call prompt"),
            patch("src.main.write_results") as write_results,
            patch("src.main.tqdm", return_value=prompts.prompts),
        ):
            with self.assertRaisesRegex(DecoderError, "broken"):
                run()

        write_results.assert_not_called()

    def test_run_preserves_prompt_order_in_saved_results(self) -> None:
        args = SimpleNamespace(
            functions_definition="functions.json",
            input="prompts.json",
            output="results.json",
        )
        first_function = SimpleNamespace(name="fn_first")
        second_function = SimpleNamespace(name="fn_second")
        functions = SimpleNamespace(func=[first_function, second_function])
        prompts = SimpleNamespace(
            prompts=[
                SimpleNamespace(prompt="first request"),
                SimpleNamespace(prompt="second request"),
            ]
        )
        decoder = MagicMock()
        decoder.generate_call.side_effect = [
            (first_function, {"value": 1}),
            (second_function, {"value": 2}),
        ]

        with (
            patch("src.main.parse_args", return_value=args),
            patch("src.main.Small_LLM_Model"),
            patch("src.main.load_functions", return_value=functions),
            patch("src.main.load_prompts", return_value=prompts),
            patch("src.main.Vocabulary.from_sdk"),
            patch("src.main.ConstrainedDecoder", return_value=decoder),
            patch("src.main.build_call_prompt", return_value="call prompt"),
            patch("src.main.write_results") as write_results,
            patch("src.main.tqdm", return_value=prompts.prompts),
            patch("builtins.print"),
        ):
            self.assertEqual(run(), 0)

        saved_results = write_results.call_args.args[1]
        self.assertEqual(
            [result.prompt for result in saved_results],
            ["first request", "second request"],
        )
        self.assertEqual(
            [result.name for result in saved_results],
            ["fn_first", "fn_second"],
        )

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
        with (
            patch("src.main.run", side_effect=RuntimeError("bug")),
            self.assertRaisesRegex(RuntimeError, "bug"),
        ):
            main()


if __name__ == "__main__":
    unittest.main()
