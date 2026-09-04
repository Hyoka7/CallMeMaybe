import io
import unittest
from unittest.mock import patch

from src.main import main


class MainTests(unittest.TestCase):
    """Verify process-level error handling and exit statuses."""

    def test_returns_run_exit_status(self) -> None:
        with patch("src.main.run", return_value=0):
            self.assertEqual(main(), 0)

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

    def test_unexpected_exception_returns_failure(self) -> None:
        stderr = io.StringIO()

        with patch("src.main.run", side_effect=RuntimeError("broken")), patch(
            "sys.stderr", stderr
        ):
            self.assertEqual(main(), 1)

        self.assertEqual(stderr.getvalue(), "Aborting: broken\n")


if __name__ == "__main__":
    unittest.main()
