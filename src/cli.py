import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse input and output paths from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Select functions and extract typed arguments from "
            "natural-language prompts."
        ),
    )

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
    )

    return parser.parse_args()
