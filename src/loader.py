import json
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from src.model import JsonInput, PromptInput


def _duplicate_key_hook(
    path: Path,
) -> Callable[[list[tuple[str, object]]], dict[str, object]]:
    """Create a JSON object hook that reports the source file on failure."""
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        """Build a JSON object while rejecting duplicate member names."""
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: Duplicate JSON key: {key!r}")
            result[key] = value
        return result

    return reject_duplicate_keys


def load_functions(path: Path) -> JsonInput:
    """Load and validate function definitions from a JSON file."""
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file, object_pairs_hook=_duplicate_key_hook(path))
        return JsonInput(func=data)
    except OSError as err:
        raise ValueError(f"Could not read function data: {err}")
    except json.JSONDecodeError as json_err:
        raise ValueError(f"Invalid Json in function: {json_err}")
    except ValidationError as val_err:
        raise ValueError(
            f"Error while function validation: {val_err.errors()[0]['msg']}"
        )


def load_prompts(path: Path) -> PromptInput:
    """Load and validate natural-language prompts from a JSON file."""
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file, object_pairs_hook=_duplicate_key_hook(path))
        return PromptInput(prompts=data)
    except OSError as err:
        raise ValueError(f"Could not read prompt data: {err}")
    except json.JSONDecodeError as json_err:
        raise ValueError(f"Invalid Json in prompt: {json_err}")
    except ValidationError as val_err:
        message = val_err.errors()[0]["msg"]
        raise ValueError(f"Error while prompt validation: {message}")
