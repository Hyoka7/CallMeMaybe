import json
from pathlib import Path

from pydantic import ValidationError

from src.model import JsonInput, PromptInput


def load_functions(path: Path) -> JsonInput:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
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
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        return PromptInput(prompts=data)
    except OSError as err:
        raise ValueError(f"Could not read prompt data: {err}")
    except json.JSONDecodeError as json_err:
        raise ValueError(f"Invalid Json in prompt: {json_err}")
    except ValidationError as val_err:
        raise ValueError(f"Error while prompt validation: {val_err.errors()[0]['msg']}")
