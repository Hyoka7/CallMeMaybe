import json
from pathlib import Path

from src.model import JsonResult


def write_results(path: Path, results: list[JsonResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [result.model_dump() for result in results]

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
