import json
import sys

from llm_sdk import Small_LLM_Model
from src.cli import parse_args
from src.constrained_decoder import ConstrainedDecoder, Vocabulary
from src.json_to_file import write_results
from src.loader import load_functions, load_prompts
from src.model import JsonResult
from src.prompt import build_call_prompt


def run() -> int:
    """Select functions and generate schema-constrained arguments."""
    args = parse_args()
    model = Small_LLM_Model()
    funcs = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)
    vocabulary = Vocabulary.from_sdk(model)
    decoder = ConstrainedDecoder(model=model, vocabulary=vocabulary)
    results: list[JsonResult] = []
    for item in prompts.prompts:
        selected, parameters = decoder.generate_call(
            build_call_prompt(funcs, item.prompt),
            funcs.func,
            item.prompt,
        )
        print(f"{item.prompt} -> {selected.name}")
        print(json.dumps(parameters))
        results.append(JsonResult(
            prompt=item.prompt,
            name=selected.name,
            parameters=parameters,
        ))
    write_results(args.output, results)
    return 0


def main() -> int:
    """Run the application and translate failures into process exit codes."""
    try:
        return run()
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        return 130
    except MemoryError:
        print("Aborting: insufficient memory.", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"Aborting: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
