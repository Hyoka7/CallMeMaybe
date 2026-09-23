import sys

from tqdm import tqdm

from llm_sdk import Small_LLM_Model
from src.cli import parse_args
from src.generation_engine import ConstrainedDecoder
from src.json_to_file import write_results
from src.loader import load_functions, load_prompts
from src.model import JsonInput, JsonResult
from src.prompt import build_call_prompt
from src.vocabulary import Vocabulary

MAX_INPUT_TOKENS = 256
MAX_FUNCTION_NAME_TOKENS = 64
MAX_PARAMETER_NAME_TOKENS = 64
MAX_DESCRIPTION_TOKENS = 256


def validate_prompt_length(
    model: Small_LLM_Model,
    prompt: str,
    limit: int = MAX_INPUT_TOKENS,
) -> None:
    """Reject user input that exceeds the inference token budget."""
    token_count = len(model.encode(prompt)[0].tolist())
    if token_count > limit:
        raise ValueError(
            f"Prompt has {token_count} tokens; maximum is {limit}."
        )


def validate_function_lengths(
    model: Small_LLM_Model,
    functions: JsonInput,
) -> None:
    """Reject individually oversized function-definition fields."""
    for function in functions.func:
        fields = (
            ("function name", function.name, MAX_FUNCTION_NAME_TOKENS),
            ("description", function.description, MAX_DESCRIPTION_TOKENS),
            *(
                ("parameter name", name, MAX_PARAMETER_NAME_TOKENS)
                for name in function.parameters
            ),
        )
        for label, value, limit in fields:
            token_count = len(model.encode(value)[0].tolist())
            if token_count > limit:
                raise ValueError(
                    f"{label.capitalize()} {value!r} has {token_count} "
                    f"tokens; maximum is {limit}."
                )


def run() -> int:
    """Select functions and generate schema-constrained arguments."""
    args = parse_args()
    funcs = load_functions(args.functions_definition)
    if not funcs.func:
        raise ValueError("No function definition provided.")
    model = Small_LLM_Model()
    validate_function_lengths(model, funcs)
    prompts = load_prompts(args.input)
    if not prompts.prompts:
        raise ValueError("No prompt provided.")
    vocabulary = Vocabulary.from_sdk(model)
    decoder = ConstrainedDecoder(model=model, vocabulary=vocabulary)
    results: list[JsonResult] = []
    for item in tqdm(
        prompts.prompts,
        desc="Generating calls",
        unit="prompt",
    ):
        validate_prompt_length(model, item.prompt)
        selected, parameters = decoder.generate_call(
            build_call_prompt(funcs, item.prompt),
            funcs.func,
            item.prompt,
        )
        results.append(JsonResult(
            prompt=item.prompt,
            name=selected.name,
            parameters=parameters,
        ))
    write_results(args.output, results)
    print(f"Result successfully saved to '{args.output}'.")
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
    except (TypeError, ValueError, OSError, RuntimeError) as err:
        print(f"Aborting: {err}", file=sys.stderr)
        return 1
    except Exception as err:  # noqa: BLE001
        print(
            f"Aborting: unexpected error ({type(err).__name__}): {err}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
