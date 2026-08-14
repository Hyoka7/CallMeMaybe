import json

from llm_sdk import Small_LLM_Model
from src.cli import parse_args
from src.constrained_decoder import ConstrainedDecoder, Vocabulary
from src.json_to_file import write_results
from src.loader import load_functions, load_prompts
from src.model import JsonResult
from src.prompt import build_call_prompt


def main() -> None:
    """Select functions and generate schema-constrained arguments."""
    args = parse_args()
    model = Small_LLM_Model()
    try:
        funcs = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
        vocabulary = Vocabulary.from_sdk(model)
    except (OSError, ValueError) as err:
        print(f"Aborting: {err}")
        return
    decoder = ConstrainedDecoder(model, vocabulary)
    print(f"{len(funcs.func)} functions loaded")
    print(f"{len(prompts.prompts)} prompts loaded")
    results: list[JsonResult] = []
    for item in prompts.prompts:
        try:
            selected, parameters = decoder.generate_call(
                build_call_prompt(funcs, item.prompt),
                funcs.func,
                item.prompt,
            )
        except (RuntimeError, ValueError) as err:
            print(f"Aborting prompt {item.prompt!r}: {err}")
            return
        print(f"{item.prompt} -> {selected.name}")
        print(json.dumps(parameters))
        results.append(JsonResult(
            prompt=item.prompt,
            name=selected.name,
            parameters=parameters,
        ))
    write_results(args.output, results)


if __name__ == "__main__":
    main()
