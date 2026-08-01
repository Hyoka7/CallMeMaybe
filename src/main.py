import json

from llm_sdk import Small_LLM_Model
from src.build_tokenn import encode_options
from src.cli import parse_args
from src.function_selector import function_selector
from src.json_to_file import write_results
from src.loader import load_functions, load_prompts
from src.model import JsonResult
from src.params_maker import params_maker
from src.prompt import build_func_select_prompt, build_param_prompt


def main() -> None:
    args = parse_args()
    model = Small_LLM_Model()
    try:
        funcs = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except ValueError as err:
        print(f"Aborting: {err}")
        return
    print(f"{len(funcs.func)} functions loaded")
    print(f"{len(prompts.prompts)} prompts loaded")
    encoded_options = encode_options(model, len(funcs.func))
    results: list[JsonResult] = []
    for pro in prompts.prompts:
        syspro = build_func_select_prompt(funcs, pro.prompt)
        option_ids = function_selector(model, syspro, encoded_options)
        option_index = int(model.decode(option_ids))
        selected_func = funcs.func[option_index]
        selected = selected_func.name
        print(f"{pro.prompt} -> {selected}")
        parampro = build_param_prompt(selected_func, pro.prompt)
        parampro_id = model.encode(parampro)[0].tolist()
        param = params_maker(model, parampro_id, selected_func)
        print(param)
        parameters = json.loads(param)
        results.append(
            JsonResult(
                prompt=pro.prompt,
                name=selected,
                parameters=parameters,
            )
        )

    write_results(args.output, results)


if __name__ == "__main__":
    main()
