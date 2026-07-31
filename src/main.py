import json

from llm_sdk import Small_LLM_Model
from src.build_tokenn import encode_funcs
from src.cli import parse_args
from src.function_selector import funcjson_returner, function_selector
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
    encoded_funcs = encode_funcs(model, funcs)
    results: list[JsonResult] = []
    for pro in prompts.prompts:
        syspro = build_func_select_prompt(funcs, pro.prompt)
        func_id = function_selector(model, syspro, encoded_funcs)
        selected = model.decode(func_id)
        print(f"{pro.prompt} -> {selected}")
        selected_func = funcjson_returner(selected, funcs)
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
