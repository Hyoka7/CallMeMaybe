from llm_sdk import Small_LLM_Model
from src.cli import parse_args
from src.loader import load_functions, load_prompts
from src.prompt import build_system_prompt


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
    prompt = build_system_prompt(funcs, "What is sum 2 and 3?")
    prompt_ids = model.encode(prompt)[0].tolist()
    prompt_ids.append(8822)  # 8822 = fn
    logits = model.get_logits_from_input_ids(prompt_ids)
    allowed_ids = [2891, 1889, 43277, 3062, 5228]
    next_id = max(allowed_ids, key=logits.__getitem__)
    print(repr(model.decode([next_id])))


if __name__ == "__main__":
    main()
