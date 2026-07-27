from llm_sdk import Small_LLM_Model
from src.build_tokenn import allow_token
from src.model import JsonInput


def function_selector(
    model: Small_LLM_Model, prompt: str, encoded_funcs: list[list[int]]
):
    prompt_ids = model.encode(prompt)[0].tolist()
    func_tokens = []
    while True:
        allowed_tokens = allow_token(encoded_funcs, func_tokens)
        if not allowed_tokens:
            raise RuntimeError("Allowed tokens is empty")
        logits = model.get_logits_from_input_ids(prompt_ids)
        cand_id = max(
            allowed_tokens,
            key=logits.__getitem__,
        )
        prompt_ids.append(cand_id)
        func_tokens.append(cand_id)
        if func_tokens in encoded_funcs:
            break
    return func_tokens


def funcjson_returner(func_name: str, funcs: JsonInput):
    for f in funcs.func:
        if f.name == func_name:
            return f
    raise ValueError("Specified Function does not exist")
