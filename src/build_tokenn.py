from llm_sdk import Small_LLM_Model
from src.model import JsonInput


def encode_funcs(llm: Small_LLM_Model, funcs: JsonInput):
    res = []
    for func in funcs.func:
        encoded_ids = llm.encode(func.name)[0].tolist()
        res.append(encoded_ids)
    return res


def allow_token(encoded_func: list[list[int]], current_token: list[int]):
    cur_token_len = len(current_token)
    allow = set()
    for func in encoded_func:
        slice_func = func[:cur_token_len]
        if slice_func == current_token and len(func) > cur_token_len:
            allow.add(func[cur_token_len])
    return allow
