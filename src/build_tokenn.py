from llm_sdk import Small_LLM_Model


def encode_options(
    llm: Small_LLM_Model, option_count: int
) -> list[list[int]]:
    """Encode neutral numeric labels used for function selection."""
    return [
        llm.encode(str(index))[0].tolist()
        for index in range(option_count)
    ]


def allow_token(encoded: list[list[int]], current_token: list[int]):
    cur_token_len = len(current_token)
    allow = set()
    for func in encoded:
        slice_func = func[:cur_token_len]
        if slice_func == current_token and len(func) > cur_token_len:
            allow.add(func[cur_token_len])
    return allow
