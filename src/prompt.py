from src.model import JsonFunction, JsonInput


def build_func_select_prompt(funcs: JsonInput, user_input: str) -> str:
    prompt = (
        "Select the option whose function best matches the user request.\n"
        "Return only the option number. Do not return the function name "
        "or any explanation.\n"
        "Options:\n"
    )
    for index, function in enumerate(funcs.func):
        prompt += (
            f"[{index}] {function.name}: {function.description}\n"
        )
    prompt += f"User request: {user_input}\n"
    prompt += "Option number: "
    return prompt


def build_param_prompt(func: JsonFunction, user_input: str):
    prompt = "Extract the function arguments from the request.\n"
    prompt += f"Selected function: {func.name}\n"
    prompt += f"Function description: {func.description}\n"
    prompt += "Required parameters:\n"
    for k, v in func.parameters.items():
        prompt += f"{k}: {v['type']}\n"
    prompt += (
        "Rules:\n"
        "- Output only one JSON object.\n"
        "- Include every required parameter exactly once.\n"
        "- Do not include extra parameters.\n"
        "- Preserve values from the user's request.\n"
        "- Do not calculate or execute the function.\n"
    )
    prompt += f"Request: {user_input}\n"
    prompt += "Parameters JSON output:\n"

    return prompt
