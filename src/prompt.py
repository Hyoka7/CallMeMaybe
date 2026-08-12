import json

from src.model import JsonFunction, JsonInput


def build_chat_prompt(
    system: str,
    user: str,
    assistant_prefix: str = "",
) -> str:
    """Build a Qwen chat prompt with thinking disabled."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        f"{assistant_prefix}"
    )


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


def build_param_prompt(
    func: JsonFunction,
    user_input: str,
    regex_parameters: set[str] | None = None,
):
    regex_parameters = regex_parameters or set()
    prompt = "Extract the function arguments from the request.\n"
    prompt += f"Selected function: {func.name}\n"
    prompt += f"Function description: {func.description}\n"
    prompt += "Required parameters:\n"
    for name, definition in func.parameters.items():
        prompt += f"{name}: {definition['type']}\n"

    prompt += (
        "Rules:\n"
        "- Output only one JSON object.\n"
        "- Include every required parameter exactly once.\n"
        "- Preserve input values from the user's request.\n"
        "- Do not calculate, transform, or execute the function.\n"
        "- Do not include extra parameters.\n"
        f"Request: {user_input}\n"
        "Parameters JSON output:\n"
    )
    return prompt


def build_regex_prompt(
    func: JsonFunction,
    parameter_name: str,
    user_input: str,
) -> str:
    system = (
        "You translate a matching intent into one reusable regular expression. "
        "Return one JSON object containing only the requested pattern. Never "
        "copy individual matches or replacement content. A set of alternative "
        "single characters must be written as one bracketed character class."
    )
    assistant_start = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    key = json.dumps(parameter_name)
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\nMatch every numeric sequence in 'A12 B345'."
        "<|im_end|>\n"
        f'{assistant_start}{{{key}: "[0-9]+"}}<|im_end|>\n'
        "<|im_start|>user\nMatch every vowel in 'Example sentence'."
        "<|im_end|>\n"
        f'{assistant_start}{{{key}: "[aeiouAEIOU]"}}<|im_end|>\n'
        "<|im_start|>user\nMatch every exact occurrence of 'bird'."
        "<|im_end|>\n"
        f'{assistant_start}{{{key}: "bird"}}<|im_end|>\n'
        "<|im_start|>user\n"
        f"Function description: {func.description}\n"
        f"Pattern parameter: {parameter_name}\n"
        f"Request: {user_input}\n"
        "Return one applicable pattern only."
        "<|im_end|>\n"
        f"{assistant_start}"
    )
    return prompt + "{" + key + ': "'
