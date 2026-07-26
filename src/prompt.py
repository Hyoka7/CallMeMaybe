from src.model import JsonInput


def build_system_prompt(funcs: JsonInput, user_input: str) -> str:
    prompt = "Select a function that most matches the user input. You must output only the function name.\n"
    for f in funcs.func:
        func_prompt = f"{f.name}: {f.description}\n"
        prompt += func_prompt
    prompt += f"User's input: {user_input}\n"
    prompt += "Function name: "
    return prompt
