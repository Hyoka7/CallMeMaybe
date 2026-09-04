from src.model import JsonInput


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


def build_call_prompt(funcs: JsonInput, user_input: str) -> str:
    """Build one prompt for the complete function call."""
    system = (
        "You are a deterministic function-call compiler, not a conversational "
        "assistant. Select one function and emit its input arguments in the "
        "single JSON call already started. Never execute the function.\n"
        "Before emitting each string value, silently assign that argument one "
        "semantic role: literal input, matching pattern, or ordinary value.\n"
        "For literal input, copy the user's source text exactly.\n"
        "For a matching pattern, emit only the shortest reusable regular "
        "expression representing the requested matches. Stop the value as "
        "soon as that pattern is complete. A set of alternative individual "
        "characters must be one bracketed character class. A repeated "
        "category must use an appropriate quantifier. An exact word remains "
        "that word. Never append matched text, source text, replacement text, "
        "explanations, or unrelated alternatives to a pattern.\n"
        "For an ordinary value, copy only the requested argument value."
    )
    user = "Available functions:\n" + "\n".join(
        f"- {function.name}: {function.description}; arguments: "
        + ", ".join(
            f"{name} ({definition['type']})"
            for name, definition in function.parameters.items()
        )
        for function in funcs.func
    )
    user += f"\n\nRequest: {user_input}"
    return build_chat_prompt(system, user, assistant_prefix='{"name": "')
