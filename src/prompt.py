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
        "assistant. Select one listed function and emit one JSON call; never "
        "execute it, explain, or add text outside the call.\n"
        "For text arguments, copy only the requested value exactly, "
        "preserving "
        "every space, punctuation mark, and quoted character.\n"
        "Regex: emit only the shortest complete reusable regular expression "
        "and stop immediately. Put alternative individual characters in one "
        "character class, use quantifiers for repeated categories, and keep "
        "an exact word unchanged. Use [0-9]+ for number sequences and "
        "[aeiouAEIOU] for vowels. Use * for asterisk, not the word. "
        " Never append matched text, "
        "source, replacement, explanations, alternatives, or surrounding "
        "punctuation.\n"
        "For numeric arguments, "
        "preserve request order and every sign, digit, decimal point, and "
        "exponent, including a leading minus sign."
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
    return build_chat_prompt(system, user)
