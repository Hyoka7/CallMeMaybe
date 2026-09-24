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
        "preserving every space and punctuation mark inside the "
        "quotation marks. The enclosing quotation marks only delimit "
        "the value and must not be included. The source_string begins "
        "and ends with the exact spaces inside those delimiters."
        "MUST NOT LEAVE ANYTHING OUT! if you violate this rule, "
        "you will be punished!!\n"
        "Regex/replacement examples:\n"
        'numbers with NUMBERS -> regex "\\d+", replacement '
        '"NUMBERS"\n'
        "vowels with asterisks -> regex "
        '"a|e|i|o|u|A|E|I|O|U", replacement '
        '"*"\n'
        'word cat with dog -> regex "cat", replacement "dog"\n'
        "For numbers use regex \\d+ exactly once; close immediately "
        "after + with no | or repeated pattern.\n"
        "For numeric arguments, "
        "preserve request order and every sign, digit, decimal point, and "
        "exponent, including a leading minus sign. if type definition is "
        "'number', '3' must be output as a form '3.0', not '3'. "
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
