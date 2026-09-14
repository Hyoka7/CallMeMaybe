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
        "semantic role: literal input, matching pattern, replacement value, "
        "or ordinary value.\n"
        "For literal input, copy the user's source text exactly. Do not add, "
        "remove, or alter any character, including spaces, punctuation, or "
        "quoted text. Never insert text that is not part of the requested "
        "source value.\n"
        "For a matching pattern, emit only the shortest reusable regular "
        "expression representing the requested matches. Stop the value as "
        "soon as that pattern is complete. A set of alternative individual "
        "characters must be one bracketed character class. A repeated "
        "category must use an appropriate quantifier. An exact word remains "
        "that word. Never append matched text, source text, replacement text, "
        "explanations, or unrelated alternatives to a pattern. Do not append "
        "unrelated punctuation. Do not add source text or replacement text "
        "to the pattern. For example, numbers means [0-9]+, not "
        "2|3|4|5|6|7|8|9|0; individual vowels means [aeiouAEIOU], "
        "not repeated alternatives.\n"
        "For example, all vowels means exactly [aeiouAEIOU] with no "
        "trailing text. The word 'cat' means the exact pattern cat, not cat. "
        "The period in the surrounding sentence is not part of the regex.\n"
        "In the request 'Replace all vowels in Programming is fun with "
        "asterisks', the regex value is [aeiouAEIOU]. In the request "
        "'Substitute the word cat with dog', the regex value is cat.\n"
        "For a replacement value, emit the exact text that should replace "
        "each match. if there is no unit word, output EXACTLY ONE value. "
        "The plural form does NOT signify plurality.\n"
        "If the request names a symbol descriptively, emit that symbol "
        "as the value itself: do not add parentheses, brackets, or "
        "explanatory text. Preserve an explicitly quoted replacement literally"
        ".\nFor example, 'replace vowels with asterisks' means the replacement"
        " value is exactly one '*' character (not the word 'asterisk' or "
        "'asterisks'), while"
        " 'with NUMBERS' means exactly 'NUMBERS'. For a symbolic replacement, "
        "prefer the symbol representation over the descriptive word. For "
        "example, asterisks should be emitted as '**', not the word "
        "'asterisks'. Repeating the symbol is allowed when the request uses "
        "the plural symbolic form.\n"
        "For an ordinary value, copy only the requested argument value."
        " For numeric arguments, preserve the order of numeric values in the "
        "request when assigning them to the function's arguments. Do not "
        "drop a leading minus sign, change a number's sign, or alter its "
        "digits, decimal point, or exponent notation. For example, in a "
        "request containing -1 followed by 345, keep -1 as the first "
        "numeric argument and 345 as the second."
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
