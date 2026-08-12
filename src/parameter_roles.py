from llm_sdk import Small_LLM_Model

REGEX_CONFIDENCE_MARGIN = 0.5

def normalize_parameter_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def build_regex_name_prompt(parameter_name: str) -> str:
    normalized_name = normalize_parameter_name(parameter_name)
    return (
        "Classify the meaning of this string parameter name.\n"
        "[0] REGEX: a regular-expression pattern used for matching.\n"
        "[1] NON_REGEX: a value interpreted directly rather than as a "
        "matching expression.\n"
        "Examples:\n"
        "payload -> 1\n"
        "contentvalue -> 1\n"
        "matchexpression -> 0\n"
        "searchpattern -> 0\n"
        f"Parameter name: {normalized_name}\n"
        "Return only the option number.\n"
        "Option number: "
    )


def is_regex_parameter(
    model: Small_LLM_Model,
    parameter_name: str,
) -> bool:
    option_ids = [model.encode(str(index))[0].tolist() for index in range(2)]
    if any(len(ids) != 1 for ids in option_ids):
        raise RuntimeError("Classification options must be single tokens")

    def score(name: str) -> float:
        prompt_ids = model.encode(build_regex_name_prompt(name))[0].tolist()
        logits = model.get_logits_from_input_ids(prompt_ids)
        return float(logits[option_ids[0][0]] - logits[option_ids[1][0]])

    # Remove the model's inherent preference for option 0 or option 1.
    normalized_name = normalize_parameter_name(parameter_name)
    return (
        score(normalized_name) - score("value")
        >= REGEX_CONFIDENCE_MARGIN
    )


def detect_regex_parameters(
    model: Small_LLM_Model,
    parameters: dict[str, dict[str, str]],
) -> set[str]:
    return {
        name
        for name, definition in parameters.items()
        if definition["type"] == "string"
        and is_regex_parameter(model, name)
    }
