import heapq
import itertools

import numpy as np

from llm_sdk import Small_LLM_Model
from src.build_tokenn import allow_token
from src.model import JsonFunction, JsonInput


Beam = tuple[float, int, list[int], list[int]]


def token_log_probabilities(
    logits: list[float], allowed_tokens: set[int]
) -> dict[int, float]:
    """Calculate normalized log probabilities for allowed tokens."""
    token_ids = tuple(allowed_tokens)
    allowed_logits = np.asarray(
        [logits[token_id] for token_id in token_ids],
        dtype=np.float64,
    )
    log_total = float(np.logaddexp.reduce(allowed_logits))
    return {
        token_id: float(logits[token_id] - log_total)
        for token_id in token_ids
    }


def function_selector(
    model: Small_LLM_Model,
    prompt: str,
    encoded_funcs: list[list[int]],
    beam_width: int = 8,
) -> list[int]:
    """Select a function name using constrained beam search."""
    if not encoded_funcs:
        raise ValueError("No encoded function candidates")
    if beam_width < 1:
        raise ValueError("Beam width must be positive")

    prompt_ids = model.encode(prompt)[0].tolist()
    counter = itertools.count()
    beams: list[Beam] = [
        (0.0, next(counter), [], prompt_ids),
    ]
    completed: list[Beam] = []
    width = min(beam_width, len(encoded_funcs))
    max_steps = max(len(tokens) for tokens in encoded_funcs)

    for _ in range(max_steps):
        next_beams: list[Beam] = []

        for score, _, func_tokens, current_prompt_ids in beams:
            allowed_tokens = allow_token(encoded_funcs, func_tokens)
            if not allowed_tokens:
                continue

            logits = model.get_logits_from_input_ids(current_prompt_ids)
            token_scores = token_log_probabilities(logits, allowed_tokens)

            for token_id, token_score in token_scores.items():
                new_func_tokens = func_tokens + [token_id]
                new_prompt_ids = current_prompt_ids + [token_id]
                new_beam: Beam = (
                    score + token_score,
                    next(counter),
                    new_func_tokens,
                    new_prompt_ids,
                )

                if new_func_tokens in encoded_funcs:
                    heapq.heappush(completed, new_beam)

                if allow_token(encoded_funcs, new_func_tokens):
                    heapq.heappush(next_beams, new_beam)
                    if len(next_beams) > width:
                        heapq.heappop(next_beams)

        beams = next_beams
        if not beams:
            break

    if not completed:
        raise RuntimeError("No function candidate was completed")

    best = max(
        completed,
        key=lambda beam: beam[0] / len(beam[2]),
    )
    return best[2]


def funcjson_returner(
    func_name: str, funcs: JsonInput
) -> JsonFunction:
    """Return the definition matching a selected function name."""
    for function in funcs.func:
        if function.name == func_name:
            return function
    raise ValueError("Specified function does not exist")
