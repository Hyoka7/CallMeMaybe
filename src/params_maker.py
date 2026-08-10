import heapq
import itertools
import json
import math
import re

import numpy as np

from llm_sdk import Small_LLM_Model
from src.build_tokenn import allow_token
from src.model import JsonFunction

STRING_TOKEN_CACHE: dict[object, set[int]] = {}
MAX_STRING_TOKENS = 32
MAX_NUMBER_TOKENS = 16
STRING_BEAM_WIDTH = 2
STRING_END_LOGIT_MARGIN = 3.0

StringBeam = tuple[float, int, list[int], list[int]]


def append_json_token(
    llm: Small_LLM_Model, prompt_ids: list[int], output_ids: list[int], literal
):
    literal_id = llm.encode(literal)[0].tolist()
    prompt_ids.extend(literal_id)
    output_ids.extend(literal_id)


def build_num_token_list(llm: Small_LLM_Model):
    tokens = set()
    for i in range(10):
        num_token = llm.encode(str(i))[0].tolist()
        if len(num_token) == 1:
            tokens.add(num_token[0])
    return tokens


def is_safe_string_token(token_text: str) -> bool:
    if not token_text:
        return False
    return all(
        character.isprintable() and character not in {'"', "\\", "\ufffd"}
        for character in token_text
    )


def build_chr_token_list(llm: Small_LLM_Model, vocab_size: int) -> set[int]:
    cached = STRING_TOKEN_CACHE.get(llm)
    if cached is not None:
        return cached

    tokens = set()
    for token_id in range(vocab_size):
        token_text = llm.decode([token_id])
        if is_safe_string_token(token_text):
            tokens.add(token_id)

    STRING_TOKEN_CACHE[llm] = tokens
    return tokens


def make_string_tokens(
    llm: Small_LLM_Model,
    prompt_ids: list[int],
    initial_logits,
    allowed_tokens: set[int],
    end_token: int,
    beam_width: int = STRING_BEAM_WIDTH,
    max_tokens: int = MAX_STRING_TOKENS,
    regex_format: bool = False,
) -> list[int]:
    """Generate one JSON string value with constrained beam search."""
    counter = itertools.count()
    beams: list[StringBeam] = [
        (0.0, next(counter), [], prompt_ids.copy())
    ]
    completed: list[StringBeam] = []

    def completion_score(beam: StringBeam) -> tuple[float, float]:
        if regex_format:
            return (-float(len(beam[2])), beam[0])
        return (beam[0], beam[0])

    def is_valid_completion(beam: StringBeam) -> bool:
        if not regex_format:
            return True
        pattern = llm.decode(beam[2])
        if not pattern:
            return False
        try:
            re.compile(pattern)
        except re.error:
            return False
        return True

    for step in range(max_tokens):
        next_beams: list[StringBeam] = []
        natural_completions: list[StringBeam] = []
        for score, _, tokens, current_prompt in beams:
            if step == 0:
                logits = initial_logits
            else:
                logits = llm.get_logits_from_input_ids(current_prompt)
            candidates = allowed_tokens | {end_token}
            candidate_logits = np.asarray(
                [logits[token_id] for token_id in candidates],
                dtype=np.float64,
            )
            log_total = float(np.logaddexp.reduce(candidate_logits))

            end_logit = float(logits[end_token])
            if math.isfinite(end_logit):
                completion: StringBeam = (
                    score + end_logit - log_total,
                    next(counter),
                    tokens,
                    current_prompt,
                )
                completed.append(completion)

                # A small model often ranks one ordinary character just above
                # the closing quote and then continues indefinitely.  Once a
                # non-empty value has been produced, accept a quote that is
                # already close to the best permitted continuation.
                best_text_logit = max(
                    float(logits[token_id]) for token_id in allowed_tokens
                )
                if (
                    tokens
                    and is_valid_completion(completion)
                    and best_text_logit - end_logit <= STRING_END_LOGIT_MARGIN
                ):
                    natural_completions.append(completion)

            next_token_ids = heapq.nlargest(
                beam_width,
                allowed_tokens,
                key=logits.__getitem__,
            )
            for token_id in next_token_ids:
                token_logit = float(logits[token_id])
                if not math.isfinite(token_logit):
                    continue
                new_beam: StringBeam = (
                    score + token_logit - log_total,
                    next(counter),
                    tokens + [token_id],
                    current_prompt + [token_id],
                )
                heapq.heappush(next_beams, new_beam)
                if len(next_beams) > beam_width:
                    heapq.heappop(next_beams)

        if natural_completions:
            valid_completed = [
                beam
                for beam in completed
                if is_valid_completion(beam)
            ]
            return max(valid_completed, key=completion_score)[2]

        beams = next_beams
        if not beams:
            break
        if completed and max(beam[0] for beam in completed) >= max(
            beam[0] for beam in beams
        ):
            break

    if completed:
        valid = [
            beam for beam in completed if is_valid_completion(beam)
        ]
        non_empty = [beam for beam in completed if beam[2]]
        candidates = valid or non_empty or completed
        return max(candidates, key=completion_score)[2]
    if beams:
        return max(beams, key=lambda beam: beam[0])[2]
    return []


def params_maker(
    llm: Small_LLM_Model,
    prompt_ids: list[int],
    func: JsonFunction,
):
    output = []
    append_json_token(llm, prompt_ids, output, "{")
    params = list(func.parameters.items())
    params_len = len(params)
    for index, (name, types) in enumerate(params):
        if index > 0:
            append_json_token(llm, prompt_ids, output, ",")
        append_json_token(llm, prompt_ids, output, json.dumps(name))
        append_json_token(llm, prompt_ids, output, ": ")
        cur_tokens = []
        if types["type"] == "number":
            cur_tokens = []
            allowed_tokens = build_num_token_list(llm)
            end_tokens = set()
            while len(cur_tokens) < MAX_NUMBER_TOKENS:
                if len(cur_tokens) > 0 and len(end_tokens) == 0:
                    if index == params_len - 1:
                        end_tokens.add(llm.encode("}")[0].tolist()[0])
                    else:
                        end_tokens.add(llm.encode(",")[0].tolist()[0])
                logits = llm.get_logits_from_input_ids(prompt_ids)
                candicates = allowed_tokens | end_tokens
                cand_id = max(
                    candicates,
                    key=logits.__getitem__,
                )
                if cand_id in end_tokens:
                    break
                prompt_ids.append(cand_id)
                output.append(cand_id)
                cur_tokens.append(cand_id)
        elif types["type"] == "string":
            append_json_token(llm, prompt_ids, output, '"')
            logits = llm.get_logits_from_input_ids(prompt_ids)
            allowed_tokens = build_chr_token_list(llm, len(logits))
            end_token = llm.encode('"')[0].tolist()[0]
            string_tokens = make_string_tokens(
                llm,
                prompt_ids,
                logits,
                allowed_tokens,
                end_token,
                beam_width=(STRING_BEAM_WIDTH if params_len == 1 else 1),
                max_tokens=MAX_STRING_TOKENS,
            )
            prompt_ids.extend(string_tokens)
            output.extend(string_tokens)
            append_json_token(llm, prompt_ids, output, '"')
        elif types["type"] == "boolean":
            boolean_tokens = [
                llm.encode("true")[0].tolist(),
                llm.encode("false")[0].tolist(),
            ]
            cur_tokens = []
            while cur_tokens not in boolean_tokens:
                allowed = allow_token(boolean_tokens, cur_tokens)
                if not allowed:
                    raise RuntimeError("Allowed boolean tokens are empty")
                logits = llm.get_logits_from_input_ids(prompt_ids)
                cand_id = max(
                    allowed,
                    key=logits.__getitem__,
                )
                prompt_ids.append(cand_id)
                cur_tokens.append(cand_id)
                output.append(cand_id)
    append_json_token(llm, prompt_ids, output, "}")
    generated = llm.decode(output)
    return generated
