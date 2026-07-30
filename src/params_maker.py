import json
import string

from llm_sdk import Small_LLM_Model
from src.build_tokenn import allow_token
from src.model import JsonFunction


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


def build_chr_token_list(llm: Small_LLM_Model):
    tokens = set()
    alphas = string.ascii_letters
    for c in alphas:
        chr_token = llm.encode(c)[0].tolist()
        if len(chr_token) == 1:
            tokens.add(chr_token[0])
    return tokens


def params_maker(llm: Small_LLM_Model, prompt_ids: list[int], func: JsonFunction):
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
            while True:
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
            cur_tokens = []
            allowed_tokens = build_chr_token_list(llm)
            end_tokens = llm.encode('"')[0].tolist()[0]
            while True:
                logits = llm.get_logits_from_input_ids(prompt_ids)
                candicates = allowed_tokens | {end_tokens}
                cand_id = max(
                    candicates,
                    key=logits.__getitem__,
                )
                if cand_id == end_tokens:
                    break
                prompt_ids.append(cand_id)
                output.append(cand_id)
                cur_tokens.append(cand_id)
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
