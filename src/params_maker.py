import json

from llm_sdk import Small_LLM_Model
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
        tokens.add(llm.encode(str(i))[0].tolist())
    return tokens


def params_maker(llm: Small_LLM_Model, prompt_ids: list[int], func: JsonFunction):
    output = []
    append_json_token(llm, prompt_ids, output, "{")
    params = list(func.parameters.items())
    for index, (name, types) in enumerate(params):
        if index > 0:
            append_json_token(llm, prompt_ids, output, ",")
        append_json_token(llm, prompt_ids, output, json.dumps(name))
        append_json_token(llm, prompt_ids, output, ":")
        logits = llm.get_logits_from_input_ids(prompt_ids)
        cand_id = max(range(len(logits)), key=logits.__getitem__)
        prompt_ids.append(cand_id)
        output.append(cand_id)
    append_json_token(llm, prompt_ids, output, "}")
    generated = llm.decode(output)
    return generated
