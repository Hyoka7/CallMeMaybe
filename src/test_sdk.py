from llm_sdk import Small_LLM_Model


def main() -> None:
    model = Small_LLM_Model()

    token_ids = model.encode("Hello")
    print(f"Encoded: {token_ids}")
    print(f"Shape: {token_ids.shape}")

    ids = token_ids[0].tolist()
    logits = model.get_logits_from_input_ids(ids)

    print(f"Token IDs: {ids}")
    print(f"Vocabulary size: {len(logits)}")
    print(f"Highest-logit token ID: {max(range(len(logits)), key=logits.__getitem__)}")


if __name__ == "__main__":
    main()
