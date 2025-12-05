def pack_by_token_budget(tokenizer, tokens, labels, max_length):
    """
    Packs tokens and labels into chunks that fit within the token budget of max_length after tokenization.

    Args:
        tokenizer: The tokenizer
        tokens: Tokens
        labels: Labels
        max_length: Maximum token length

    Returns:
        Tuple[List[str], List[str]]: Lists of token chunks and label chunks
    """
    chunks_t, chunks_l = [], []
    i = 0
    while i < len(tokens):
        j = i + 1
        while j <= len(tokens):
            enc = tokenizer([tokens[i:j]], is_split_into_words=True, add_special_tokens=True, truncation=False, padding=False)
            if len(enc["input_ids"][0]) <= max_length:
                j += 1
            else:
                break

        if j > len(tokens):
            j = len(tokens)

        if j == i:
            j = i + 1

        # minimal patch: prefer breaking on 'O' when the chunk end hits inside an entity
        end = j - 1
        if labels[end] != "O":
            back = end
            while back > i and labels[back] != "O":
                back -= 1
            if labels[back] == "O" and back >= i + 1:
                j = back + 1  # cut after the O-token

        chunks_t.append(tokens[i:j])
        chunks_l.append(labels[i:j])
        i = j

    return chunks_t, chunks_l


def pack_by_token_budget_tokens(tokenizer, tokens, max_length):
    """
    Packs tokens into chunks that fit within the token budget of max_length after tokenization.

    :param tokenizer: The tokenizer
    :param tokens: Tokens
    :param max_length: Maximum token length
    :return: (tok_chunks, lab_chunks) after packing
    """
    chunks_t = []
    i = 0
    while i < len(tokens):
        j = i + 1
        while j <= len(tokens):
            enc = tokenizer([tokens[i:j]], is_split_into_words=True, add_special_tokens=True, truncation=False, padding=False)
            if len(enc["input_ids"][0]) <= max_length:
                j += 1
            else:
                break

        if j > len(tokens):
            j = len(tokens)

        if j == i:
            j = i + 1

        chunks_t.append(tokens[i:j])
        i = j

    return chunks_t
