def oversample_positive_chunks(tok_chunks, lab_chunks, factor):
    """
    Oversample positive chunks (not "O") by a given factor.

    :param tok_chunks: Token chunks
    :param lab_chunks: Label chunks
    :param factor: Factor to oversample by

    :return: (tok_chunks, lab_chunks) after oversampling
    """
    pos_idx = [i for i, labs in enumerate(lab_chunks) if any(l != "O" for l in labs)]
    for _ in range(factor - 1):
        for i in pos_idx:
            tok_chunks.append(tok_chunks[i])
            lab_chunks.append(lab_chunks[i])
    return tok_chunks, lab_chunks
