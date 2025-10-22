def one_word_per_line(text: str) -> str:
    """
    Splits the text to one word per line:
    - Split on whitespace (spaces, tabs, etc.)
    - Keep special chars (brackets, hyphens, punctuation, etc.) as they might be part of the token.
    """
    out_lines = []
    for ln in text.splitlines():
        if ln.strip() != "":
            for raw_tok in ln.split():  # split on any whitespace (spaces, tabs, ...)
                out_lines.append(raw_tok)

    return out_lines
