import re

from nltk.stem.snowball import GermanStemmer  # type: ignore

STEMMER = GermanStemmer(ignore_stopwords=False).stem


# Replace German umlauts/ß after lowercasing
UMLAUT_MAP = str.maketrans({
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "ß": "ss",
})

# Pattern for gender specific endings:
GENDER_SUFFIXES = re.compile(
    r"""
    (
      ([-/_:\*\(]?)         # optional separators or opening parenthesis 
      (in|innen)            # the gender token
      \)?                   # optional closing parenthesis
    )$
    """,
    re.VERBOSE,
)


def strip_gender(token: str) -> str:
    """
    Remove common German gender-specific forms.
    (e.g., "Schneider/in" -> "schneider")
    """
    out = GENDER_SUFFIXES.sub("", token)
    return out


def simplify(token: str, do_stemming: bool = True) -> str:
    """
    Simplify a single token.

    (1) Basic operations: lowercase, umlaut/ß replacement, strip whitespace, remove punctuation (keep internal hyphens)
    (2) Stemming:
        - Strip common German gendered endings
        - Apply Snowball stemming (if available)

    Args:
        token: The input token
        do_stemming: Apply stemming or not

    Returns:
        str: The simplified token.
    """
    if token is None:
        return ""

    # (1) Basic text operations
    t = token.lower().strip().translate(UMLAUT_MAP)

    # Normalize non-breaking hyphen to regular hyphen
    t = t.replace("\u2011", "-")

    # Strip leading/trailing punctuation while keeping internal hyphens
    start, end = 0, len(t)
    while start < end and not (t[start].isalnum() or t[start] == "-"):
        start += 1
    while end > start and not (t[end - 1].isalnum() or t[end - 1] == "-"):
        end -= 1
    t = t[start:end].strip("-")

    if not t:
        return ""

    # (2) Stemming and gender-normalization
    t = strip_gender(t)
    if do_stemming:
        t = STEMMER(t)

    return t
