from enum import Enum
import json
import os
import random
from typing import List, Dict, Any, Optional

from params import RANDOM_SEED
from utils.logging_util import logger


random.seed(RANDOM_SEED)

# To inject random characters
_ALPHA_NUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _load_typical_errors() -> List[Dict[str, Any]]:
    """
    Loads typical OCR errors from the JSON file.

    Note: Run OCR error analysis beforehand!
    """

    error_analysis_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ocr_error_analysis.json")
    try:
        with open(error_analysis_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        ocr_errors = []
        for rec in data:
            t = (rec.get("type") or "").strip().lower()
            gt = (rec.get("gt") or "")
            ocr = (rec.get("ocr") or "")
            cnt = int(rec.get("count", 1))
            if t in {"substitution", "deletion", "insertion"} and cnt > 0:
                ocr_errors.append({"type": t, "gt": gt, "ocr": ocr, "count": cnt})

        return ocr_errors
    except Exception as e:
        logger.error(f"Failed to load OCR error analysis data: {e}")
        return []


_TYPICAL_ERRORS: List[Dict[str, Any]] = _load_typical_errors()
_TYP_WEIGHTS: List[int] = [rec["count"] for rec in _TYPICAL_ERRORS]


class ErrorType(Enum):
    """ Error types """
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"
    TYPICAL = "typical"


def _substitute_char(word: str) -> str:
    """ Substitutes one character. """
    if not word:
        # On empty input, fall back to insertion to guarantee one change
        return _insert_char(word)
    idx = random.randrange(len(word))
    ch = word[idx]

    # random different single character
    repl = ch
    while repl == ch:
        repl = random.choice(_ALPHA_NUM)

    return word[:idx] + repl + word[idx + 1:]


def _delete_char(word: str) -> str:
    """ Deletes one character. """
    if not word:
        # Nothing to delete; fall back to insertion to guarantee one change
        return _insert_char(word)
    idx = random.randrange(len(word))
    return word[:idx] + word[idx + 1:]


def _insert_char(word: str) -> str:
    """ Inserts one character. """
    pos = random.randrange(len(word) + 1)
    ch = random.choice(_ALPHA_NUM)
    return word[:pos] + ch + word[pos:]


def _inject_typical_error(word: str) -> str:
    """ Injects a typical error into the word.

    Args:
        word (str): The input word.

    Returns:
        str: The word with a typical error injected, or the original word if no error could be applied.
    """
    if not _TYPICAL_ERRORS:
        logger.warning(f"No typical OCR errors loaded; skipping.")
        return word

    max_tries = 10
    for _ in range(max_tries):
        rec = random.choices(_TYPICAL_ERRORS, weights=_TYP_WEIGHTS, k=1)[0]
        changed = _apply_typical_error_once(word, rec)
        if changed is not None and changed != word:
            return changed

    return word


def _find_all_occurrences(s: str, sub: str) -> List[int]:
    """Return all start indices of sub in s."""
    if sub == "":
        return []
    out: List[int] = []
    start = 0
    while True:
        i = s.find(sub, start)
        if i == -1:
            break
        out.append(i)
        start = i + 1
    return out


def _apply_typical_error_once(word: str, rec: Dict[str, Any]) -> Optional[str]:
    """
    Applies a single typical OCR error to the word.
    Returns the modified word if applicable; otherwise None.
    """
    etype = rec["type"]
    gt = rec.get("gt", "")
    ocr = rec.get("ocr", "")

    if etype == "substitution":
        if not gt:
            return None
        positions = _find_all_occurrences(word, gt)
        if not positions:
            return None
        i = random.choice(positions)
        return word[:i] + ocr + word[i + len(gt):]

    elif etype == "deletion":
        if not gt:
            return None
        positions = _find_all_occurrences(word, gt)
        if not positions:
            return None
        i = random.choice(positions)
        return word[:i] + word[i + len(gt):]

    elif etype == "insertion":
        if not ocr:
            return None
        pos = random.randrange(len(word) + 1)
        return word[:pos] + ocr + word[pos:]

    return None


def inject_error(word: str, error_type: ErrorType) -> str:
    """
    Injects an OCR-like error of type `error_type` into `word`.

    Args:
        word (str): The input word.
        error_type (ErrorType): The type of error to inject.

    Returns:
        str: The word with the injected error.
    """

    if error_type is ErrorType.SUBSTITUTION:
        return _substitute_char(word)
    if error_type is ErrorType.DELETION:
        return _delete_char(word)
    if error_type is ErrorType.INSERTION:
        return _insert_char(word)
    if error_type is ErrorType.TYPICAL:
        return _inject_typical_error(word)

    # Should not happen, but keep safe:
    logger.warning(f"Unknown error type {error_type}; returning original word.")
    return word


if __name__ == "__main__":
    for s in [
        "Margherita",
        "Salami",
        "Quattro Formaggi",
        "Prosciutto e Funghi",
        "Diavola",
        "Capricciosa",
        "Quattro Stagioni",
        "Napolitana",
        "Vegetariana",
        "Calzone",
        "",
    ]:
        print(f"{inject_error(s)}")
