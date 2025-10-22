"""
OCR Error Analysis

Scans recursively for pairs of files:
  page<i>_ocr.txt              (OCR result; one word per line)
  page<i>_ocr_corrected.txt    (ground truth; one word per line)

For each line (word), align OCR vs. GT and count character-level errors:
- Substitution: (gt_char -> ocr_char)
- Deletion:     (gt_char -> '')      # character missing in OCR
- Insertion:    ('' -> ocr_char)     # extra character in OCR

Build the confusion dict sorted by descending frequency.
"""

import csv
from collections import Counter
import json
import os
from pathlib import Path
import re
from typing import List, Tuple
import unicodedata

from utils.directories import get_data_dir


RE_OCR = re.compile(r"^page(\d+)_ocr\.txt$", re.IGNORECASE)
RE_GT = re.compile(r"^page(\d+)_ocr_corrected\.txt$", re.IGNORECASE)


def _normalize_token(s: str) -> str:
    """ Normalize a token for character-level comparison """
    s = unicodedata.normalize("NFC", s)
    # Remove non-printing control characters except space, tab
    return "".join(ch for ch in s if ch == " " or ch == "\t" or ch.isprintable())


def char_confusions(gt: str, ocr: str) -> List[Tuple[str, str]]:
    """
    Return a list of (gt_char, ocr_char) pairs representing character-level errors
    between the ground-truth word `gt` and the OCR word `ocr`,
    using a minimal-edit alignment (Levenshtein) with deterministic tie-breaking:

    Hierarchy of operations:
    Prefer deletion > insertion > substitution when costs are equal.

    :param gt: Ground-truth word
    :param ocr: OCR word
    :return: List of (gt_char, ocr_char) pairs for non-equal edits only
    """
    if gt == ocr:
        return []

    gt = _normalize_token(gt)
    ocr = _normalize_token(ocr)

    n, m = len(gt), len(ocr)
    # dp[i][j] = edit distance from gt[:i] to ocr[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # op[i][j] stores the operation chosen to reach (i,j)
    # 'E' diag equal, 'S' substitute, 'D' delete(gt), 'I' insert(ocr)
    op = [[''] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = 'D'
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = 'I'

    for i in range(1, n + 1):
        g = gt[i - 1]
        for j in range(1, m + 1):
            o = ocr[j - 1]
            if g == o:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = 'E'
                continue

            # Costs
            c_sub = dp[i - 1][j - 1] + 1
            c_del = dp[i - 1][j] + 1
            c_ins = dp[i][j - 1] + 1
            best = min(c_sub, c_del, c_ins)

            dp[i][j] = best
            # Deterministic tie-breaking: D > I > S
            if best == c_del:
                op[i][j] = 'D'
            elif best == c_ins:
                op[i][j] = 'I'
            else:
                op[i][j] = 'S'

    # Backtrace
    i, j = n, m
    pairs: List[Tuple[str, str]] = []
    while i > 0 or j > 0:
        cur = op[i][j]
        if cur == 'E':
            i -= 1
            j -= 1
        elif cur == 'S':
            pairs.append((gt[i - 1], ocr[j - 1]))
            i -= 1
            j -= 1
        elif cur == 'D':
            pairs.append((gt[i - 1], ""))  # deletion
            i -= 1
        elif cur == 'I':
            pairs.append(("", ocr[j - 1]))  # insertion
            j -= 1
        else:
            # Should never happen; fallback to diagonal if possible
            if i > 0 and j > 0:
                if gt[i - 1] == ocr[j - 1]:
                    i -= 1
                    j -= 1
                else:
                    pairs.append((gt[i - 1], ocr[j - 1]))
                    i -= 1
                    j -= 1
            elif i > 0:
                pairs.append((gt[i - 1], ""))
                i -= 1
            else:
                pairs.append(("", ocr[j - 1]))
                j -= 1

    pairs.reverse()
    return pairs


def process_gt_page(gt_path: str, counter: Counter) -> Tuple[int, int]:
    """
    Processes a single ground-truth page file "page*_ocr_corrected.txt".
    Returns (num_words, num_errors_found).
    """
    num_words = 0
    num_errs = 0

    with open(gt_path, "r", encoding="utf-8", errors="replace") as f_gt:
        for _, gt_line in enumerate(f_gt):

            num_words += 1

            if " " not in gt_line:
                # No correction present
                continue

            ocr_word, gt_word = gt_line.split(" ", 1)

            if gt_word is None or gt_word.strip() == "":
                # ground truth word deleted - deletion of whole word
                for ch in ocr_word:
                    counter[(ch, "")] += 1
                # Count per-character deletions: (gt_char -> '')
                num_errs += len(ocr_word)
                continue

            ocr_word = ocr_word.rstrip("\n\r")
            gt_word = gt_word.rstrip("\n\r")

            pairs = char_confusions(gt=gt_word, ocr=ocr_word)  # GT vs OCR
            if pairs:
                num_errs += len(pairs)
                for p in pairs:
                    counter[p] += 1

    return num_words, num_errs


def main():
    """ Run OCR error analysis. """
    data_dir = get_data_dir()
    confusions = Counter()

    total_words = 0
    total_errs = 0
    pages = 0

    for gt_page in Path(data_dir).rglob("page*_ocr_corrected.txt"):
        pages += 1
        n_w, n_e = process_gt_page(gt_page, confusions)
        total_words += n_w
        total_errs += n_e

    print(f"Processed ground truth pages: {pages}")
    print(f"Total words compared: {total_words}")
    print(f"Total character-level errors recorded: {total_errs}")

    # Sort confusions by frequency (most common first)
    confusions = Counter(dict(confusions.most_common()))

    # Save confusion data as JSON
    outfile_json = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ocr_error_analysis.json")
    outfile_csv = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ocr_error_analysis.csv")

    rows = []
    for (gt_ch, ocr_ch), cnt in confusions.most_common():
        if gt_ch == "":
            err_type = "insertion"
        elif ocr_ch == "":
            err_type = "deletion"
        else:
            err_type = "substitution"
        rows.append({"gt": gt_ch, "ocr": ocr_ch, "type": err_type, "count": cnt})

    # Write to JSON file
    with open(outfile_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Confusion data written to {outfile_json}.")

    # Write to CSV file
    with open(outfile_csv, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf, delimiter=";")
        w.writerow(["gt", "ocr", "type", "count"])
        for row in rows:
            w.writerow([row["gt"], row["ocr"], row["type"], row["count"]])
    print(f"Confusion data written to {outfile_csv}.")


if __name__ == "__main__":
    main()
