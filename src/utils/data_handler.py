import math
import os
import random
from typing import List, Tuple, Optional

from nat.error_injector import inject_error, ErrorType
from params import RANDOM_SEED
from utils.label_util import replace_unknown_labels
from utils.logging_util import logger


def load_all_document_ids(data_dir):
    """
    Scans the data directory and returns a sorted list of document IDs (filenames).
    Each annotated page "page<i>_annotated_ner.txt" is considered a document.

    Args:
        data_dir: Data directory.
    """
    document_ids = []

    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.lower().endswith("_annotated_ner.txt"):
                document_ids.append(os.path.join(root, filename))

    return sorted(document_ids)


def load_all_document_ids_pretraining(data_dir):
    """
    Scans the data directory and returns a sorted list of document IDs (filenames).
    Each file "*_pretraining.txt" is considered.

    Args:
        data_dir: Data directory.
    """
    document_ids = []

    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.lower().endswith("_pretraining.txt"):
                document_ids.append(os.path.join(root, filename))

    return sorted(document_ids)


def load_all_from_ids(document_ids) -> List[Tuple[str, str, str]]:
    """
    Loads all tokens and labels from a list of document IDs.

    Args:
        document_ids: List of document IDs.

    Returns:
        List[Tuple[str, str, str]]: List of (token, label, token_c) tuples.
    """
    all_tokens = []
    all_labels = []
    all_tokens_c = []  # Corrected tokens

    for document_id in document_ids:
        doc_tokens, doc_labels = load_tokens_labels(document_id)

        if len(doc_tokens) != len(doc_labels):
            logger.warning(f"Mismatch in number of tokens/labels for document {document_id}")
            continue

        # Load clean tokens from corresponding *_ocr_corrected.txt file
        ocr_file = document_id.replace("_annotated_ner.txt", "_ocr_corrected.txt")
        doc_tokens_c = []

        # Fallback if no ocr_file exists
        if not os.path.exists(ocr_file):
            logger.warning(f"OCR-corrected file not found for {document_id}; using annotated tokens without correction.")
            doc_tokens_c = [None] * len(doc_tokens)
        else:
            # read OCR-corrected tokens: each line is "<orig> <corrected>" or just "<orig>"
            ocr_lines: List[str] = []
            with open(ocr_file, "r", encoding="utf-8") as fo:
                for line in fo:
                    # keep trailing spaces (deletion marker), strip only newline
                    s = line.rstrip("\r\n")
                    if not s:
                        continue
                    ocr_lines.append(s)

            if len(ocr_lines) != len(doc_tokens):
                logger.warning(f"Line-count mismatch for {document_id}: OCR={len(ocr_lines)} vs ANN={len(doc_tokens)}. Falling back to annotated tokens.")
                doc_tokens_c = [None] * len(doc_tokens)
            else:
                for ocr_line in ocr_lines:
                    orig, sep, corrected = ocr_line.partition(" ")
                    if sep == "":  # no separator present
                        corrected_tok = None
                    else:
                        corrected_tok = corrected
                        # "<orig><space> " (single space or nothing after sep) considered as deletion (replace orig by separator/nothing)

                    doc_tokens_c.append(corrected_tok)

        assert len(doc_tokens) == len(doc_labels) == len(doc_tokens_c), f"Mismatch in number of tokens/labels/corrected tokens for document {document_id}!"

        all_tokens.extend(doc_tokens)
        all_labels.extend(doc_labels)
        all_tokens_c.extend(doc_tokens_c)

    # Replace unallowed labels
    all_labels = replace_unknown_labels(all_labels)

    assert len(all_tokens) == len(all_labels), "Mismatch in number of tokens/labels!"
    assert len(all_tokens) == len(all_tokens_c), "Mismatch in number of tokens/corrected tokens!"

    return list(zip(all_tokens, all_labels, all_tokens_c))


def inject_errors(all_tokens_labels: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Injects synthetic OCR errors into a list of (token, label, token_corrected) tuples.

    :param all_tokens_labels: List of (token, label, token_corrected) tuples
    :return: List of (token, label, token_corrected) tuples (different length than input due to injected errors)
    """
    all_tokens, all_labels, all_tokens_c = map(list, zip(*all_tokens_labels))

    all_tokens_injected = []
    all_labels_injected = []
    all_tokens_c_injected = []

    #error_types = list(ErrorType) # Inject all types of  errors
    error_types = [ErrorType.TYPICAL]  # Only inject typical errors
    n_error_types = len(error_types)

    for error_type in error_types:
        i = 0
        while i < len(all_tokens):
            token = all_tokens[i]
            label = all_labels[i]
            token_c = all_tokens_c[i]
            if label.startswith("B-") or label == "O":
                # Only inject errors to tokens with "B-" labels, not in "I-" labels, to only inject one error for multi-word entities
                token_error = inject_error(token, error_type)  # token, as OCR correction has been applied before
                all_tokens_injected.append(token_error)
                all_labels_injected.append(label)
                all_tokens_c_injected.append(token_c)

                # Also append corresponding "I-" tokens
                j = i + 1
                while j < len(all_tokens) and all_labels[j].startswith("I-"):
                    all_tokens_injected.append(all_tokens[j])
                    all_labels_injected.append(all_labels[j])
                    all_tokens_c_injected.append(all_tokens_c[j])
                    j += 1

                i = j

    assert len(all_tokens_injected) == len(all_labels_injected), "Mismatch in number of tokens/labels after error injection!"
    assert len(all_tokens_injected) == len(all_tokens_c_injected), "Mismatch in number of tokens/corrected tokens after error injection!"

    assert len(all_tokens_injected) == n_error_types * len(all_tokens), "Mismatch in number of tokens after error injection!"
    assert len(all_labels_injected) == n_error_types * len(all_labels), "Mismatch in number of labels after error injection!"
    assert len(all_tokens_c_injected) == n_error_types * len(all_tokens_c), "Mismatch in number of corrected tokens after error injection!"

    # Add at the end! (to not destroy sequence structure)
    all_tokens.extend(all_tokens_injected)
    all_labels.extend(all_labels_injected)
    all_tokens_c.extend(all_tokens_c_injected)

    return list(zip(all_tokens, all_labels, all_tokens_c))


def correct_ocr_errors(all_tokens_labels: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Corrects OCR errors into a list of (token, label, token_corrected) tuples.

    :param all_tokens_labels: List of (token, label, token_corrected) tuples
    :return: List of (token, label, token_corrected) tuples (same length as input list)
    """
    all_tokens, all_labels, all_tokens_c = map(list, zip(*all_tokens_labels))

    for i in range(len(all_tokens)):
        if all_tokens_c[i] is not None:
            # all_tokens_c[i] contains an OCR correction
            all_tokens[i] = all_tokens_c[i]

    return list(zip(all_tokens, all_labels, all_tokens_c))


def _segment_by_entities(all_tokens_labels: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
    """
    Turn a flat (token, label) list into segments:
      - Each 'B-TYPE' + following 'I-TYPE'... is one segment.
      - Each 'O' token is its own segment (length 1).
    This guarantees we never cut inside a multi-token entity.
    """
    segs: List[List[Tuple[str, str]]] = []
    i = 0
    n = len(all_tokens_labels)


    def _is_I(lbl: str, ent: str) -> bool:
        return lbl.startswith("I-") and (ent == "" or lbl[2:] == ent)


    while i < n:
        _, lab, _ = all_tokens_labels[i]
        if lab == "O":
            segs.append([all_tokens_labels[i]])
            i += 1
            continue

        # start of span (prefer a proper B-, but recover from stray I-)
        if lab.startswith("B-"):
            ent = lab[2:]
        elif lab.startswith("I-"):
            ent = lab[2:]  # recover: treat stray I- as a span start
        else:  # any unknown label -> treat like O to be safe
            # Should never happen
            segs.append([all_tokens_labels[i]])
            i += 1
            continue

        j = i + 1
        while j < n and _is_I(all_tokens_labels[j][1], ent):
            j += 1
        segs.append(all_tokens_labels[i:j])  # one entity segment
        i = j
    return segs


def _floor_to_batch(x: int, batch_size: int) -> int:
    """ Floor to batch size """
    if batch_size > 0 and x >= batch_size:
        return (x // batch_size) * batch_size
    return x


def split_train_test_validation(
        all_tokens_labels: List[Tuple[str, str]],
        ratio: Tuple[int, int, int],
        batch_size: int,
        val_cap: Optional[int] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Span-safe micro-level splitter: Shuffles segments (multi-word entities kept intact; split at "O").

    Args:
        all_tokens_labels: Flat list of (token, label) tuples.
        ratio: Split ratio (train, test, val)
        batch_size: Batch size (keep sizes as multiples of batch size)
        val_cap: Optional token budget for validation; None = no cap.

    Returns:
        train_set, test_set, val_set as flat (token, label) lists.
    """

    assert sum(ratio) == 100, "The sum of the split ratios must be 100."
    segs = _segment_by_entities(all_tokens_labels)
    rnd = random.Random(RANDOM_SEED)
    rnd.shuffle(segs)

    N = len(all_tokens_labels)

    # target tokens by ratio
    tgt_val = math.floor(ratio[2] / 100 * N)
    if val_cap is not None and val_cap >= 0:
        tgt_val = min(tgt_val, int(val_cap))
    rest = N - tgt_val
    tgt_train = math.floor(rest * ratio[0] / (ratio[0] + ratio[1]))  # by tokens

    # Adjust targets to batch size (tokens)
    tgt_val = _floor_to_batch(tgt_val, batch_size)
    tgt_train = _floor_to_batch(tgt_train, batch_size)
    # test will just take the remainder

    # Fill validation up to target tokens (don’t cut segments; don’t exceed target)
    val_segs: List[List[Tuple[str, str]]] = []
    train_segs: List[List[Tuple[str, str]]] = []

    t_val = 0
    idx = 0
    while idx < len(segs) and t_val + len(segs[idx]) <= tgt_val:
        val_segs.append(segs[idx])
        t_val += len(segs[idx])
        idx += 1

    # Fallback: ensure non-empty validation if tgt_val > 0 but nothing fit
    if not val_segs and tgt_val > 0 and idx < len(segs):
        # pick the smallest remaining segment to minimize overshoot
        min_j = min(range(idx, len(segs)), key=lambda j: len(segs[j]))
        val_segs.append(segs[min_j])
        # remove that segment from the pool by swapping into position idx
        if min_j != idx:
            segs[idx], segs[min_j] = segs[min_j], segs[idx]
        idx += 1

    t_train = 0
    while idx < len(segs) and t_train + len(segs[idx]) <= tgt_train:
        train_segs.append(segs[idx])
        t_train += len(segs[idx])
        idx += 1

    # Remainder
    test_segs = segs[idx:]

    # Flatten
    def flatten(segs):
        out: List[Tuple[str, str]] = []
        for s in segs:
            out.extend(s)
        return out

    train_set = flatten(train_segs)
    test_set = flatten(test_segs)
    val_set = flatten(val_segs)

    assert len(train_set) > 0, "Training set is empty!"
    assert len(test_set) > 0, "Test set is empty!"
    assert len(val_set) > 0, "Validation set is empty!"

    assert len(train_set) + len(test_set) + len(val_set) == N
    return train_set, test_set, val_set


def load_tokens_labels(document_id: str) -> Tuple[List[str], List[str]]:
    """
    Loads a linewise annotated file where each line contains a token and its corresponding label.

    Args:
        document_id: Document ID (filename to annotated file).

    Returns:
        Two lists: tokens, labels
    """
    tokens = []
    labels = []
    with open(document_id, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                parts = line.split()
                assert len(parts) == 2, (
                    f"{document_id}:{i}: line '{line}' is not in format '<token> <label>'"
                )
                token, label = parts
                tokens.append(token)
                labels.append(label)

    return tokens, labels
