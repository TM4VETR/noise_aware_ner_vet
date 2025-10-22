"""
Utility functions to build a dataset with Hugging Face structure as required for transformer training.
"""

from typing import List, Dict, Any
from datasets import Dataset

from entities import LABEL_IDS
from training.simplification import simplify
from training.chunking import pack_by_token_budget
from training.oversampling import oversample_positive_chunks
from params import DO_SIMPLIFY, DO_STEMMING, MAX_LENGTH, OVERSAMPLING_FACTOR


def tokenize_and_align_labels_batch(
    tokenizer,
    batch_tokens: List[List[str]],
    batch_labels: List[List[str]],
    max_length: int,
) -> Dict[str, Any]:
    """
    Tokenizes a batch of token lists and aligns the BIO labels with the tokenized output.
    :param tokenizer: The tokenizer
    :param batch_tokens: Batch of token lists
    :param batch_labels: Batch of label lists
    :param max_length: Maximum sequence length
    :return: Dict of lists: input_ids, attention_mask, labels
    """

    if len(batch_tokens) != len(batch_labels):
        raise ValueError(f"batch size mismatch: {len(batch_tokens)=} vs {len(batch_labels)=}")

    for idx, (toks, labs) in enumerate(zip(batch_tokens, batch_labels)):
        if len(toks) != len(labs):
            raise ValueError(f"example {idx}: tokens/labels length mismatch ({len(toks)} vs {len(labs)})")

    # Tokenize the batch
    # Important: NO return_tensors here; keep Python lists so we can wrap in a Dataset easily
    enc = tokenizer(
        batch_tokens,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    aligned: List[List[int]] = []
    for i, labels in enumerate(batch_labels):
        word_ids = enc.word_ids(batch_index=i)
        label_ids_row: List[int] = []
        for widx in word_ids:
            if widx is None:
                label_ids_row.append(-100)  # special tokens
            else:
                lab = labels[widx]
                try:
                    label_ids_row.append(LABEL_IDS[lab])
                except KeyError:
                    allowed = list(LABEL_IDS.keys())
                    raise ValueError(f"Unknown label '{lab}'. Allowed labels: {allowed}")
        aligned.append(label_ids_row)

    enc["labels"] = aligned  # lists of ints; Trainer and data collator will turn into tensors
    return enc  # a plain dict of lists


def build_hf_dataset(encodings_dict: Dict[str, List]) -> Dataset:
    """
    Builds a dataset with Hugging Face structure from a dict of encodings;

    :param encodings_dict:  lists of equal length: input_ids, attention_mask, labels (and optionally token_type_ids)
    :return: The dataset
    """
    lengths = {k: len(v) for k, v in encodings_dict.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Field lengths differ: {lengths}")

    return Dataset.from_dict(encodings_dict)


def build_dataset(tokenizer, tokens: List[str], labels: List[str], oversample: bool):
    """
    Build datasets. Performs several steps as simplification, chunking, oversampling, tokenization.

    :param tokenizer: The tokenizer
    :param tokens: The tokens
    :param labels: The labels
    :param oversample: Do oversample positive chunks or not (only for training!)

    :return: Dataset
    """

    # (1) Simplify (if enabled)
    if DO_SIMPLIFY:
        tokens = [simplify(t, do_stemming=DO_STEMMING) for t in tokens]

    # (2) Pack by token budget
    tok_chunks, lab_chunks = pack_by_token_budget(tokenizer, tokens, labels, max_length=MAX_LENGTH)

    # (3) Oversample positive chunks in training set
    if oversample:
        tok_chunks, lab_chunks = oversample_positive_chunks(tok_chunks, lab_chunks, factor=OVERSAMPLING_FACTOR)

    # (4) Tokenize + align
    max_len = getattr(tokenizer, "model_max_length", MAX_LENGTH) or MAX_LENGTH  # max_len: subword token budget AFTER tokenization
    enc = tokenize_and_align_labels_batch(tokenizer, tok_chunks, lab_chunks, max_length=max_len)

    # (5) wrap as real datasets
    dataset = build_hf_dataset(enc)

    return dataset
