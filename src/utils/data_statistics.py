from typing import List

from entities import ENTITIES
from utils.data_handler import load_all_from_ids
from utils.logging_util import logger


def print_data_statistics(all_document_ids: List[str]) -> None:
    """
    Prints statistics for the annotated data.

    :param all_document_ids: List of all document IDs (each page becomes a document)
    """

    tokens, labels, _ = zip(*load_all_from_ids(all_document_ids))

    b_labels = [l for l in labels if l.startswith("B-")]

    logger.debug("ANNOTATED DATA:")
    logger.debug(f"  Document pages: {len(all_document_ids)}")
    logger.debug(f"  Entities: {len(b_labels)}")  # Only count "B-" tags to count each multi-word entity once
    for entity in ENTITIES:
        logger.debug(f"    {entity}: {count_entities(entity, labels)}")

    find_longest_multi_word_entity(tokens, labels)
    find_most_frequent_token(tokens, labels)


def find_longest_multi_word_entity(tokens: List[str], labels: List[str]) -> int:
    """
    Finds the length of the longest multi of multi-word entity in the annotated data.

    :param tokens: List of all tokens
    :param labels: List of all BIO labels
    :return: Length of the longest multi-word entity
    """
    bgn = -1
    end = -1
    l = 0

    max_len = 0
    max_idx_bgn = -1
    max_idx_end = -1

    for i, label in enumerate(labels):
        label = labels[i]
        if label.startswith("B-"):
            l = 1
            bgn = i
        elif label.startswith("I-"):
            l += 1
            end = i
        else:
            l = 0

        # Store max
        if l > max_len:
            max_len = l
            max_idx_bgn = bgn
            max_idx_end = end

    if max_len > 0:
        mwe = " ".join(tokens[max_idx_bgn:max_idx_end + 1])
        logger.debug(f"Longest multi-word entity: '{mwe}' ({max_len} words).")

    return max_len


def find_most_frequent_token(tokens: List[str], labels: List[str]) -> str:
    """
    Finds the most frequent token per entity.

    :param tokens: List of all tokens
    :param labels: List of all BIO labels
    :return: The most frequent token
    """

    for entity in ENTITIES:
        freq = {}

        for i in range(len(labels)):
            label = labels[i]
            if label.endswith(entity) and (label.startswith("B-")):
                token = tokens[i]

                # Append multi-word entities
                while i + 1 < len(labels) and labels[i + 1].startswith("I-"):
                    token += " " + tokens[i + 1]
                    i += 1

                freq[token] = freq.get(token, 0) + 1

        # Sort by frequency
        freq = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))

        key, value = next(iter(freq.items()))
        logger.debug(f"Most frequent token for entity {entity}: '{key}' ({value} occurrences).")
        #logger.debug(f"All tokens:\n {freq}")


def count_entities(entity: str, labels: List[str]) -> int:
    """
    Counts the number of occurrences of a given entity in a list of BIO labels.

    :param entity: The entity
    :param labels: List of labels
    :return: Number of occurrences
    """
    i = 0
    for label in labels:
        if label == f"B-{entity}":  # Only count "B-" tags to count each multi-word entity once
            i += 1

    return i
