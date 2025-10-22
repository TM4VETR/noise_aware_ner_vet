from typing import List

from entities import LABELS


def replace_unknown_labels(labels: List[str]) -> List[str]:
    """
    Maps any label not in allowed_labels to "O".

    :param labels: List of labels
    :return: Modified list of labels
    """

    allowed_labels = set(LABELS)
    return [lab if lab in allowed_labels else "O" for lab in labels]
