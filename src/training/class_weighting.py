from collections import Counter
from typing import Iterable, Tuple, Optional

import torch
from torch import nn
from transformers import BertForTokenClassification

from entities import LABEL_IDS
from utils.logging_util import logger


def compute_class_weights_from_labels(
        train_labels: Iterable[str],
        scheme: str = "balanced",  # "balanced" or "effective"
        beta: float = 0.999,  # used if scheme == "effective"
        clamp: Optional[Tuple[float, float]] = None,
        normalize: str = "mean1",  # "mean1" or "none"
) -> torch.Tensor:
    """
    Build a weight vector weights[class_id] from the frequency of BIO labels in the *training* split.
    """

    # Count labels present in training data
    cnt = Counter(train_labels)

    # Ensure all labels appear in the dict (missing -> 0)
    counts = {lab: cnt.get(lab, 0) for lab in LABEL_IDS.keys()}

    N = sum(counts.values())
    K = len(LABEL_IDS)


    # Avoid zero-divisions
    def safe(n):
        return max(n, 1)


    raw_weights = {}
    if scheme == "balanced":
        # scikit-learn style: N / (K * n_c)
        for lab, idx in LABEL_IDS.items():
            raw_weights[idx] = N / (K * safe(counts[lab])) if N > 0 else 1.0
    elif scheme == "effective":
        # Cui et al., CVPR'19: (1 - beta) / (1 - beta^n_c)
        for lab, idx in LABEL_IDS.items():
            n = safe(counts[lab])
            raw_weights[idx] = (1.0 - beta) / (1.0 - (beta ** n))
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    weights = torch.tensor([raw_weights[i] for i in range(K)], dtype=torch.float)

    # Normalize so average weight is 1.0 (keeps LR scale intuitive)
    if normalize == "mean1":
        weights = weights / weights.mean()

    # Clamp to keep things stable
    if clamp is not None:
        lo, hi = clamp
        for i, w in enumerate(weights):
            if w < lo or w > hi:
                logger.debug(f"Weight {w:.3f} has been clamped to range [{lo}, {hi}]")
        weights = torch.clamp(weights, lo, hi)

    return weights


class WeightedBertForTokenClassification(BertForTokenClassification):
    """
    Weighted version of BertForTokenClassification that used a weighted loss by overriding compute_loss.
    """
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # shape [num_labels] or None


    def compute_loss(self, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = super().forward(**inputs)
        logits = outputs.logits  # [B, T, C]
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            ignore_index=-100
        )
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
