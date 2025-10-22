import os
from types import SimpleNamespace
from typing import List, Dict

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, DataCollatorForTokenClassification

from entities import LABELS
from error_analysis.ner_confusion import plot_token_confusion, plot_span_confusion
from params import BATCH_SIZE
from utils.dataset_builder import build_dataset
from utils.directories import get_models_dir


@torch.no_grad()
def evaluate_model(
        model=None,
        tokenizer=None,
        model_name: str = "",
        test_set: List[str] = None,
        timestamp: str = "",
) -> Dict[str, float]:
    """
    Evaluate a given model.
    Returns a dict with loss, precision, recall, f1.
    """
    assert test_set, "test_set must be provided"

    models_dir = get_models_dir()
    model_dir = os.path.join(models_dir, timestamp, model_name)

    # Load model/tokenizer if not provided
    if model is None or tokenizer is None:
        tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        model = BertForTokenClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Build test dataset
    test_tokens, test_labels, _ = zip(*test_set)

    # Build datasets
    test_dataset = build_dataset(tokenizer, test_tokens, test_labels, oversample=False)

    # DataLoader (no shuffle)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    pin = torch.cuda.is_available()  # only pin if CUDA
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, pin_memory=pin)

    all_logits = []
    all_labels = []
    losses = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)  # returns .logits and (since labels present) .loss
        if getattr(out, "loss", None) is not None:
            losses.append(out.loss.detach().cpu().item())
        all_logits.append(out.logits.detach().cpu().numpy())
        all_labels.append(batch["labels"].detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    label_ids = np.concatenate(all_labels, axis=0)

    y_true, y_pred = _to_seq_lists(label_ids, logits)

    metrics = compute_metrics_fn(SimpleNamespace(predictions=logits, label_ids=label_ids))
    if losses:
        metrics = {"loss": float(np.mean(losses)), **metrics}

    # Write metrics to file
    with open(os.path.join(model_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    plot_token_confusion(y_true, y_pred, out_png=os.path.join(model_dir, "confusion", "cm_tokens_test"))
    plot_span_confusion(y_true, y_pred, out_png=os.path.join(model_dir, "confusion", "cm_spans_test"))

    return metrics


def compute_metrics_fn(pred):
    """
    Compute precision, recall, f1 for a single prediction using BIO tagging.

    :param pred The prediction
    :return: Dict with precision, recall, f1
    """
    logits = pred.predictions[0] if isinstance(pred.predictions, (tuple, list)) else pred.predictions
    preds = logits.argmax(-1)
    labels = pred.label_ids

    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []

    for gold_seq, pred_seq in zip(labels, preds):
        t_seq, p_seq = [], []
        for l, p in zip(gold_seq, pred_seq):
            if l == -100:
                continue
            t_seq.append(LABELS[int(l)])
            p_seq.append(LABELS[int(p)])
        y_true.append(t_seq)
        y_pred.append(p_seq)

    # overall
    metrics: Dict[str, float] = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # per-entity (flat keys so HF can log them)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for k in ("micro avg", "macro avg", "weighted avg"):
        report.pop(k, None)
    metrics.update({f"f1_{ent}": v["f1-score"] for ent, v in report.items()})

    return metrics


def _to_seq_lists(label_ids, logits):
    """Map ids → BIO tags and ignore -100 for seqeval."""
    preds = logits.argmax(-1)
    y_true, y_pred = [], []
    for gold_seq, pred_seq in zip(label_ids, preds):
        t_seq, p_seq = [], []
        for l, p in zip(gold_seq, pred_seq):
            if l == -100:
                continue
            t_seq.append(LABELS[int(l)])
            p_seq.append(LABELS[int(p)])
        y_true.append(t_seq)
        y_pred.append(p_seq)
    return y_true, y_pred


if __name__ == "__main__":
    # Example usage:
    evaluate_model(
        model_name="noisy",
        test_set=list(zip(["This", "is", "a", "test", "."], ["O", "O", "O", "O", "O"])),
        timestamp="20250915-155921"
    )

    # Evaluate all three models
    # data_dir = get_data_dir()
    # all_document_ids = load_all_document_ids(data_dir)  # Each annotated page "page<i>_annotated_ner.txt" is considered a document
    #
    # all_tokens_labels = load_all_from_ids(all_document_ids)  # flat lists of tokens and labels
    #
    # # Train / test / validation split (70:20:10)
    # train_set, test_set, val_set = split_train_test_validation(all_tokens_labels, ratio=(70, 20, 10), batch_size=BATCH_SIZE, val_cap=10000)
    #
    # for model_name in ["noisy"]:
    #     # for model_name in ["noisy", "clean", "artificial"]:
    #     evaluate_model(
    #         model_name=model_name,
    #         test_set=test_set,
    #         timestamp="final"
    #     )
