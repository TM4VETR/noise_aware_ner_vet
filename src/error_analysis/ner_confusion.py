from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
import sklearn.metrics

from entities import ENTITIES
from utils.logging_util import logger
from utils.plotting_util import plot_and_save


MISS = "(miss)"

custom_blues = LinearSegmentedColormap.from_list(
    "custom_blues",
    ["#f7fbff", "#dbe8fb", "#aac6f5", "#6a9be6", "#2f72bd", "#0f3a6e"],
    N=256,
    gamma=0.9,
)


def collapse_label(tag: str) -> str:
    """ Collapses tags """
    # "B-ORG" -> "ORG"; "I-PER" -> "PER"; "O" -> "O"
    if tag == "O":
        return "O"

    if "-" in tag:
        return tag.split("-", 1)[1]

    return tag


def extract_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    """
    BIO -> list of spans as (start, end_inclusive, TYPE).
    Exact-match evaluation uses these boundaries.
    """
    spans = []
    start, etype = None, None
    for i, t in enumerate(tags + ["O"]):  # sentinel O to flush last span
        if t.startswith("B-") or t == "O":
            if etype is not None:
                spans.append((start, i - 1, etype))
                start, etype = None, None
            if t.startswith("B-"):
                start = i
                etype = t.split("-", 1)[1]
        elif t.startswith("I-"):
            cur = t.split("-", 1)[1]
            if etype is None:
                # ill-formed: treat as B-
                start = i
                etype = cur
            elif cur != etype:
                # type switched: close previous and start new
                spans.append((start, i - 1, etype))
                start, etype = i, cur
    return spans

# (1) token-level confusion (no 'O')
def plot_token_confusion(y_true: List[List[str]], y_pred: List[List[str]], out_png: str):
    """
    Build a token-level confusion matrix between entity TYPES (not BIO),
    excluding 'O' to avoid dominance.
    Rows: true labels, columns: predicted labels.
    """
    try:
        # collapse BIO -> TYPE
        true_flat, pred_flat = [], []
        for t_seq, p_seq in zip(y_true, y_pred):
            for t, p in zip(t_seq, p_seq):
                if t == "O" and p == "O":
                    continue  # skip O/O
                t_c = collapse_label(t)
                p_c = collapse_label(p)
                if t_c == "O" or p_c == "O":
                    continue
                true_flat.append(t_c)
                pred_flat.append(p_c)

        if not true_flat:
            print("No non-'O' tokens to plot.")
            return

        # shorten label for plotting
        labels = ENTITIES
        labels = [l.replace("GROUP", "GR.") for l in labels]
        true_flat = [l.replace("GROUP", "GR.") for l in true_flat]
        pred_flat = [l.replace("GROUP", "GR.") for l in pred_flat]

        cm = sklearn.metrics.confusion_matrix(true_flat, pred_flat, labels=labels, normalize="true")

        with matplotlib.rc_context({
            "axes.labelsize": 8, # axis labels
            "axes.titlesize": 8,
            "figure.titlesize": 8,
            "xtick.labelsize": 8, # tick labels
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.size": 8,
        }):
            fig, ax = plt.subplots(figsize=(6, 4))

            disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(
                ax=ax,
                values_format=".1%",  # one decimal, with %
                colorbar=True,
                cmap=custom_blues,
                xticks_rotation=45,
                text_kw={"fontsize": 6}
            )

            # Tighten/resize text for axes labels explicitly
            ax.set_xlabel("Predicted label", fontsize=8)
            ax.set_ylabel("True label", fontsize=8)

            # Make the colorbar ticks match (0–100%, one decimal)
            cb = getattr(disp.im_, "colorbar", None)
            if cb is not None:
                cb.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
                cb.ax.tick_params(labelsize=7)

            plot_and_save(out_png, dpi=300)
    except Exception as e:
        logger.error(f"Error plotting token confusion: {e}")


# (2) span-level confusion (entity vs. entity; exact match)
def plot_span_confusion(y_true: List[List[str]], y_pred: List[List[str]], out_png: str):
    """
    Build a span-level confusion matrix between entity TYPES (not BIO).
    """
    try:
        col_names = ENTITIES + [MISS]
        row_names = ENTITIES

        counts = np.zeros((len(row_names), len(col_names)), dtype=int)
        idx_row = {t: i for i, t in enumerate(row_names)}
        idx_col = {t: i for i, t in enumerate(col_names)}

        for t_seq, p_seq in zip(y_true, y_pred):
            gold_spans = {(s, e): et for (s, e, et) in extract_spans(t_seq)}
            pred_spans = {(s, e): et for (s, e, et) in extract_spans(p_seq)}

            matched = set()
            for (s, e), gtype in gold_spans.items():
                if (s, e) in pred_spans:
                    ptype = pred_spans[(s, e)]
                    counts[idx_row[gtype], idx_col[ptype]] += 1
                    matched.add((s, e))
                else:
                    counts[idx_row[gtype], idx_col[MISS]] += 1

        # row-normalize to percentages
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums = counts.sum(axis=1, keepdims=True)
            norm = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums != 0)

        # plot
        plt.figure(figsize=(6, 4 + 0.2 * len(ENTITIES)))
        plt.imshow(norm, aspect="auto")
        plt.colorbar()
        plt.xticks(np.arange(len(col_names)), col_names, rotation=45, ha="right")
        plt.yticks(np.arange(len(row_names)), row_names)
        plt.xlabel("Predicted entity")
        plt.ylabel("Ground truth entity")
        for i in range(len(row_names)):
            for j in range(len(col_names)):
                if row_sums[i] > 0:
                    plt.text(j, i, f"{counts[i,j]}", ha="center", va="center", fontsize=8)

        plot_and_save(out_png, dpi=300)
    except Exception as e:
        logger.error(f"Error plotting span confusion: {e}")
