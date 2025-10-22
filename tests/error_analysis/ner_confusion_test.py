import os

import matplotlib
matplotlib.use("Agg", force=True) # Use non-interactive backend for CI

from error_analysis.ner_confusion import plot_token_confusion

def test_plot_token_confusion(tmp_path):

    # Minimal sample data: two sequences of BIO tags
    y_true = [
        ["B-JOB_TITLE", "I-JOB_TITLE", "O", "B-SKILL"],
        ["B-ACTIVITY", "O", "B-SUBJECT"],
    ]
    y_pred = [
        ["B-JOB_TITLE", "I-JOB_TITLE", "O", "B-ACTIVITY"],
        ["B-ACTIVITY", "O", "B-SUBJECT"],
    ]

    out_png = os.path.join(tmp_path, "confusion")
    plot_token_confusion(y_true, y_pred, out_png)

    assert os.path.isfile(out_png + ".png")
    assert os.path.isfile(out_png + ".pdf")
