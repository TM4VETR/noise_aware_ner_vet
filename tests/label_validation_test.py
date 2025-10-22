import os
from pathlib import Path

import pytest

from utils.data_handler import load_all_document_ids, load_all_from_ids
from utils.directories import get_data_dir
from utils.line_splitting import one_word_per_line


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS", "").lower() == "true",
    reason="Skipped in CI because it requires local data."
)
def test_valid_ner_labels():
    """Tests that no 'O' label is followed by an 'I-' label in BIO format."""

    data_dir = get_data_dir()
    all_document_ids = load_all_document_ids(data_dir)

    fail = False
    for doc_id in all_document_ids:
        # Load all labels
        tokens, labels, _ = zip(*load_all_from_ids([doc_id]))

        # Validate BIO format: iterate over all labels and assert no 'O' is followed by 'I-'
        for i in range(len(labels) - 1):
            if labels[i] == "O":
                if labels[i + 1].startswith("I-"):
                    fail = True
                    print(f"{doc_id}. Invalid BIO sequence: 'O' (position: {i}, token: {tokens[i]}) followed by '{labels[i + 1]}' (position: {i + 1}, token: {tokens[i + 1]})!")

    assert not fail, "Failed due to invalid BIO sequences!"


@pytest.mark.skip(reason="Compare annotated OCR file directly to NER file")
@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS", "").lower() == "true",
    reason="Skipped in CI because it requires local data."
)
def test_same_number_of_lines_ner():
    """Tests that the OCR result and the annotated NER file have the same number of lines."""
    data_dir = get_data_dir()

    for page in Path(data_dir).rglob("page*.txt"):
        page_annotated_ner = str(page).replace(".txt", "_annotated_ner.txt")
        if os.path.isfile(page_annotated_ner):
            with open(str(page), "r", encoding="utf-8") as f:
                raw = f.read()
                original_lines = one_word_per_line(raw)

            with open(page_annotated_ner, "r", encoding="utf-8") as f:
                annotated_lines = f.readlines()

            assert len(original_lines) == len(annotated_lines), f"Line count mismatch between {page} and {page_annotated_ner}!"


@pytest.mark.skip(reason="Compare annotated OCR file directly to NER file")
@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS", "").lower() == "true",
    reason="Skipped in CI because it requires local data."
)
def test_same_number_of_lines_ocr():
    """Tests that the OCR result and the annotated NER file have the same number of lines."""
    data_dir = get_data_dir()

    for page in Path(data_dir).rglob("page*.txt"):
        page_corrected_ocr = str(page).replace(".txt", "_ocr_corrected.txt")
        if os.path.isfile(page_corrected_ocr):
            with open(str(page), "r", encoding="utf-8") as f:
                raw = f.read()
                original_lines = one_word_per_line(raw)

            with open(page_corrected_ocr, "r", encoding="utf-8") as f:
                annotated_lines = f.readlines()

            assert len(original_lines) == len(annotated_lines), f"Line count mismatch between {page} and {page_corrected_ocr}!"


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS", "").lower() == "true",
    reason="Skipped in CI because it requires local data."
)
def test_same_number_of_lines_ner_ocr():
    """Tests that the OCR result and the annotated NER file have the same number of lines."""
    data_dir = get_data_dir()

    for page_annotated_ner in Path(data_dir).rglob("page*_annotated_ner.txt"):
        page_corrected_ocr = str(page_annotated_ner).replace("_annotated_ner.txt", "_ocr_corrected.txt")
        if os.path.isfile(page_corrected_ocr):
            with open(str(page_annotated_ner), "r", encoding="utf-8") as f:
                original_lines = f.readlines()

            with open(page_corrected_ocr, "r", encoding="utf-8") as f:
                annotated_lines = f.readlines()

            assert len(original_lines) == len(annotated_lines), f"Line count mismatch between {page_annotated_ner} and {page_corrected_ocr}!"
