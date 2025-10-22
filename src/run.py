"""
Run training and evaluation for three models: noisy, clean, artificial.
"""
from datetime import datetime
import traceback

from evaluate import evaluate_model
from params import BATCH_SIZE
from train import train_model
from utils.cleaning_util import clean_model_directories
from utils.data_handler import load_all_document_ids, split_train_test_validation
from utils.data_handler import load_all_from_ids, correct_ocr_errors, inject_errors
from utils.data_statistics import print_data_statistics
from utils.date_util import DATE_FORMAT
from utils.directories import get_data_dir
from utils.logging_util import logger


timestamp = datetime.now().strftime(DATE_FORMAT)


def train_and_evaluate(model_name: str, all_document_ids, timestamp: str):
    all_tokens_labels = load_all_from_ids(all_document_ids)  # flat lists of tokens and labels

    # Train / test / validation split (70:20:10)
    train_set, test_set, val_set = split_train_test_validation(all_tokens_labels, ratio=(70, 20, 10), batch_size=BATCH_SIZE, val_cap=10000)

    # Adapt tokens for clean and artificial model
    if model_name == "clean":
        train_set = correct_ocr_errors(train_set)
        val_set = correct_ocr_errors(val_set)
    elif model_name == "artificial":
        train_set = correct_ocr_errors(train_set) # Clean tokens before
        val_set = correct_ocr_errors(val_set)
        train_set = inject_errors(train_set)
        val_set = inject_errors(val_set)

    # Number of tokens and entities (in the training set!)
    tokens, labels, _ = zip(*train_set)
    logger.info(f"Total tokens:\t\t{len(tokens)}")
    logger.info(f"Total entities:\t{len([l for l in labels if l.startswith('B-')])}")

    model, tokenizer = train_model(model_name, train_set, val_set, timestamp)
    metrics = evaluate_model(model, tokenizer, model_name, test_set, timestamp)

    logger.info(f"Test results for model {model_name}:\n{metrics}")


def run():
    # Load annotated data
    data_dir = get_data_dir()
    all_document_ids = load_all_document_ids(data_dir)  # Each annotated page "page<i>_annotated_ner.txt" is considered a document

    # Statistics on annotated data
    print_data_statistics(all_document_ids)

    # Train three models: noisy, clean, artificial
    for model_name in ["noisy", "clean", "artificial"]:
        try:
            logger.info(f"Training model '{model_name}'...")
            train_and_evaluate(model_name, all_document_ids, timestamp)
        except Exception as e:
            logger.error(f"Error training/evaluating model {model_name}: {e}")
            print(traceback.format_exc())


if __name__ == "__main__":
    run()
