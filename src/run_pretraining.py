"""
Run pretraining and evaluation of the pretrained model.
"""
from datetime import datetime

from transformers import AutoTokenizer

from evaluate import evaluate_model
from params import BATCH_SIZE_PT
from pretrain import pretrain_model
from utils.cleaning_util import clean_model_directories
from utils.data_handler import load_all_document_ids_pretraining, split_train_test_validation
from utils.data_handler import load_all_from_ids
from utils.date_util import DATE_FORMAT
from utils.directories import get_data_dir


timestamp = datetime.now().strftime(DATE_FORMAT)


def pretrain_and_evaluate(tokenizer, all_document_ids_pt, base_model, timestamp: str):
    """ Pretrain and evaluate a model. """
    model_name = "pretrained"

    all_tokens_labels = load_all_from_ids(all_document_ids_pt)

    # Train / test / validation split (70:20:10)
    train_set, test_set, val_set = split_train_test_validation(all_tokens_labels, ratio=(70, 20, 10), batch_size=BATCH_SIZE_PT, val_cap=10000)

    model = pretrain_model(tokenizer, train_set, val_set, base_model, timestamp)
    metrics = evaluate_model(model, tokenizer, model_name, test_set, timestamp)

    print(f"Test results for model {model_name}:\n{metrics}")


def run():
    #clean_model_directories()

    # Load annotated data
    data_dir = get_data_dir()
    all_document_ids_pt = load_all_document_ids_pretraining(data_dir)  # Each document "*_pretraining.txt" is considered

    # Tokenizer
    base_model = "dbmdz/bert-base-german-cased"
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    pretrain_and_evaluate(tokenizer, all_document_ids_pt, base_model, timestamp)


if __name__ == "__main__":
    run()
