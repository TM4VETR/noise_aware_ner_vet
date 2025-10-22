import random

from entities import LABELS
from utils.data_handler import split_train_test_validation


SIZE = 1000  # Size of the mock dataset

# Tolerance for size checks
TOL = 0.05

ratio = (70, 20, 10)

tokens = [f"token{i}" for i in range(SIZE)]
labels = [random.choice(list(LABELS)) for _ in range(SIZE)]
tokens_c = [None for _ in range(SIZE)]
all_tokens_labels = list(zip(tokens, labels, tokens_c))


def test_split_train_test_validation():
    """ Tests split_train_test_validation(). """
    train_set, test_set, val_set = split_train_test_validation(all_tokens_labels, ratio, batch_size=32, val_cap=-1)

    # Check sizes
    assert abs(len(train_set) - ratio[0] / 100 * len(labels)) <= TOL * len(labels)
    assert abs(len(test_set) - ratio[1] / 100 * len(labels)) <= TOL * len(labels)
    assert abs(len(val_set) - ratio[2] / 100 * len(labels)) <= TOL * len(labels)

    # Check disjoint
    all_splits = train_set + test_set + val_set
    assert len(set(all_splits)) == len(all_splits), "Splits should be disjoint"

    # Check full coverage
    assert sorted(all_splits) == sorted(all_tokens_labels), "Splits should cover all IDs"


def test_split_train_test_validation_capped():
    """ Tests split_train_test_validation() with capping. """
    train_set, test_set, val_set = split_train_test_validation(all_tokens_labels, ratio, batch_size=32, val_cap=64)

    assert len(set(val_set)) == 64, "Validation set should be capped at 64"
