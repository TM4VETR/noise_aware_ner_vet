"""
Parameters for preprocessing, pretraining, and training.
"""

# --- training ---
EPOCHS = 25
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
BATCH_SIZE = 16
WEIGHT_DECAY = 0.01

# --- pretraining ---
EPOCHS_PT = 50
LEARNING_RATE_PT = 5e-5
WARMUP_RATIO_PT = 0.01
BATCH_SIZE_PT = 128
WEIGHT_DECAY_PT = 0.01

# --- preprocessing ---
DO_STEMMING = False
DO_SIMPLIFY = False

DO_OVERSAMPLING = True
OVERSAMPLING_FACTOR = 3

CLAMP = (0.005, 5)
MAX_LENGTH = 64

# --- misc ---
RANDOM_SEED = 56

def export_params() -> dict:
    """
    Export training parameters.
    :return: Dict with parameters
    """
    return {
        "training": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
        },
        "pretraining": {
            "epochs": EPOCHS_PT,
            "learning_rate": LEARNING_RATE_PT,
            "warmup_ratio": WARMUP_RATIO_PT,
            "batch_size": BATCH_SIZE_PT,
            "weight_decay": WEIGHT_DECAY_PT,
        },
        "preprocessing": {
            "do_stemming": DO_STEMMING,
            "do_oversampling": DO_OVERSAMPLING,
            "do_simplify": DO_SIMPLIFY,
            "clamp": CLAMP,
            "max_length": MAX_LENGTH,
            "oversampling_factor": OVERSAMPLING_FACTOR,
        },
        "misc": {
            "random_seed": RANDOM_SEED,
        }
    }
