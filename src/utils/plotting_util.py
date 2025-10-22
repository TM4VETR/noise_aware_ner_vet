import os

import matplotlib.pyplot as plt

from utils.logging_util import logger


def plot_and_save(filename: str, **kwargs):
    """ Helper to save a plot to a file. """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        plt.savefig(f"{filename}.{ext}", **kwargs)
        logger.debug(f"Saved plot to {filename}.{ext}.")

    plt.close()
