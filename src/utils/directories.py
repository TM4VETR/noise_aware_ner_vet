import os

from dotenv import load_dotenv
load_dotenv()

from utils.logging_util import logger


def get_data_dir() -> str:
    """
    Returns the absolute path to the data directory.
    """

    default_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    data_dir = os.getenv("DATA_DIR_VET")

    if data_dir is None:
        logger.info(f"Environment variable DATA_DIR_VET is not set. Using default directory: '{default_dir}'")
        data_dir = default_dir
        os.makedirs(data_dir, exist_ok=True)

    return str(data_dir)


def get_models_dir() -> str:
    """
    Returns the absolute path to the directory where models are stored.
    """
    default_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    models_dir = os.getenv("MODELS_DIR")

    if models_dir is None:
        logger.info(f"Environment variable MODELS_DIR is not set. Using default directory: '{default_dir}'")
        models_dir = default_dir
        os.makedirs(models_dir, exist_ok=True)

    return str(models_dir)
