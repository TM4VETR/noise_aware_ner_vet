# Noise-Aware Named Entity Recognition for Historical VET Documents

*This repository will be made public.*

This repository contains the code (for training and evaluation) for our paper "Noise-Aware Named Entity Recognition for Historical VET Documents".

## Overview
Historical Vocational Education and Training (VET) documents often suffer from OCR-induced noise, which makes downstream NLP tasks like Named Entity Recognition (NER) challenging.  
This project explores *noise-aware training* strategies and synthetic error injection to improve the robustness of NER models in this domain.

## Citation
If you use this code, please cite:

```bibtex
@inproceedings{Esser2026,
  author    = {Esser, Alexander M. and D{\"o}rpinghaus, Jens},
  title     = {Noise-Aware Named Entity Recognition for Historical VET Documents},
  booktitle = {International Conference on Computer Vision Theory and Applications (VISAPP)},
  year      = {2026}
}
```

## Requirements

### Required packages

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file has been frozen, i.e., for each package the version used has been specified.  
Only for `torch`, the version to be used heavily depends on the system (with/without CUDA, underlying Python and C++ version).

### Required environment variables

The following environment variables are required:

* **DATA_DIR_VET**: Directory for VET data.
* **MODELS_DIR**: Directory to save models.

You can specify these variables in a `.env` file in the root directory, which will be loaded automatically. 
For example:

```env
DATA_DIR_VET=path/to/data/vet
```

## Getting started

After you have

1. installed the required packages and  
2. set the required environment variables,

you can run the following commands to train and evaluate a model.

**Run pre-training** (intermediate model):

```bash
python run_pretraining.py 
```

**Run training** (final models: *noisy*, *clean*, *artificial*):
```bash
python run_training.py
```

When running the above training scripts, the data will be split in a train, test, and validation set.  
The best model of all epochs (based on the validation set) will be saved.  
After training, the model will be evaluated on the test set.