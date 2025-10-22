import json
import os
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer

from entities import LABELS
from evaluate import compute_metrics_fn
from params import BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO, DO_OVERSAMPLING, CLAMP, export_params, \
    RANDOM_SEED
from training.class_weighting import compute_class_weights_from_labels, WeightedBertForTokenClassification
from training.training_visualization import MetricsPlotterCallback
from utils.dataset_builder import build_dataset
from utils.directories import get_models_dir
from utils.early_stopping import LoggingEarlyStoppingCallback
from utils.logging_util import logger


models_dir = get_models_dir()


def train_model(model_name: str, train_set: List[Tuple[str, str, str]], val_set: List[Tuple[str, str, str]], timestamp: str):
    """
    Trains a model (starting from the pretrained model) and saves it.

    :param model_name: The name of the model
    :param train_set: List of (token, label, token_c) tuples for training
    :param val_set: List of (token, label, token_c) tuples for validation
    :param timestamp: Timestamp string for saving the model
    :return: model, tokenizer
    """
    model_dir = os.path.join(models_dir, timestamp, model_name)
    os.makedirs(model_dir, exist_ok=True)

    train_tokens, train_labels, _ = zip(*train_set)
    val_tokens, val_labels, _ = zip(*val_set)

    # Class weighting
    weights = compute_class_weights_from_labels(
        train_labels,
        scheme="balanced",  # or "effective"
        # beta=0.999,         # only used for "effective"
        clamp=CLAMP,
        normalize="mean1"
    )

    # Pretrained model and tokenizer
    pretrained_model_dir = os.path.join(models_dir, "pretrained")
    assert os.path.isdir(pretrained_model_dir), f"Pretrained model directory not found: {pretrained_model_dir}"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    model = WeightedBertForTokenClassification.from_pretrained(
        pretrained_model_dir,
        num_labels=len(LABELS),
    )
    model.class_weights = weights

    # Label maps
    label2id = {lab: i for i, lab in enumerate(LABELS)}
    id2label = {i: lab for lab, i in label2id.items()}
    model.config.label2id = label2id
    model.config.id2label = id2label

    # Build datasets
    train_dataset = build_dataset(tokenizer, tokens=train_tokens, labels=train_labels, oversample=DO_OVERSAMPLING)
    val_dataset = build_dataset(tokenizer, tokens=val_tokens, labels=val_labels, oversample=False)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, pad_to_multiple_of=8)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    if not use_cuda:
        logger.warning("Using a GPU (CUDA) is strongly recommended for training!")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,

        eval_strategy="epoch",
        save_strategy="epoch",

        # pick the best checkpoint at the end
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Learning rate
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="inverse_sqrt",
        warmup_ratio=WARMUP_RATIO,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,

        optim="adamw_torch_fused" if use_cuda else "adamw_torch",
        weight_decay=WEIGHT_DECAY,

        label_smoothing_factor=0.05,

        # Logging (TensorBoard)
        logging_strategy="steps",
        logging_steps=100,
        report_to=["tensorboard"],
        run_name=os.path.basename(model_dir),

        # speedups
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_cuda,  # allow TF32 for GPUs
        group_by_length=True,
        dataloader_num_workers=min(8, (os.cpu_count() or 2) // 2),

        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        seed=RANDOM_SEED
    )

    # Save training_args and params to model_dir
    with open(os.path.join(model_dir, "training_args_pytorch.json"), "w", encoding="utf-8") as f:
        f.write(training_args.to_json_string())
    with open(os.path.join(model_dir, "training_params.json"), "w", encoding="utf-8") as f:
        json.dump(export_params(), f, ensure_ascii=False, indent=2)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_fn,

        # Callbacks
        callbacks=[
            LoggingEarlyStoppingCallback(
                early_stopping_patience=5,  # stop after 5 evals without improvement
                early_stopping_threshold=1e-4  # small margin to avoid stopping on noise
            ),
            MetricsPlotterCallback(save_dir=os.path.join(model_dir, "eval_plots"))
        ],
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(model_dir)  # save_pretrained also saves weights and config
    tokenizer.save_pretrained(model_dir)

    logger.info(f"Training successful. Model saved in: {model_dir}.")

    return model, tokenizer
