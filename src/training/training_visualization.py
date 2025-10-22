import os
import csv
from typing import List, Dict

import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from utils.plotting_util import plot_and_save


class MetricsPlotterCallback(TrainerCallback):
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.train_points: List[Dict] = []
        self.eval_points: List[Dict]  = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return
        logs = dict(logs)
        step  = state.global_step
        epoch = state.epoch

        # collect training loss
        if "loss" in logs:
            self.train_points.append({"step": step, "epoch": epoch, "loss": logs["loss"]})

        # collect eval metrics (HF prefixes them with 'eval_')
        if any(k.startswith("eval_") for k in logs):
            self.eval_points.append({
                "step": step,
                "epoch": epoch,
                "eval_loss": logs.get("eval_loss"),
                "eval_f1": logs.get("eval_f1"),
                "eval_precision": logs.get("eval_precision"),
                "eval_recall": logs.get("eval_recall"),
            })

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        os.makedirs(self.save_dir, exist_ok=True)

        # write CSVs
        if self.train_points:
            with open(os.path.join(self.save_dir, "train_log.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["step", "epoch", "loss"])
                w.writeheader(); w.writerows(self.train_points)

        if self.eval_points:
            with open(os.path.join(self.save_dir, "eval_log.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["step", "epoch", "eval_loss", "eval_f1", "eval_precision", "eval_recall"])
                w.writeheader(); w.writerows(self.eval_points)

        # plots
        if self.train_points:
            epochs = [p["epoch"] for p in self.train_points]
            losses = [p["loss"] for p in self.train_points]
            plt.figure()
            plt.plot(epochs, losses)            # no explicit colors
            plt.xlabel("Step"); plt.ylabel("Train loss");
            plot_and_save(os.path.join(self.save_dir, "train_loss"), dpi=300)

        if self.eval_points:
            epochs = [p["epoch"] for p in self.eval_points if p.get("eval_loss") is not None]
            eval_loss = [p["eval_loss"] for p in self.eval_points if p.get("eval_loss") is not None]
            plt.figure()
            plt.plot(epochs, eval_loss)
            plt.xlabel("Step"); plt.ylabel("Eval loss");
            plot_and_save(os.path.join(self.save_dir, "eval_loss"), dpi=300)

            epochs_f1 = [p["epoch"] for p in self.eval_points if p.get("eval_f1") is not None]
            eval_f1  = [p["eval_f1"] for p in self.eval_points if p.get("eval_f1") is not None]
            if eval_f1:
                plt.figure()
                plt.plot(epochs_f1, eval_f1)
                plt.xlabel("Step"); plt.ylabel("Eval F1");
                plot_and_save(os.path.join(self.save_dir, "eval_f1"), dpi=300)

        # Precision/Recall over time (one figure with two curves)
        epochs_pr = [p["epoch"] for p in self.eval_points if p.get("eval_precision") is not None and p.get("eval_recall") is not None]
        prec = [p["eval_precision"] for p in self.eval_points if p.get("eval_precision") is not None]
        rec = [p["eval_recall"] for p in self.eval_points if p.get("eval_recall") is not None]
        f1 = [p["eval_f1"] for p in self.eval_points if p.get("eval_f1") is not None]
        if prec and rec:
            plt.figure()
            plt.plot(epochs_pr, f1, label="F1", color="#706868")
            plt.plot(epochs_pr, prec, label="Precision", color="#7b3294")
            plt.plot(epochs_pr, rec, label="Recall", color="#d01c8b")
            plt.xlabel("Step");
            plt.ylabel("Score");
            plt.legend()
            plot_and_save(os.path.join(self.save_dir, "eval_precision_recall"), dpi=300)
