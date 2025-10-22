from transformers.trainer_callback import EarlyStoppingCallback

from utils.logging_util import logger


class LoggingEarlyStoppingCallback(EarlyStoppingCallback):
    """ Subclass of EarlyStoppingCallback that logs when early stopping is triggered. """


    def on_evaluate(self, args, state, control, **kwargs):
        # run the normal early-stopping logic
        super().on_evaluate(args, state, control, **kwargs)

        # if early stopping just triggered, log a helpful message
        if control.should_training_stop:
            metric_name = args.metric_for_best_model or "eval_loss"
            best_ckpt = getattr(state, "best_model_checkpoint", None)
            best_metric = getattr(state, "best_metric", None)
            logger.info(
                f"Early stopping triggered at epoch={state.epoch:.2f}, "
                f"step={state.global_step}. Best {metric_name}={best_metric} "
                f"at checkpoint: {best_ckpt}"
            )
