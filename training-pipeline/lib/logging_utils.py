import logging
from pathlib import Path
import sys
import json
import numpy as np


def configure_logger(nni_mode):
    """
    Configures the root logger.
    Input Params:
      - nni_mode: Controls the logs destination (True: Logs to nni trial file, False: Logs to stdout )
    """
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    handler = None
    if nni_mode:
        from nni_utils import get_nni_trial_path

        log_file_path = get_nni_trial_path()
        Path(log_file_path).mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(f"{log_file_path}/trial.log")
    else:
        handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def save_lifnet_model_summary(model, config, val_accuracy, path):
    """
    Stores a model checkpoint.
    Input Params:
      - model: Model to save the checkpoint for
      - config: Model hyperparameters
      - val_accuracy: Accuracy of the model on the validation set
      - path: Path in which the model checkpoint will be saved
    """
    Path(path).mkdir(parents=True, exist_ok=True)

    model.save(f"{path}/model_params")

    with open(f"{path}/model_metadata.json", "w", encoding="utf-8") as f:
        model_metadata = {
            "net": str(model),
            "params": config,
            "val_accuracy": round(float(val_accuracy), 3),
        }
        json.dump(model_metadata, f, ensure_ascii=False, indent=4)


def save_dynapsimnet_model_summary(model_params, model, config, val_accuracy, path):
    """
    Stores a model checkpoint.
    Input Params:
      - model_params: Model trained parameters
      - model: Model to save the checkpoint for
      - config: Model hyperparameters
      - val_accuracy: Accuracy of the model on the validation set
      - path: Path in which the model checkpoint will be saved
    """
    Path(path).mkdir(parents=True, exist_ok=True)

    np.save(f"{path}/model_params", model_params, allow_pickle=True)
    np.save(
        f"{path}/model_sim_params", model.simulation_parameters(), allow_pickle=True
    )

    with open(f"{path}/model_metadata.json", "w", encoding="utf-8") as f:
        model_metadata = {
            "net": str(model),
            "params": config,
            "val_accuracy": round(float(val_accuracy), 3),
        }
        json.dump(model_metadata, f, ensure_ascii=False, indent=4)


def save_training_metadata(
    input_params, lr, num_epochs, use_scheduled_lr, nni_mode, loss_t, acc_t, path
):
    """
    Saves a training summary.
    Input Params:
      - input_params: Object containing information about the input data
      - lr: Learning rate
      - num_epochs: Duration of the training in number of epochs
      - use_scheduled_lr: Boolean, checks if the training employed a scheduled learning rate
      - nni_mode: Boolean, checks if the training was ran in NNI mode
      - loss_t: List of the loss values for each epoch
      - acc_t: List of the validation accuracy values for each epoch
      - path: Path in which the training summary will be saved
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(f"{path}/training_metadata.json", "w", encoding="utf-8") as f:
        training_metadata = {
            "input_params": input_params,
            "loss_t": loss_t,
            "acc_t": acc_t,
            "lr": lr,
            "num_epochs": num_epochs,
            "use_scheduled_lr": use_scheduled_lr,
            "nni_mode": nni_mode,
        }
        json.dump(training_metadata, f, ensure_ascii=False, indent=4)
