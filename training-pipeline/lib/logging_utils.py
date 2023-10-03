import logging
from pathlib import Path
import sys
import json
import numpy as np

from nni_utils import get_nni_trial_path


def configure_logger(nni_mode):
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    handler = None
    if nni_mode:
        log_file_path = get_nni_trial_path()
        Path(log_file_path).mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(f"{log_file_path}/trial.log")
    else:
        handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def save_lifnet_model_summary(net, config, val_accuracy, path):
    Path(path).mkdir(parents=True, exist_ok=True)

    net.save(f"{path}/model_params")

    with open(f"{path}/model_metadata.json", 'w', encoding='utf-8') as f:
        model_metadata = {"net": str(net),
                          "params": config,
                          "val_accuracy": round(float(val_accuracy), 3)}
        json.dump(model_metadata, f, ensure_ascii=False, indent=4)


def save_dynapsimnet_model_summary(model_params, net, config, val_accuracy, path):
    Path(path).mkdir(parents=True, exist_ok=True)

    np.save(f"{path}/model_params", model_params, allow_pickle=True)
    np.save(f"{path}/model_sim_params",
            net.simulation_parameters(), allow_pickle=True)

    with open(f"{path}/model_metadata.json", 'w', encoding='utf-8') as f:
        model_metadata = {"net": str(net),
                          "params": config,
                          "val_accuracy": round(float(val_accuracy), 3)}
        json.dump(model_metadata, f, ensure_ascii=False, indent=4)


def save_training_metadata(input_params, lr, num_epochs, use_scheduled_lr, nni_mode, loss_t, acc_t, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(f"{path}/training_metadata.json", 'w', encoding='utf-8') as f:
        training_metadata = {"input_params": input_params,
                             "loss_t": loss_t,
                             "acc_t": acc_t,
                             "lr": lr,
                             "num_epochs": num_epochs,
                             "use_scheduled_lr": use_scheduled_lr,
                             "nni_mode": nni_mode,
                             }
        json.dump(training_metadata, f, ensure_ascii=False, indent=4)
