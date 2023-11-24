# Utils
import os
import numpy as np
import argparse
import json
import copy
import pickle

# - Rockpool imports
from rockpool.nn.modules.jax import LinearJax
from rockpool.nn.combinators import Sequential
from rockpool.devices.dynapse import (
    DynapSim,
    mapper,
    autoencoder_quantization,
    config_from_specification,
    dynapsim_net_from_config,
)

# NNI Imports
import nni

# Local Imports
from lib.data_loading import load_data
from lib.logging_utils import (
    configure_logger,
)
from lib.nni_utils import get_nni_trial_path

logger = None


def load_data_from_model_path(model_path):
    """
    Loads the input data associated to the model checkpoint passed as input
    Input Params:
      - model_path: Path of the model checkpoint
    """
    # Obtain training metadata
    training_metadata = None
    with open(f"{model_path}/training_metadata.json") as f:
        training_metadata = json.load(f)

    # Load dataset
    input_params = training_metadata["input_params"]

    train_dl, val_dl, test_dl = load_data(**input_params)
    return train_dl, val_dl, test_dl, input_params


def load_model_from_path(model_path):
    """
    Restores the model from the checkpoint files
    Input Params:
      - model_path: Path of the model checkpoint
    """
    # Load simulation parameters
    sim_params = np.load(f"{model_path}/model_sim_params.npy", allow_pickle=True)
    layer_params = sim_params.item()["1_DynapSim"]

    # Obtain pretrained layer weights
    opt_params = np.load(f"{model_path}/model_params.npy", allow_pickle=True)
    w_in_opt = opt_params.item()["0_LinearJax"]["weight"]
    w_rec_opt = opt_params.item()["1_DynapSim"]["w_rec"]

    # Build network
    n_input_channels = input_params["n_channels"]
    n_output_channels = (
        len(input_params["enabled_classes"])
        if input_params["enabled_classes"] != None
        else input_params["n_classes"]
    )

    model = Sequential(
        LinearJax(
            (n_input_channels, n_output_channels), has_bias=False, weight=w_in_opt
        ),
        DynapSim(
            (n_output_channels, n_output_channels),
            has_rec=True,
            w_rec=w_rec_opt,
            **layer_params,
        ),
    )

    logger.info(f"Built network: \n\t{model}")

    return model


def quantize_model(model, tuning_params):
    """
    Quantizes the model after applying Quantization Tuning.
    Returns an hardware configuration for evaluation and the tuned quantized parameters.
    Input Params:
      - model: Model to tune
      - tuning_params: Correction factors for Quantization Tuning
    """
    # Convert network to spec
    net_graph = model.as_graph()
    spec = mapper(net_graph)

    # Fine tune the network weights prior quantization
    # Global weights
    spec["weights_in"] *= tuning_params["global_activity_factor"]
    spec["weights_rec"] *= tuning_params["global_activity_factor"]

    # Per-Class weights
    for c in range(spec["weights_rec"].shape[1]):
        spec["weights_in"][:, c] *= tuning_params[f"c{c}_activity_factor"]
        spec["weights_rec"][:, c] *= tuning_params[f"c{c}_activity_factor"]

    # Quantize the parameters
    spec_Q = copy.deepcopy(spec)
    params_Q = autoencoder_quantization(**spec_Q)
    spec_Q.update(params_Q)

    # Convert spec to DYNAP-SE2 configuration
    config = config_from_specification(**spec_Q)

    # Print the most significative parameters from hardware configuration object
    enabled_params = [
        "SYAM_W0_P",
        "SYAM_W1_P",
        "SYAM_W2_P",
        "SYAM_W3_P",
        "DEAM_ITAU_P",
        "DEAM_ETAU_P",
        "DEAM_IGAIN_P",
        "DEAM_EGAIN_P",
        "SOIF_LEAK_N",
        "SOIF_SPKTHR_P",
        "SOIF_GAIN_N",
        "SOIF_REFR_N",
        "SOIF_DC_P",
    ]

    params = []
    for k, v in config["config"].chips[0].cores[0].parameters.items():
        if k in enabled_params:
            params.append((k, v))

    for param in sorted(params):
        logger.info(
            f"{param[0]} -> (Coarse:{param[1].coarse_value}, Fine:{param[1].fine_value})"
        )

    return config, params_Q


def evaluate_model(model, val_dl):
    """
    Evaluates the model using the provided data.
    Input Params:
      - val_dl: Validation set DataLoader
      - model: Model to be evaluated
    """

    ds = val_dl.dataset
    output, _, _ = model(ds.x.numpy())
    m = np.sum(output, axis=1)
    preds = np.argmax(m, axis=1)
    acc = np.mean(np.array(preds == ds.y.numpy()))

    logger.info(f"Final accuracy: {np.round(acc, decimals=4)*100}%")
    return acc


if __name__ == "__main__":
    # This is a workaround needed because of the old Cuda version running on the machine
    os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

    # CLI config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nni_mode",
        type=bool,
        help="Mandatory if the script is executed with NNI_mode. Enables NNI features as logging, status reporting.",
        default=False,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path of the model to fine-tune for DYNAP-SE2 deployment.",
        required=True,
    )
    args = parser.parse_args()

    # Logger config
    logger = configure_logger(args.nni_mode)

    logger.info(f"Model Path: {args.model_path}")

    train_dl, val_dl, test_dl, input_params = load_data_from_model_path(args.model_path)

    model = load_model_from_path(args.model_path, input_params["dt"])

    tuning_params = {
        "global_activity_factor": 1,
        "c0_activity_factor": 1,
        "c1_activity_factor": 1,
        "c2_activity_factor": 1,
        "c3_activity_factor": 1,
        "c4_activity_factor": 1,
        "c5_activity_factor": 1,
        "c6_activity_factor": 1,
    }

    if args.nni_mode:
        optimized_params = nni.get_next_parameter()
        tuning_params.update(optimized_params)
        logger.info(f"Initial parameters: {tuning_params}")

    config, params_Q = quantize_model(model, tuning_params)

    model = dynapsim_net_from_config(**config)

    acc = evaluate_model(model, val_dl, nni_mode=args.nni_mode)

    if args.nni_mode:
        nni.report_final_result({"default": float(acc)})

    # Save tuning metadata
    tuning_metadata = {
        "params": tuning_params,
        "val_accuracy_after_tuning": acc,
    }

    with open(
        f"{get_nni_trial_path() if args.nni_mode else '.'}/dynapse2_qtuning_metadata",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(tuning_metadata, f, ensure_ascii=False, indent=4)

    # Update model parameters if better score
    tuning_metadata_path = f"{args.model_path}/dynapse2_qtuning_metadata"
    qtuned_params_path = f"{args.model_path}/dynapse2_qtuned_params"
    prev_acc = 0
    if os.path.exists(tuning_metadata_path):
        with open(tuning_metadata_path) as f:
            prev_acc = json.load(f)["val_accuracy_after_tuning"]

    if acc > prev_acc:
        with open(tuning_metadata_path, "w", encoding="utf-8") as f:
            json.dump(tuning_metadata, f, ensure_ascii=False, indent=4)

        with open(qtuned_params_path, "wb") as f:
            pickle.dump(params_Q, f)
