# Utils
import argparse
import copy
import json
import numpy as np
import os
import pickle
from tqdm import tqdm

# Rockpool Imports
from rockpool.parameters import Constant

# - Modules required to interface and deploy models to the Xylo HDK
from rockpool.devices.xylo.syns61201 import (
    config_from_specification,
    mapper,
    XyloSim,
)
from rockpool.transform import quantize_methods as q

# NNI Imports
import nni

# Local Imports
from lib.data_loading import load_data
from lib.logging_utils import (
    configure_logger,
)
from lib.xylo_net_adaptation import net_from_params, adapt_network
from lib.nni_utils import get_nni_trial_path

logger = None


def load_data_from_model_path(model_path):
    # Obtain training metadata
    training_metadata = None
    with open(f"{model_path}/training_metadata.json") as f:
        training_metadata = json.load(f)

    # Load dataset
    input_params = training_metadata["input_params"]

    train_dl, val_dl, test_dl = load_data(**input_params)
    return train_dl, val_dl, test_dl, input_params


def load_network_from_path(model_path, dt):
    # Obtain neuron parameters
    model_metadata = None
    with open(f"{model_path}/model_metadata.json") as f:
        model_metadata = json.load(f)

    neuron_parameters = {
        "tau_mem": model_metadata["params"]["tau_mem"],
        "tau_syn": model_metadata["params"]["tau_syn"],
        "bias": Constant(0.0),
        "threshold": Constant(1.0),
        "dt": dt,
    }

    # Build network from model parameters
    model_params = None
    with open(f"{model_path}/model_params") as f:
        model_params = json.load(f)

    net = net_from_params(model_params, neuron_parameters)

    logger.info(f"Built network: \n\t{net}")

    # Load network parameters
    net.load(f"{model_path}/model_params")

    # Adapt network to be compatible with and deployable on Xylo
    adapt_network(net, max_synapses=62)

    return net


def quantize_network(net, tuning_params):
    # Convert network to spec
    spec = mapper(
        net.as_graph(),
        weight_dtype="float",
        threshold_dtype="float",
        dash_dtype="float",
    )

    # Fine tune the network weights prior quantization
    # Global weights
    spec["weights_in"] *= tuning_params["global_activity_factor"]
    spec["weights_rec"] *= tuning_params["global_activity_factor"]
    spec["weights_out"] *= tuning_params["global_activity_factor"]

    # Per-Class weights
    for c in range(spec["weights_out"].shape[1]):
        spec["weights_out"][:, c] *= tuning_params[f"c{c}_activity_factor"]

    # Quantise the parameters
    spec_Q = copy.deepcopy(spec)
    params_Q = q.global_quantize(**spec_Q)
    spec_Q.update(params_Q)

    if spec_Q["dash_syn"][0] == 0:
        spec_Q["dash_syn"] += 1
    if spec_Q["dash_syn_out"][0] == 0:
        spec_Q["dash_syn_out"] += 1

    # Convert spec to Xylo configuration
    config, is_valid, m = config_from_specification(**spec_Q)
    if not is_valid:
        err = f"Error detected in spec:\n{m}"
        logger.error(err)
        raise err
    else:
        print("ok")

    return config, params_Q


def evaluate_model(mod, val_dl, nni_mode=False):
    # Evaluate the XyloSim network on the validation set
    ds = val_dl.dataset
    preds = []
    scores = []

    for x, y in tqdm(
        zip(ds.x, ds.y),
        desc="Evaluation",
        unit="Sample",
        total=len(ds),
        disable=nni_mode,
    ):
        output, _, _ = mod(x.numpy(), record=False)
        pred = np.sum(output, axis=0)
        preds.append(np.argmax(pred))
        scores.append(np.argmax(pred) == y.numpy())

    acc = np.sum(scores) / len(scores)

    logger.info(f"Final accuracy: {np.round(acc, decimals=4)*100}%")
    return acc


if __name__ == "__main__":
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
        help="Path of the model to fine-tune for Xylo deployment.",
        required=True,
    )
    args = parser.parse_args()

    # Logger config
    logger = configure_logger(args.nni_mode)

    logger.info(f"Model Path: {args.model_path}")

    train_dl, val_dl, test_dl, input_params = load_data_from_model_path(args.model_path)

    net = load_network_from_path(args.model_path, input_params["dt"])

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

    config, params_Q = quantize_network(net, tuning_params)

    mod = XyloSim.from_config(config, dt=input_params["dt"])

    acc = evaluate_model(mod, val_dl, nni_mode=args.nni_mode)

    if args.nni_mode:
        nni.report_final_result({"default": float(acc)})

    # Save tuning metadata
    tuning_metadata = {
        "params": tuning_params,
        "val_accuracy_after_tuning": acc,
    }

    with open(
        f"{get_nni_trial_path() if args.nni_mode else '.'}/xylo_qtuning_metadata",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(tuning_metadata, f, ensure_ascii=False, indent=4)

    # Update model parameters if better score
    tuning_metadata_path = f"{args.model_path}/xylo_qtuning_metadata"
    qtuned_params_path = f"{args.model_path}/xylo_qtuned_params"
    prev_acc = 0
    if os.path.exists(tuning_metadata_path):
        with open(tuning_metadata_path) as f:
            prev_acc = json.load(f)["val_accuracy_after_tuning"]

    if acc > prev_acc:
        with open(tuning_metadata_path, "w", encoding="utf-8") as f:
            json.dump(tuning_metadata, f, ensure_ascii=False, indent=4)

        with open(qtuned_params_path, "wb") as f:
            pickle.dump(params_Q, f)
