# Utils
import numpy as np
from tqdm import tqdm
import os
import argparse

# Rockpool Imports
from rockpool.nn.modules.jax import LinearJax
from rockpool.nn.combinators import Sequential
from rockpool.devices.dynapse import DynapSim, dynamic_mismatch_prototype
from rockpool.transform.mismatch import mismatch_generator
from rockpool.devices.dynapse.lookup import (
    default_currents,
)

# Jax Imports
import jax
from jax import random as rand
from jax import numpy as jnp
from jax.example_libraries.optimizers import adam, exponential_decay
from rockpool.training import jax_loss as jl

# NNI Imports
import nni

# Local Imports
from lib.data_loading import load_data
from lib.dynapse2_parameters import get_itau_parameter, get_igain_parameter
from lib.logging_utils import (
    configure_logger,
    save_dynapsimnet_model_summary,
    save_training_metadata,
)
from lib.nni_utils import get_nni_trial_path

logger = None


def build_model(
    n_input_channels, n_output_channels, dt, neuron_parameters, load_model_path=None
):
    """
    Builds the model to be trained.
    Input Params:
      - n_input_channels: Number of input channels of the model. Must match the data channels
      - n_output_channels: Number of output channels. Must match the amount of classes in the task
      - dt: Duration of a single timestep. Depends on the input data sampling rate
      - neuron_parameters: Neuron specific parameters (i.e. tau_mem, tau_syn)
      - load_model_path: Load a model checkpoint for resuming training
    """
    neuron_parameters = {
        **neuron_parameters,
        "percent_mismatch": 0.05,
        "dt": dt,
        "has_rec": True,
    }

    # Model loading for resume training
    layer_params = {}
    w_in_opt = None
    w_rec_opt = None
    if load_model_path != None:
        # Load simulation parameters
        sim_params = np.load(
            f"{load_model_path}/model_sim_params.npy", allow_pickle=True
        )
        layer_params = sim_params.item()["1_DynapSim"]

        # Obtain pretrained layer weights
        opt_params = np.load(f"{load_model_path}/model_params.npy", allow_pickle=True)
        w_in_opt = opt_params.item()["0_LinearJax"]["weight"]
        w_rec_opt = opt_params.item()["1_DynapSim"]["w_rec"]

        logger.info(f"Loaded model: {load_model_path}")
    layer_params.update(**neuron_parameters)

    # - Build network
    model = Sequential(
        LinearJax(
            (n_input_channels, n_output_channels), has_bias=False, weight=w_in_opt
        ),
        DynapSim(
            (n_output_channels, n_output_channels),
            w_rec=w_rec_opt,
            **layer_params,
        ),
    )

    logger.info(f"Built network: \n{model}")

    return model


def build_target_signal(y, n_samples, n_timesteps):
    """
    Generates a target signal for training, in which the output neuron associated to the correct label
    fires at each timestep.
    Input Params:
      - y: One-hot encoded labels
      - n_timesteps: Number of timesteps in each sample
    """
    return np.array([[y[i]] * n_timesteps for i in range(len(y))])


@jax.jit
@jax.value_and_grad
def loss_vgf(params, model, input, target):
    """
    Evolves the network with the input samples, evaluate with the target signal and calculates loss and
    gradient.
    Input Params:
      - params: Model parameters
      - model: Model under training
      - input: Batch of samples
      - target: Batch of target signals
    """
    model = model.set_attributes(params)
    model = model.reset_state()
    output, _, _ = model(input)
    return jl.mse(output, target)


def train(
    input_params,
    train_dl,
    val_dl,
    model,
    num_epochs=1500,
    apply_mismatch_after=None,
    mismatch_epochs=100,
    lr=1e-3,
    lr_scheduler_fn=None,
    nni_mode=False,
):
    """
    Runs the training of the passed model.
    Input Params:
      - input_params: An object which includes dataset specific properties
      - train_dl: Training DataLoader
      - val_dl: Validation DataLoader
      - model: Model to train
      - num_epochs: Controls the duration of the training in terms of number of epochs
      - apply_mismatch_after: Controls after how many epochs to start running mismatch generation
      - mismatch_epochs: Controls every how many epochs mismatch generation must be ran
      - lr: Learning rate
      - lr_scheduler_fn: Controls if the training must use a scheduled learning rate
      - nni_mode: Controls if the HPO functionalities must be activated
    """
    # - Initialise optimiser
    init_fun, update_fun, get_params = adam(
        step_size=lr_scheduler_fn if lr_scheduler_fn else lr
    )
    opt_state = init_fun(model.parameters())
    update_fun = jax.jit(update_fun)

    regenerate_mismatch = None
    if apply_mismatch_after != None:
        # Obtain the prototoype and the random number generator keys
        rng_key = jnp.array([2021, 2022], dtype=jnp.uint32)
        mismatch_prototype = dynamic_mismatch_prototype(model)

        # Get the mismatch generator function (TEST: turn mismatch off for training, only use for finetuning)
        regenerate_mismatch = mismatch_generator(
            mismatch_prototype, percent_deviation=0.30, sigma_rule=3.0
        )
        regenerate_mismatch = jax.jit(regenerate_mismatch)

    # - Training loop
    loss_t = []
    acc_t = []
    best_acc = 0
    te = tqdm(
        range(num_epochs),
        desc="Training",
        unit="Epoch",
        total=num_epochs,
        disable=nni_mode,
    )
    for epoch in te:
        epoch_cum_loss = 0

        for i, (x_train_batch, y_train_batch) in enumerate(train_dl):
            # - Build target signals
            target = build_target_signal(
                y_train_batch.numpy(), x_train_batch.shape[0], x_train_batch.shape[1]
            )

            # - Get parameters
            opt_params = get_params(opt_state)

            # - Regenerate mismatch once in a while
            if (
                apply_mismatch_after != None
                and epoch >= apply_mismatch_after
                and epoch % mismatch_epochs == 0
                and i == 0
            ):
                rng_key, _ = rand.split(rng_key)
                new_params = regenerate_mismatch(model, rng_key=rng_key)
                model = model.set_attributes(new_params)
                logger.info("Applied mismatch")

            # - Compute loss and gradient
            l, g = loss_vgf(opt_params, model, x_train_batch.numpy(), target)
            epoch_cum_loss += l.item()

            # - Update optimiser
            opt_state = update_fun(epoch, g, opt_state)

        loss_t.append(epoch_cum_loss / len(train_dl))

        # Evaluate model on validation set
        acc = evaluate(model, opt_params, val_dl)
        acc_t.append(float(acc))
        if acc > best_acc:
            save_dynapsimnet_model_summary(
                opt_params,
                model,
                train_params,
                acc,
                get_nni_trial_path() if nni_mode else "best_model",
            )
            best_acc = acc

        te.set_postfix(
            {
                "Loss": loss_t[-1] if len(loss_t) > 0 else 0,
                "Acc": format(acc, ".3f"),
                "Best Acc": format(best_acc, ".3f"),
            }
        )

        if nni_mode:
            nni.report_intermediate_result(
                {"default": float(acc), "best_acc": float(best_acc), "loss": loss_t[-1]}
            )

        if nni_mode and epoch % 100 == 0:
            logger.info(
                f"Epoch {epoch} -> Loss:{loss_t[-1]}, Acc:{acc}, Best Acc:{best_acc}"
            )

        # Early termination for trial not improving (due generally to very bad parameters configuration)
        if (
            nni_mode
            and epoch > 0
            and epoch % 100 == 0
            and abs(loss_t[-100] - loss_t[-1]) < 1e-5
        ):
            logger.info(
                f"Trial killed at epoch {epoch} for reason: not improving (e-100:{loss_t[-100]}, e:{loss_t[-1]})"
            )
            break

    if nni_mode:
        nni.report_final_result({"default": float(best_acc)})

    save_training_metadata(
        input_params,
        lr,
        num_epochs,
        lr_scheduler_fn != None,
        nni_mode,
        loss_t,
        acc_t,
        get_nni_trial_path() if nni_mode else "best_model",
    )

    logger.info(f"Training finished with accuracy: {best_acc}")


# @jax.jit
def evaluate_batch(x, y, model, params):
    model = model.set_attributes(params)
    model = model.reset_state()
    output, _, _ = model(x)
    m = jnp.sum(output, axis=1)
    preds = jnp.argmax(m, axis=1)
    return jnp.mean(jnp.array(preds == y))


def evaluate(model, params, dl):
    """
    Evaluates the model using the provided data.
    Input Params:
      - dl: Evaluation DataLoader
      - model: Model to be evaluated
      - params: Model parameters
    """
    ds = dl.dataset
    return evaluate_batch(ds.x.numpy(), ds.y.numpy(), model, params)


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
        "--epochs", type=int, help="The number of epochs to train for.", default=1500
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        help="The path to load a model and resume training.",
        default=None,
    )
    parser.add_argument(
        "--apply_mismatch_after",
        type=int,
        help="Enables mismatch generation after the given amount of epochs.",
        default=None,
    )
    args = parser.parse_args()

    # Logger config
    logger = configure_logger(args.nni_mode)

    train_params = {
        "tau_mem": 0.045,
        "tau_syn": 0.076,
        # "gain_mem": 4,
        # "gain_syn": 100,
    }

    if args.nni_mode:
        optimized_params = nni.get_next_parameter()
        train_params.update(optimized_params)
        logger.info(f"Initial parameters: {train_params}")

    input_params = {
        "file_path": "../data/4bit_spikeset_PHASE_full.npy",
        "n_channels": 12,
        "n_classes": 7,
        "sample_duration": 2,
        "batch_size": 128,
        # Conversion from 40Hz to 1000Hz, to be compatible with a timestep (dt) of 1e-3
        "sample_time_modifier": 0.020,
        "dt": 1e-3,
        "enabled_classes": [13, 15, 16, 17],
        "use_onehot_labels": True,
    }

    train_dl, val_dl, test_dl = load_data(**input_params)

    model = build_model(
        input_params["n_channels"],
        len(input_params["enabled_classes"])
        if input_params["enabled_classes"] != None
        else input_params["n_classes"],
        input_params["dt"],
        neuron_parameters={
            "Itau_mem": get_itau_parameter("mem", train_params["tau_mem"]),
            "Itau_syn": get_itau_parameter("ampa", train_params["tau_syn"]),
            # "Igain_mem": get_igain_parameter("mem", train_params["tau_mem"], train_params["gain_mem"]),
            # "Igain_syn": get_igain_parameter("ampa", train_params["tau_syn"], train_params["gain_syn"]),
        },
        load_model_path=args.load_model_path,
    )

    # lr_scheduler_fn = exponential_decay(train_params["step_size"],
    #                                   train_params["decay_steps"],
    #                                   1e-1)

    train(
        input_params,
        train_dl,
        val_dl,
        model,
        num_epochs=args.epochs,
        apply_mismatch_after=args.apply_mismatch_after,
        lr=1e-3,
        lr_scheduler_fn=None,
        nni_mode=args.nni_mode,
    )
