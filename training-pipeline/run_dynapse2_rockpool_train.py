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
from lib.logging_utils import configure_logger, save_dynapsimnet_model_summary, save_training_metadata
from lib.nni_utils import get_nni_trial_path

logger = None


def build_network(n_input_channels, n_output_channels, dt, neuron_parameters):
    neuron_parameters = {
        **neuron_parameters,
        'percent_mismatch': 0,  # 0.05,
        'dt': dt,
        "has_rec": True,
    }

    # - Build network
    net = Sequential(
        LinearJax((n_input_channels, n_output_channels), has_bias=False),
        DynapSim(
            (n_output_channels, n_output_channels),
            **neuron_parameters,
        )
    )

    logger.info(
        f"Built network: \n{net}")

    return net


def build_target_signal(y, n_samples, n_timesteps):
    return np.array([[y[i]]*n_timesteps for i in range(n_samples)])

# - Loss function


@jax.jit
@jax.value_and_grad
def loss_vgf(params, net, input, target):
    net = net.set_attributes(params)
    net = net.reset_state()
    output, _, _ = net(input)
    return jl.mse(output, target)


def train(input_params, train_dl, val_dl, net, num_epochs=20000, apply_mismatch_epochs=None, lr=1e-3, lr_scheduler_fn=None, nni_mode=False):
    # - Initialise optimiser
    init_fun, update_fun, get_params = adam(
        step_size=lr_scheduler_fn if lr_scheduler_fn else lr)
    opt_state = init_fun(net.parameters())
    update_fun = jax.jit(update_fun)

    # Obtain the prototoype and the random number generator keys
    rng_key = jnp.array([2021, 2022], dtype=jnp.uint32)
    mismatch_prototype = dynamic_mismatch_prototype(net)

    # Get the mismatch generator function (TEST: turn mismatch off for training, only use for finetuning)
    regenerate_mismatch = mismatch_generator(
        mismatch_prototype, percent_deviation=0.30, sigma_rule=3.0
    )
    regenerate_mismatch = jax.jit(regenerate_mismatch)

    # - Training loop
    loss_t = []
    acc_t = []
    best_acc = 0
    te = tqdm(range(num_epochs), desc="Training",
              unit="Epoch", total=num_epochs, disable=nni_mode)
    for epoch in te:
        epoch_cum_loss = 0

        for i, (x_train_batch, y_train_batch) in enumerate(train_dl):
            # - Build target signals
            target = build_target_signal(y_train_batch.numpy(
            ), x_train_batch.shape[0], x_train_batch.shape[1])

            # - Get parameters
            opt_params = get_params(opt_state)

            # - Regenerate mismatch once in a while
            if apply_mismatch_epochs != None and epoch % apply_mismatch_epochs == 0 and i == 0:
                rng_key, _ = rand.split(rng_key)
                new_params = regenerate_mismatch(net, rng_key=rng_key)
                net = net.set_attributes(new_params)

            # - Compute loss and gradient
            l, g = loss_vgf(opt_params, net, x_train_batch.numpy(), target)
            epoch_cum_loss += l.item()

            # - Update optimiser
            opt_state = update_fun(epoch, g, opt_state)

        loss_t.append(epoch_cum_loss/len(train_dl))

        # Evaluate model on validation set
        acc = evaluate(net, opt_params, val_dl)
        acc_t.append(float(acc))
        if acc > best_acc:
            save_dynapsimnet_model_summary(opt_params, net, train_params, acc,
                                           get_nni_trial_path() if nni_mode else "best_model")
            best_acc = acc

        te.set_postfix(
            {"Loss": loss_t[-1] if len(loss_t) > 0 else 0,
             "Acc": format(acc, ".3f"),
             "Best Acc": format(best_acc, ".3f")})

        if nni_mode:
            nni.report_intermediate_result(
                {"default": float(acc), "best_acc": float(best_acc), "loss": loss_t[-1]})

        if nni_mode and epoch % 100 == 0:
            logger.info(
                f"Epoch {epoch} -> Loss:{loss_t[-1]}, Acc:{acc}, Best Acc:{best_acc}")

        # Early termination for trial not improving (due generally to very bad parameters configuration)
        if nni_mode and epoch > 0 and epoch % 100 == 0 and abs(loss_t[-100] - loss_t[-1]) < 1e-5:
            logger.info(
                f"Trial killed at epoch {epoch} for reason: not improving (e-100:{loss_t[-100]}, e:{loss_t[-1]})")
            break

    if nni_mode:
        nni.report_final_result(
            {"default": float(best_acc)})

    save_training_metadata(input_params, lr, num_epochs, lr_scheduler_fn != None, nni_mode,
                           loss_t, acc_t, get_nni_trial_path() if nni_mode else "best_model")

    logger.info(f"Training finished with accuracy: {best_acc}")


# @jax.jit
def evaluate_batch(x, y, net, params):
    net = net.set_attributes(params)
    net = net.reset_state()
    output, _, _ = net(x)
    m = jnp.sum(output, axis=1)
    preds = jnp.argmax(m, axis=1)
    return jnp.mean(jnp.array(preds == y))


def evaluate(net, params, dl):
    ds = dl.dataset
    return evaluate_batch(ds.x.numpy(), ds.y.numpy(), net, params)


if __name__ == "__main__":
    # This is a workaround needed because of the old Cuda version running on the machine
    os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

    # CLI config
    parser = argparse.ArgumentParser()
    parser.add_argument("--nni_mode", type=bool,
                        help="Mandatory if the script is executed with NNI_mode. Enables NNI features as logging, status reporting.", default=False)
    parser.add_argument("--epochs", type=int,
                        help="The number of epochs to train for.", default=1500)
    args = parser.parse_args()

    # Logger config
    logger = configure_logger(args.nni_mode)

    train_params = {
        "tau_mem":  0.160,
        "tau_syn": 0.065,
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
        "enabled_classes": [6,8,13,1],
        "use_onehot_labels": True
    }

    train_dl, val_dl, test_dl = load_data(**input_params)

    net = build_network(input_params["n_channels"],
                        len(input_params["enabled_classes"]
                            ) if input_params["enabled_classes"] != None else input_params["n_classes"],
                        input_params["dt"],
                        neuron_parameters={
                            "Itau_mem": get_itau_parameter("mem", train_params["tau_mem"]),
                            "Itau_syn": get_itau_parameter("ampa", train_params["tau_syn"]),
                            # "Igain_mem": get_igain_parameter("mem", train_params["tau_mem"], train_params["gain_mem"]),
                            # "Igain_syn": get_igain_parameter("ampa", train_params["tau_syn"], train_params["gain_syn"]),
    })

    # lr_scheduler_fn = exponential_decay(train_params["step_size"],
    #                                   train_params["decay_steps"],
    #                                   1e-1)

    train(input_params,
          train_dl,
          val_dl,
          net,
          num_epochs=args.epochs,
          lr=1e-3,
          lr_scheduler_fn=None,
          nni_mode=args.nni_mode)
