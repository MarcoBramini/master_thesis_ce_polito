# Utils
import argparse
from tqdm import tqdm

# Rockpool Imports
from rockpool.parameters import Constant

# Torch Imports
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss

# NNI Imports
import nni

# Local Imports
from lib.data_loading import load_data
from lib.logging_utils import (
    configure_logger,
    save_lifnet_model_summary,
    save_training_metadata,
)
from lib.nni_utils import get_nni_trial_path
from lib.xylo_net_adaptation import adapt_network
import lib.xylo_networks as xylo_networks

logger = None

device = None


def build_network(
    n_input_channels, n_population, n_output_channels, dt, neuron_parameters={}
):
    """
    Builds the model to be trained.
    Input Params:
      - n_input_channels: Number of input channels of the model. Must match the data channels
      - n_population: Number of neurons in the hidden layers of the network.
      - n_output_channels: Number of output channels. Must match the amount of classes in the task
      - dt: Duration of a single timestep. Depends on the input data sampling rate
      - neuron_parameters: Neuron specific parameters (i.e. tau_mem, tau_syn)
      - load_model_path: Load a model checkpoint for resuming training
    """
    neuron_parameters = {
        **neuron_parameters,
        "bias": Constant(0.0),
        "threshold": Constant(1.0),
        "dt": dt,
    }

    net = xylo_networks.get_ff_deep_deep_res(
        n_input_channels, n_population, n_output_channels, neuron_parameters
    )

    logger.info(f"Built network: \n{net}")

    return net


def train(
    input_params,
    train_dl,
    val_dl,
    net,
    num_epochs=1500,
    lr=1e-3,
    lr_scheduler_fn=None,
    adapt_network_epochs=10,
    nni_mode=False,
):
    """
    Runs the training of the passed model.
    Input Params:
      - input_params: An object which includes dataset specific properties
      - train_dl: Training DataLoader
      - val_dl: Validation DataLoader
      - net: Model to train
      - num_epochs: Controls the duration of the training in terms of number of epochs
      - lr: Learning rate
      - lr_scheduler_fn: Controls if the training must use a scheduled learning rate
      - adapt_network_epochs: Controls every how many epochs Synapse Pruning must be ran
      - nni_mode: Controls if the HPO functionalities must be activated
    """
    # - Initialise optimiser
    optimizer = Adam(net.parameters().astorch(), lr=lr)
    scheduler = None
    if lr_scheduler_fn is not None:
        scheduler = lr_scheduler_fn(optimizer)

    # - Loss function
    loss_fun = CrossEntropyLoss()

    # - Training loop
    net = net.to(device)
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

        if adapt_network_epochs is not None and epoch % adapt_network_epochs == 0:
            adapt_network(net.cpu())
            net = net.to(device)

        for x_train_batch, y_train_batch in train_dl:
            net.train()

            # - Zero the optimiser gradients
            optimizer.zero_grad()

            # - Evolve the network with the current batch
            output, _, _ = net(x_train_batch.to(device))

            # - Get the prediction -- number of spike in each output channel
            pred = torch.sum(output, dim=1)

            # - Compute the loss value
            loss = loss_fun(pred, y_train_batch.to(device))
            epoch_cum_loss += loss.item()

            # - Compute gradients with backward step and update parameters
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_t.append(epoch_cum_loss / len(train_dl))

        # Evaluate model on validation dataset
        acc = evaluate(net, val_dl)
        acc_t.append(float(acc))
        if acc > best_acc:
            save_lifnet_model_summary(
                net,
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


def evaluate(net, dl):
    """
    Evaluates the model using the provided data.
    Input Params:
      - dl: Evaluation DataLoader
      - net: Model to be evaluated
    """
    net.eval()
    ds = dl.dataset
    with torch.no_grad():
        output, _, _ = net(ds.x.to(device))
        pred = output.sum(dim=1).argmax(dim=1)
        return torch.mean(pred == ds.y.to(device), dtype=float)


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
        "--epochs", type=int, help="The number of epochs to train for.", default=1500
    )
    args = parser.parse_args()

    # Logger config
    logger = configure_logger(args.nni_mode)

    # - Determine which device to use
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"PyTorch is currently running on: {device}")

    train_params = {
        "n_population": 256,
        "tau_mem": 0.081,
        "tau_syn": 0.077,
    }

    if args.nni_mode:
        optimized_params = nni.get_next_parameter()
        train_params.update(optimized_params)
        logger.info(f"Initial parameters: {train_params}")

    input_params = {
        "file_path": "../data/4bit_spikeset_PHASE_full.npy",
        "n_channels": 12,
        "n_classes": 7,
        "dt": 5e-2,
        "sample_duration": 2,
        "batch_size": 1024,
        "sample_time_modifier": 1,
        "enabled_classes": [6, 8],
    }

    train_dl, val_dl, test_dl = load_data(**input_params)

    net = build_network(
        input_params["n_channels"],
        train_params["n_population"],
        len(input_params["enabled_classes"])
        if input_params["enabled_classes"] != None
        else input_params["n_classes"],
        input_params["dt"],
        neuron_parameters={
            "tau_mem": Constant(train_params["tau_mem"]),
            "tau_syn": Constant(train_params["tau_syn"]),
        },
    )

    # def exponential_lr_fn(optimizer): return ExponentialLR(
    #    optimizer,
    #    # 0.996935,  #  Gamma set to have a lr decay after 1500 epochs from 1e-2 to to 1e-4
    #    0.999079,  #  Gamma set to have a lr decay after 5000 epochs from 1e-2 to to 1e-4
    # )

    train(
        input_params,
        train_dl,
        val_dl,
        net,
        num_epochs=args.epochs,
        lr=1e-3,
        lr_scheduler_fn=None,
        # lr_scheduler_fn=exponential_lr_fn,
        nni_mode=args.nni_mode,
    )
