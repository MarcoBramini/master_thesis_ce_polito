import json
import logging
import numpy as np
from rockpool.nn.modules.torch import LinearTorch, LIFTorch
from rockpool.nn.combinators import Sequential, Residual
from scipy.stats import rankdata
import torch

logger = logging.getLogger("main")

# --------------------
# Network Construction
# --------------------


# Builds a network starting from a saved state
def net_from_params(params, neuron_parameters={}):
    """
    Restores a model from its parameters, stored within its checkpoint files.
    Only for Xylo.
    Input Params:
      - params: Parameters object associated to the model
      - neuron_parameters: Neuron hyperparameters to use for the model
    """
    layers = build_layers(params, neuron_parameters)
    return Sequential(*layers)


def weight_list_from_str(weights_str):
    return json.loads(weights_str)


def get_linear_layer_shape(weights):
    return (len(weights), len(weights[0]))


def get_lif_layer_shape(last_linear_shape, rec_weights=None):
    if rec_weights != None:
        return (last_linear_shape[1], len(rec_weights))
    return last_linear_shape[1]


def build_layers(params, neuron_parameters={}):
    layers = ()
    last_shape = None
    for layer_name in sorted(params.keys()):
        layer_params = params[layer_name]
        layer_position, layer_type = layer_name.split("_")
        match layer_type:
            case "LinearTorch":
                last_shape = get_linear_layer_shape(
                    weight_list_from_str(layer_params["weight"])
                )
                layers += (LinearTorch(last_shape),)
            case "LIFTorch":
                if last_shape == None:
                    raise f"Malformed model params: LIF layer {layer_position} shape couldn't be inferred from previous linear layer"
                has_rec = True if "w_rec" in layer_params.keys() else False
                shape = get_lif_layer_shape(
                    last_shape,
                    weight_list_from_str(layer_params["w_rec"]) if has_rec else None,
                )
                layers += (LIFTorch(shape, has_rec=has_rec, **neuron_parameters),)
            case "TorchResidual":
                sublayers = build_layers(layer_params, neuron_parameters)
                last_shape = None
                layers += (Residual(*sublayers),)
    return layers


# ---------------
# Synapse Pruning
# ---------------


def adapt_network(net, max_synapses=63):
    """
    # Adapts a network that contains neurons with an exceeding number of output synapses.
    # The n synapses with lightest weight are dropped to match the neuron fan out constraint (0:63).
    # Only for Xylo
    Input Params:
      - net: Network to adapt
      - max_synapses: Maximum number of output connections for each neuron
    """
    return adapt_network_rec(net, max_synapses)


def drop_exceeding_synapses(weights, max_synapses=63):
    dropped_neurons_count = 0
    dropped_synapses_count = 0
    for neuron_id, neuron_weights in list(enumerate(weights)):
        synapses_count = np.sum(neuron_weights != 0)
        if synapses_count > max_synapses:
            count_to_remove = synapses_count - max_synapses
            mask = rankdata(
                abs(neuron_weights), method="ordinal"
            ) - 1 < count_to_remove + np.sum(neuron_weights == 0)
            weights[neuron_id] = np.where(
                mask, np.zeros(len(neuron_weights)), neuron_weights
            ).astype(float)
            dropped_neurons_count += 1
            dropped_synapses_count += count_to_remove
    return dropped_neurons_count, dropped_synapses_count


def drop_exceeding_layer_output_synapses(
    lin_layer, prev_lif_layer=None, max_synapses=63
):
    w = lin_layer.parameters()["weight"].clone().detach().numpy()
    lin_layer_shape = get_linear_layer_shape(w)

    w_stack = w
    prev_had_rec = None
    if prev_lif_layer != None and (
        prev_had_rec := "w_rec" in prev_lif_layer.parameters().keys()
    ):
        w_rec = prev_lif_layer.parameters()["w_rec"].clone().detach().numpy()
        w_stack = np.hstack([w, w_rec])

    neurons_count, synapses_count = drop_exceeding_synapses(w_stack, max_synapses)
    if neurons_count > 0:
        logger.info(
            f"Dropped {int(synapses_count/neurons_count)} output synapses each for {neurons_count} neurons in layer {lin_layer} (Rec:{prev_lif_layer if prev_had_rec else None})"
        )

    w = w_stack
    if prev_lif_layer != None and prev_had_rec:
        w, w_rec = (
            w_stack[..., : lin_layer_shape[1]],
            w_stack[..., lin_layer_shape[1] :],
        )
        prev_lif_layer.parameters()["w_rec"].data = torch.nn.parameter.Parameter(
            torch.tensor(w_rec, requires_grad=True)
        )

    lin_layer.parameters()["weight"].data = torch.nn.parameter.Parameter(
        torch.tensor(w, requires_grad=True)
    )


def adapt_network_rec(layer, max_synapses=63):
    prev_sl = None
    for sl in layer:
        layer_type = sl.name.split("_")[1].removesuffix("'")
        match layer_type:
            case "LinearTorch":
                drop_exceeding_layer_output_synapses(sl, prev_sl, max_synapses)
            case "LIFTorch":
                prev_sl = sl
            case "TorchResidual":
                adapt_network_rec(sl, max_synapses)
