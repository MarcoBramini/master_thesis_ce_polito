from rockpool.nn.modules.torch import LinearTorch, LIFTorch
from rockpool.nn.combinators import Sequential, Residual


def get_rec_deep(n_input_channels, n_population, n_output_channels, neuron_parameters):
    return Sequential(
        LinearTorch((n_input_channels, n_population)),
        LIFTorch(n_population, has_rec=True, **neuron_parameters),
        LinearTorch((n_population, n_population)),
        LIFTorch(n_population, **neuron_parameters),
        LinearTorch((n_population, n_output_channels)),
        LIFTorch(n_output_channels, **neuron_parameters),
    )


def get_ff_simple(n_input_channels, n_population, n_output_channels, neuron_parameters):
    return Sequential(
        LinearTorch((n_input_channels, n_population)),
        LIFTorch(n_population, **neuron_parameters),
        LinearTorch((n_population, n_output_channels)),
        LIFTorch(n_output_channels, **neuron_parameters),
    )


def get_ff_deep(n_input_channels, n_population, n_output_channels, neuron_parameters):
    return Sequential(
        LinearTorch((n_input_channels, n_population)),
        LIFTorch(n_population, **neuron_parameters),
        LinearTorch((n_population, n_population)),
        LIFTorch(n_population, **neuron_parameters),
        LinearTorch((n_population, n_output_channels)),
        LIFTorch(n_output_channels, **neuron_parameters),
    )


def get_ff_deep_deep(n_input_channels, n_population, n_output_channels, neuron_parameters):
    return Sequential(
        LinearTorch((n_input_channels, n_population)),
        LIFTorch(n_population, **neuron_parameters),
        LinearTorch((n_population, n_population)),
        LIFTorch(n_population, **neuron_parameters),
        LinearTorch((n_population, n_population)),
        LIFTorch(n_population, **neuron_parameters),
        LinearTorch((n_population, n_output_channels)),
        LIFTorch(n_output_channels, **neuron_parameters),
    )


def get_ff_deep_deep_deep(n_input_channels, n_population, n_output_channels, neuron_parameters):
    return Sequential(
        LinearTorch((n_input_channels, n_population)),
        LIFTorch(n_population, **neuron_parameters),
        LinearTorch((n_population, n_population)),
        LIFTorch(n_population, **neuron_parameters),
        LinearTorch((n_population, n_population)),
        LIFTorch(n_population, **neuron_parameters),
        LinearTorch((n_population, n_population)),
        LIFTorch(n_population, **neuron_parameters),
        LinearTorch((n_population, n_output_channels)),
        LIFTorch(n_output_channels, **neuron_parameters),
    )


def get_ff_deep_res(n_input_channels, n_population, n_output_channels, neuron_parameters):
    return Sequential(
        LinearTorch((n_input_channels, n_population)),
        LIFTorch(n_population, **neuron_parameters),
        Residual(
            LinearTorch((n_population, n_population)),
            LIFTorch(n_population, **neuron_parameters)
        ),
        LinearTorch((n_population, n_output_channels)),
        LIFTorch(n_output_channels, **neuron_parameters),
    )


def get_ff_deep_deep_res(n_input_channels, n_population, n_output_channels, neuron_parameters):
    return Sequential(
        LinearTorch((n_input_channels, n_population)),
        LIFTorch(n_population, **neuron_parameters),
        Residual(
            LinearTorch((n_population, n_population)),
            LIFTorch(n_population, **neuron_parameters)
        ),
        Residual(
            LinearTorch((n_population, n_population)),
            LIFTorch(n_population, **neuron_parameters)
        ),
        LinearTorch((n_population, n_output_channels)),
        LIFTorch(n_output_channels, **neuron_parameters),
    )
