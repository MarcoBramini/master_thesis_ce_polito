from rockpool.devices.dynapse.lookup import (
    default_layout,
)


def get_itau_parameter(name, tau_val):
    return ((default_layout["Ut"] / ((default_layout["kappa_p"] + default_layout["kappa_n"]) / 2)) * default_layout[f"C_{name}"])/tau_val


def get_igain_parameter(name, tau_val, val):
    return get_itau_parameter(name, tau_val) * val
