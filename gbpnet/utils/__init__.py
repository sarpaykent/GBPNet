from __future__ import print_function, absolute_import, division

import argparse
import torch


# https://github.com/sunglasses-ai/classy/blob/3e74cba1fdf1b9f9f2ba1cfcfa6c2017aa59fc04/classy/optim/factories.py#L14
def get_activations(optional=False):
    activations = {
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "sigmoid": torch.nn.Sigmoid,
        "silu": torch.nn.SiLU,
        "selu": torch.nn.SELU,
    }
    if optional:
        activations[""] = None

    return activations


def get_activations_none(optional=False):
    activations = {
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "sigmoid": torch.nn.Sigmoid,
        "silu": torch.nn.SiLU,
        "selu": torch.nn.SELU,
    }
    if optional:
        activations[""] = None
        activations[None] = None

    return activations


def dictionary_to_option(options, selected):
    if selected not in options:
        raise argparse.ArgumentTypeError(
            f'Invalid choice "{selected}", choose one from {", ".join(list(options.keys()))} '
        )
    return options[selected]()


def str2act(input_str):
    if input_str == "":
        return None

    act = get_activations(optional=True)
    out = dictionary_to_option(act, input_str)
    return out


def str2bool(v):
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
