import argparse
import numpy as np
import torch as th
import typing
from ..config import GENERATOR_CONFIGURATIONS
from .. import models


def _apply_seed(seed):
    if seed is not None:
        seed = int(seed)
        th.manual_seed(seed)
        np.random.seed(seed)
    return seed


def get_parser() -> argparse.ArgumentParser:
    """
    Create a basic parser to parameterize scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", "-s", help="random number generator seed", type=_apply_seed
    )
    parser.add_argument(
        "--configuration",
        help="configuration to use for sampling graphs",
        required=True,
        choices=GENERATOR_CONFIGURATIONS,
    )
    return parser


def dense_from_str(
    layer: str,
    activation: typing.Callable,
    final_activation: bool,
    use_layer_norm: bool = False,
) -> th.nn.Module:
    """
    Create a dense neural network from a comma-separated string of layer sizes.

    Args:
        layer: Comma-separated layer sizes, e.g. "32,32" for two 32-unit layers.
        activation: Activation function to use between layers.
        final_activation: Whether to apply activation after the final layer.
        use_layer_norm: Whether to add LayerNorm before each activation.

    Returns:
        A sequential dense neural network.
    """
    return models.create_dense_nn(
        map(int, layer.split(",")), activation, final_activation, use_layer_norm
    )
