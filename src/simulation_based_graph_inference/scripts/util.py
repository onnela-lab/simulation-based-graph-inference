import argparse
import numpy as np
import torch as th
from ..config import GENERATOR_CONFIGURATIONS


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
