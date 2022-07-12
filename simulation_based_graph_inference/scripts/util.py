import argparse
import torch as th
from .. import generators


GENERATORS = [
    "generate_duplication_mutation_complementation",
    "generate_duplication_mutation_random",
    "generate_poisson_random_attachment",
    "generate_redirection",
]


def _apply_seed(seed):
    if seed is not None:
        seed = int(seed)
        th.manual_seed(seed)
        generators.set_seed(seed)
    return seed


def get_parser(default_num_nodes: int) -> argparse.ArgumentParser:
    """
    Create a basic parser to parameterize scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", "-n", help="number of nodes", default=default_num_nodes,
                        type=int)
    parser.add_argument("--seed", "-s", help="random number generator seed", type=_apply_seed)
    parser.add_argument("generator", help="generator to use for sampling graphs",
                        choices=GENERATORS)
    return parser
