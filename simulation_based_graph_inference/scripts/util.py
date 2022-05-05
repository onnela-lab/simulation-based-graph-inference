import argparse
import torch as th
import typing
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
    parser.add_argument("generator", help="generator to use for sampling graphs",
                        choices=GENERATORS)
    parser.add_argument("--num_nodes", "-n", help="number of nodes", default=default_num_nodes,
                        type=int)
    parser.add_argument("--seed", "-s", help="random number generator seed", type=_apply_seed)
    return parser


def get_prior(generator: typing.Callable) -> typing.Mapping[str, th.distributions.Distribution]:
    """
    Get a prior for the given generator with sensible defaults.

    Args:
        generator: Graph generator for which to get a prior.

    Returns:
        prior: Mapping from parameter names to distributions.
    """
    if generator is generators.generate_duplication_mutation_complementation:
        return {
            "interaction_proba": th.distributions.Uniform(0, 1),
            "divergence_proba": th.distributions.Uniform(0, 1),
        }
    elif generator is generators.generate_duplication_mutation_random:
        return {
            "mutation_proba": th.distributions.Uniform(0, 1),
            "deletion_proba": th.distributions.Uniform(0, 1),
        }
    elif generator is generators.generate_poisson_random_attachment:
        return {
            "rate": th.distributions.Gamma(2, 1),
        }
    elif generator is generators.generate_redirection:
        return {
            "max_num_connections": th.distributions.Binomial(2, 1),
            "redirection_proba": th.distributions.Uniform(0, 1),
        }
    else:
        raise ValueError(f"{generator.__name__} is not a known generator")  # pragma: no cover
