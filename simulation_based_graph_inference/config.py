"""
This module holds configurations for different graph generators and associated conditional density
estimators, including fixed default values, priors, and architectures. There is not necessarily a
one-to-one mapping between configurations and generators. E.g., there may be multiple configurations
for a given generator.
"""
import networkx as nx
import numpy as np
import torch as th
import typing
from .models import DistributionModule
from . import generators


def _planted_partition_graph(num_nodes, num_groups, **kwargs):
    """
    Wrapper for the planted partition graph ensuring that the first argument is the number of nodes.
    """
    return nx.planted_partition_graph(num_groups, num_nodes // num_groups, **kwargs)


# Mapping from configuration name to generator function and constant arguments, i.e., not dependent
# on the prior.
GENERATOR_CONFIGURATIONS = {
    "duplication_complementation": (generators.duplication_complementation, {}),
    "duplication_mutation": (generators.duplication_mutation, {}),
    "poisson_random_attachment": (generators.poisson_random_attachment, {}),
    "redirection": (generators.redirection, {"max_num_connections": 2}),
    "random_geometric_graph": (nx.random_geometric_graph, {"dim": 2}),
    "waxman_graph": (nx.waxman_graph, {}),
    "web": (generators.web, {"dist_degree_new": np.arange(3) == 2}),
    "planted_partition_graph": (_planted_partition_graph, {"num_groups": 2}),
}


def get_prior(configuration_name: str) -> typing.Mapping[str, th.distributions.Distribution]:
    """
    Get a prior for the given generator with sensible defaults.

    Args:
        configuration_name: Name of the generator configuration.

    Returns:
        prior: Mapping from parameter names to distributions.
    """
    if configuration_name == "duplication_complementation":
        return {
            "interaction_proba": th.distributions.Uniform(0, 1),
            "divergence_proba": th.distributions.Uniform(0, 1),
        }
    elif configuration_name == "duplication_mutation":
        return {
            "mutation_proba": th.distributions.Uniform(0, 1),
            "deletion_proba": th.distributions.Uniform(0, 1),
        }
    elif configuration_name == "poisson_random_attachment":
        return {
            "rate": th.distributions.Gamma(2, 1),
        }
    elif configuration_name == "redirection":
        return {
            "redirection_proba": th.distributions.Uniform(0, 1),
        }
    elif configuration_name == "random_geometric_graph":
        return {
            "radius": th.distributions.Uniform(0, 1),
        }
    elif configuration_name == "waxman_graph":
        # Connection probability is beta * exp(- alpha * distance / L)
        return {
            "beta": th.distributions.Uniform(0, 1),
            "alpha": th.distributions.Uniform(0, 1),
        }
    elif configuration_name == "web":
        return {
            "proba_new": th.distributions.Uniform(0, 1),
            "proba_uniform_new": th.distributions.Uniform(0, 1),
            "proba_uniform_old1": th.distributions.Uniform(0, 1),
        }
    elif configuration_name == "planted_partition_graph":
        return {
            "p_in": th.distributions.Uniform(0, 1),
            "p_out": th.distributions.Uniform(0, 1),
        }
    else:  # pragma: no cover
        raise ValueError(f"{configuration_name} is not a valid configuration")


def get_parameterized_posterior_density_estimator(configuration_name: str) \
        -> typing.Mapping[str, typing.Tuple[int, typing.Callable]]:
    """
    Get factorized posterior density estimators.

    Args:
        generator: Graph generator for which to get a prior.

    Returns:
        estimator: Mapping from parameter names to a module that returns a
            :class:`torch.distributions.Distribution`.
    """
    if configuration_name == "duplication_complementation":
        return {
            "interaction_proba": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
            "divergence_proba": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
        }
    elif configuration_name == "duplication_mutation":
        return {
            "mutation_proba": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
            "deletion_proba": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
        }
    elif configuration_name == "poisson_random_attachment":
        return {
            "rate": DistributionModule(
                th.distributions.Gamma, concentration=th.nn.LazyLinear(1),
                rate=th.nn.LazyLinear(1),
            )
        }
    elif configuration_name == "redirection":
        return {
            "redirection_proba": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
        }
    elif configuration_name == "random_geometric_graph":
        return {
            "radius": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
        }
    elif configuration_name == "waxman_graph":
        return {
            "alpha": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
            "beta": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
        }
    elif configuration_name == "web":
        return {
            "proba_new": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
            "proba_uniform_new": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
            "proba_uniform_old1": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
        }
    elif configuration_name == "planted_partition_graph":
        return {
            "p_in": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
            "p_out": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
        }
    else:  # pragma: no cover
        raise ValueError(f"{configuration_name} is not a known generator")
