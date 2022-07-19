"""
This module holds configurations for different graph generators and associated conditional density
estimators, including fixed default values, priors, and architectures. There is not necessarily a
one-to-one mapping between configurations and generators. E.g., there may be multiple configurations
for a given generator.
"""
import enum
import networkx as nx
import numpy as np
from scipy import special
import torch as th
import typing
from .models import DistributionModule
from . import generators


Configuration = enum.Enum(
    "Configuration", "duplication_complementation_graph duplication_mutation_graph "
    "poisson_random_attachment_graph redirection_graph random_geometric_graph waxman_graph "
    "web_graph planted_partition_graph watts_strogatz_graph newman_watts_strogatz_graph "
    "latent_space_graph",
)


def _planted_partition_graph(num_nodes: int, num_groups: int, **kwargs) -> nx.Graph:
    """
    Wrapper for the planted partition graph ensuring that the first argument is the number of nodes.
    """
    return nx.planted_partition_graph(num_groups, num_nodes // num_groups, **kwargs)


def _latent_space_graph(num_nodes: int, alpha: float, beta: float, **kwargs) -> nx.Graph:
    """
    Wrapper for a soft random geometric graph to generate Hoff's latent space model.
    """
    # We use an enormous radius to consider all nodes.
    return nx.soft_random_geometric_graph(num_nodes, radius=1e9,
                                          p_dist=lambda dist: beta * special.expit(-alpha * dist))


# Mapping from configuration name to generator function and constant arguments, i.e., not dependent
# on the prior.
GENERATOR_CONFIGURATIONS = {
    Configuration.duplication_complementation_graph:
        (generators.duplication_complementation_graph, {}),
    Configuration.duplication_mutation_graph:
        (generators.duplication_mutation_graph, {}),
    Configuration.poisson_random_attachment_graph: (generators.poisson_random_attachment_graph, {}),
    Configuration.redirection_graph: (generators.redirection_graph, {"max_num_connections": 2}),
    Configuration.random_geometric_graph: (nx.random_geometric_graph, {"dim": 2}),
    Configuration.waxman_graph: (nx.waxman_graph, {}),
    Configuration.web_graph: (generators.web_graph, {"dist_degree_new": np.arange(3) == 2}),
    Configuration.planted_partition_graph: (_planted_partition_graph, {"num_groups": 2}),
    Configuration.watts_strogatz_graph: (nx.watts_strogatz_graph, {"k": 4}),
    Configuration.newman_watts_strogatz_graph: (nx.newman_watts_strogatz_graph, {"k": 4}),
    Configuration.latent_space_graph: (_latent_space_graph, {"dim": 2}),
}


def get_prior(configuration: Configuration) -> typing.Mapping[str, th.distributions.Distribution]:
    """
    Get a prior for the given generator with sensible defaults.

    Args:
        configuration: Generator configuration.

    Returns:
        prior: Mapping from parameter names to distributions.
    """
    if configuration == Configuration.duplication_complementation_graph:
        return {
            "interaction_proba": th.distributions.Uniform(0, 1),
            "divergence_proba": th.distributions.Uniform(0, 1),
        }
    elif configuration == Configuration.duplication_mutation_graph:
        return {
            "mutation_proba": th.distributions.Uniform(0, 1),
            "deletion_proba": th.distributions.Uniform(0, 1),
        }
    elif configuration == Configuration.poisson_random_attachment_graph:
        return {
            "rate": th.distributions.Gamma(2, 1),
        }
    elif configuration == Configuration.redirection_graph:
        return {
            "redirection_proba": th.distributions.Uniform(0, 1),
        }
    elif configuration == Configuration.random_geometric_graph:
        return {
            "radius": th.distributions.Uniform(0, 1),
        }
    elif configuration == Configuration.waxman_graph:
        # Connection probability is beta * exp(- alpha * distance / L)
        return {
            "beta": th.distributions.Uniform(0, 1),
            "alpha": th.distributions.Uniform(0, 1),
        }
    elif configuration == Configuration.web_graph:
        return {
            "proba_new": th.distributions.Uniform(0, 1),
            "proba_uniform_new": th.distributions.Uniform(0, 1),
            "proba_uniform_old1": th.distributions.Uniform(0, 1),
        }
    elif configuration == Configuration.planted_partition_graph:
        return {
            "p_in": th.distributions.Uniform(0, 1),
            "p_out": th.distributions.Uniform(0, 1),
        }
    elif configuration == Configuration.watts_strogatz_graph:
        return {
            "p": th.distributions.Uniform(0, 1),
        }
    elif configuration == Configuration.newman_watts_strogatz_graph:
        return {
            "p": th.distributions.Uniform(0, 1),
        }
    elif configuration == Configuration.latent_space_graph:
        # Connection probability is beta * expit(- alpha * distance)
        return {
            "beta": th.distributions.Uniform(0, 1),
            "alpha": th.distributions.Uniform(0, 1),
        }
    else:  # pragma: no cover
        raise ValueError(f"{configuration} is not a valid configuration")


def get_parameterized_posterior_density_estimator(configuration: Configuration) \
        -> typing.Mapping[str, typing.Tuple[int, typing.Callable]]:
    """
    Get factorized posterior density estimators.

    Args:
        generator: Graph generator for which to get a prior.

    Returns:
        estimator: Mapping from parameter names to a module that returns a
            :class:`torch.distributions.Distribution`.
    """
    if configuration == Configuration.duplication_complementation_graph:
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
    elif configuration == Configuration.duplication_mutation_graph:
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
    elif configuration == Configuration.poisson_random_attachment_graph:
        return {
            "rate": DistributionModule(
                th.distributions.Gamma, concentration=th.nn.LazyLinear(1),
                rate=th.nn.LazyLinear(1),
            )
        }
    elif configuration == Configuration.redirection_graph:
        return {
            "redirection_proba": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
        }
    elif configuration == Configuration.random_geometric_graph:
        return {
            "radius": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
        }
    elif configuration == Configuration.waxman_graph:
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
    elif configuration == Configuration.web_graph:
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
    elif configuration == Configuration.planted_partition_graph:
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
    elif configuration == Configuration.watts_strogatz_graph:
        return {
            "p": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
        }
    elif configuration == Configuration.newman_watts_strogatz_graph:
        return {
            "p": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
        }
    elif configuration == Configuration.latent_space_graph:
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
    else:  # pragma: no cover
        raise ValueError(f"{configuration} is not a known generator")
