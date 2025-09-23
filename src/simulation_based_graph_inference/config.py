"""
This module holds configurations for different graph generators and associated conditional density
estimators, including fixed default values, priors, and architectures. There is not necessarily a
one-to-one mapping between configurations and generators. E.g., there may be multiple configurations
for a given generator.
"""

import networkx as nx
import torch as th
from torch.distributions import constraints
from typing import Any, Callable, Mapping, Optional
from .models import DistributionModule
from . import generators


class Configuration:
    """
    Configuration for generative network models.

    Args:
        priors: Mapping of parameter names to prior distributions.
        generator: Generator function that takes the number of nodes as the first argument and
            parameters as keyword arguments. `sample_graph` must be overriden if not given.
        generator_kwargs: Fixed keyword arguments passed to `generator`.
        localization: Predicted localization of the growth rule.
        parameter_constraints: Mapping of parameter names to their support. Inferred from `priors`
            if not given.
    """

    def __init__(
        self,
        priors: Mapping[str, th.distributions.Distribution],
        generator: Optional[Callable] = None,
        generator_kwargs: Mapping[dict, Any] = None,
        localization: Optional[int] = None,
        parameter_constraints: Optional[Mapping[str, constraints.Constraint]] = None,
    ) -> None:
        self.priors = priors
        self.generator = generator
        self.generator_kwargs = generator_kwargs or {}
        self.localization = localization
        self.parameter_constraints = parameter_constraints or {
            name: dist.support for name, dist in self.priors.items()
        }

    def sample_params(self, size: Optional[th.Size] = None):
        size = size or th.Size()
        return {name: dist.sample(size) for name, dist in self.priors.items()}

    def sample_graph(self, num_nodes, **kwargs):
        if self.generator:
            return self.generator(num_nodes, **kwargs, **self.generator_kwargs)
        raise NotImplementedError

    def create_estimator(self):
        estimator = {}
        for name, constraint in self.parameter_constraints.items():
            if constraint is constraints.unit_interval:
                estimator[name] = DistributionModule(
                    th.distributions.Beta,
                    concentration0=th.nn.LazyLinear(1),
                    concentration1=th.nn.LazyLinear(1),
                )
            elif constraint in {constraints.nonnegative, constraints.positive}:
                estimator[name] = DistributionModule(
                    th.distributions.Gamma,
                    concentration=th.nn.LazyLinear(1),
                    rate=th.nn.LazyLinear(1),
                )
            elif constraint is constraints.real:  # pragma: no cover
                estimator[name] = DistributionModule(
                    th.distributions.Normal,
                    loc=th.nn.LazyLinear(1),
                    scale=th.nn.LazyLinear(1),
                )
            else:
                raise NotImplementedError(f"{constraint} constraint is not supported")
        return estimator


def _planted_partition_graph(
    num_nodes: int, num_groups: int, **kwargs
) -> nx.Graph:  # pragma: no cover
    """
    Wrapper for the planted partition graph ensuring that the first argument is the number of nodes.
    """
    return nx.planted_partition_graph(num_groups, num_nodes // num_groups, **kwargs)


def _poisson_random_attachment_graph(num_nodes: int, rate: float, **kwargs):
    return generators.random_attachment_graph(
        num_nodes, th.distributions.Poisson(rate).sample, **kwargs
    )


def _gn_graph(num_nodes: int, gamma: float, **kwargs) -> nx.Graph:
    return nx.gn_graph(num_nodes, lambda k: k**gamma, **kwargs)


# Mapping from configuration name to generator function and constant arguments, i.e., not dependent
# on the prior. TODO: reinstate some of the missing generators as `Configuration` objects.
GENERATOR_CONFIGURATIONS = {
    "poisson_random_attachment_graph": Configuration(
        {"rate": th.distributions.Gamma(2, 1)},
        _poisson_random_attachment_graph,
        localization=0,
    ),
    "random_connection_graph": Configuration(
        {"proba": th.distributions.Beta(1, 1)},
        generators.random_connection_graph,
        localization=0,
    ),
    "newman_watts_strogatz_graph": Configuration(
        {"p": th.distributions.Beta(1, 1)},
        nx.newman_watts_strogatz_graph,
        {"k": 5},
        localization=0,
    ),
    "redirection_graph": Configuration(
        {"redirection_proba": th.distributions.Beta(1, 1)},
        generators.redirection_graph,
        {"max_num_connections": 1},
        localization=1,
    ),
    "copy_graph": Configuration(
        {"copy_proba": th.distributions.Beta(1, 2)},
        generators.copy_graph,
        localization=1,
    ),
    "duplication_mutation_graph": Configuration(
        {
            "mutation_proba": th.distributions.Beta(1, 2),
            "divergence_proba": th.distributions.Beta(2, 1),
        },
        generators.duplication_mutation_graph,
        localization=1,
    ),
    "duplication_complementation_graph": Configuration(
        {
            "interaction_proba": th.distributions.Beta(1, 2),
            "divergence_proba": th.distributions.Beta(2, 1),
        },
        generators.duplication_complementation_graph,
    ),
    # random_geometric_graph: (nx.random_geometric_graph, {"dim": 2}),
    # waxman_graph: (nx.waxman_graph, {}),
    # web_graph: (generators.web_graph, {"dist_degree_new": np.arange(3) == 2}),
    # planted_partition_graph: (_planted_partition_graph, {"num_groups": 2}),
    "watts_strogatz_graph": Configuration(
        {"p": th.distributions.Beta(1, 1)}, nx.watts_strogatz_graph, {"k": 5}
    ),
    # latent_space_graph: (_latent_space_graph, {"dim": 2}),
    # Start with two connected nodes as described in 10.1155/2008/190836 for numerical experiments.
    # partial_duplication_graph: (nx.partial_duplication_graph, {"n": 2}),
    # thresholded_random_geometric_graph: (nx.thresholded_random_geometric_graph, {}),
    # geographical_threshold_graph: (nx.geographical_threshold_graph, {}),
    # degree_attachment_graph: (generators.degree_attachment_graph, {"m": 4}),
    # rank_attachment_graph: (generators.rank_attachment_graph, {"m": 4}),
    "jackson_rogers_graph": Configuration(
        {
            "pr": th.distributions.Beta(1, 1),
            "pn": th.distributions.Beta(1, 1),
        },
        generators.jackson_rogers_graph,
        {"mr": 4},
    ),
    "surfer_graph": Configuration(
        {"hop_proba": th.distributions.Beta(1, 1)},
        generators.surfer_graph,
    ),
    "gn_graph": Configuration(
        {"gamma": th.distributions.Beta(1, 1)},
        _gn_graph,
    ),
    "latent_space_graph": Configuration(
        {"bias": th.distributions.Normal(0, 1), "scale": th.distributions.Gamma(2, 2)},
        generators.latent_space_graph,
        {"num_dims": 2},
    ),
}
