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
        generator_kwargs: Optional[Mapping[str, Any]] = None,
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

    def create_estimator(self) -> th.nn.ModuleDict:
        estimator = {}
        for name, constraint in self.parameter_constraints.items():
            # "Safe" softplus with small addition to avoid numerical issues.
            softplus = th.distributions.transforms.ComposeTransform(
                [
                    th.distributions.transforms.SoftplusTransform(),
                    th.distributions.transforms.AffineTransform(1e-3, 1),
                ]
            )
            if constraint is constraints.unit_interval:
                estimator[name] = DistributionModule(
                    th.distributions.Beta,
                    concentration0=th.nn.LazyLinear(1),
                    concentration1=th.nn.LazyLinear(1),
                    constraint_transforms={
                        "concentration0": softplus,
                        "concentration1": softplus,
                    },
                )
            elif constraint in {constraints.nonnegative, constraints.positive}:
                estimator[name] = DistributionModule(
                    th.distributions.Gamma,
                    concentration=th.nn.LazyLinear(1),
                    rate=th.nn.LazyLinear(1),
                    constraint_transforms={"concentration": softplus, "rate": softplus},
                )
            elif constraint is constraints.real:  # pragma: no cover
                estimator[name] = DistributionModule(
                    th.distributions.Normal,
                    loc=th.nn.LazyLinear(1),
                    scale=th.nn.LazyLinear(1),
                    constraint_transforms={"scale": softplus},
                )
            elif isinstance(constraint, constraints.interval):
                estimator[name] = DistributionModule(
                    th.distributions.Beta,
                    concentration0=th.nn.LazyLinear(1),
                    concentration1=th.nn.LazyLinear(1),
                    constraint_transforms={
                        "concentration0": softplus,
                        "concentration1": softplus,
                    },
                    transforms=[
                        th.distributions.AffineTransform(
                            loc=constraint.lower_bound,
                            scale=constraint.upper_bound - constraint.lower_bound,
                        )
                    ],
                )
            else:
                raise NotImplementedError(f"{constraint} constraint is not supported")
        return th.nn.ModuleDict(estimator)


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
    gamma = float(gamma)
    graph: nx.Graph = nx.gn_graph(num_nodes, lambda k: k**gamma, **kwargs)
    graph = graph.to_undirected()
    assert not nx.number_of_selfloops(graph)
    return graph


# Mapping from configuration name to generator function and constant arguments, i.e., not dependent
# on the prior. TODO: reinstate some of the missing generators as `Configuration` objects.
GENERATOR_CONFIGURATIONS = {
    # Random attachment with Poisson-distributed degree. Gamma(2,1) gives mean=2 attachments.
    "poisson_random_attachment_graph": Configuration(
        {"rate": th.distributions.Gamma(2, 1)},
        _poisson_random_attachment_graph,
        localization=0,
    ),
    # Erdos-Renyi random graph. Uniform prior explores full density range.
    # Reference: Erdos & Renyi (1959) https://doi.org/10.5486/PMD.1959.6.3-4.12
    "random_connection_graph": Configuration(
        {"proba": th.distributions.Beta(1, 1)},
        generators.random_connection_graph,
        localization=0,
    ),
    # Small-world properties emerge at p ~ 0.01-0.1. Beta(1,9) concentrates mass there (mean=0.1).
    # Reference: Newman & Watts (1999) https://doi.org/10.1016/S0375-9601(99)00757-4
    "newman_watts_strogatz_graph": Configuration(
        {"p": th.distributions.Beta(1, 9)},
        nx.newman_watts_strogatz_graph,
        {"k": 5},
        localization=0,
    ),
    # Redirection model for web/citation networks. Beta(2,2) concentrates in [0.2, 0.8].
    # Reference: Krapivsky & Redner (2001) https://doi.org/10.1103/PhysRevE.63.066123
    "redirection_graph": Configuration(
        {"redirection_proba": th.distributions.Beta(2, 2)},
        generators.redirection_graph,
        {"max_num_connections": 1},
        localization=1,
    ),
    # Phase transition at p=0.5: sparse networks for p<0.5, dense for p>=0.5.
    # Beta(2,5) mean=0.29, keeps mass in sparse regime. ABC studies use Uniform[0.15, 0.35].
    # References:
    # - Lambiotte et al. (2016) https://doi.org/10.1103/PhysRevLett.117.218301
    # - Sheridan et al. (2022) https://doi.org/10.1214/22-BA1321
    "copy_graph": Configuration(
        {"copy_proba": th.distributions.Beta(2, 5)},
        generators.copy_graph,
        localization=1,
    ),
    # Duplication-mutation model for PPI networks.
    # - mutation_proba: typically 0.1-0.5, Beta(2,3) mean=0.4
    # - divergence_proba: ~0.6 from fitting to yeast/fly/human PPI, Beta(3,2) mean=0.6
    # References:
    # - Vazquez et al. (2003) https://doi.org/10.1159/000067642
    # - Ispolatov et al. (2005) https://doi.org/10.1103/PhysRevE.71.061911
    "duplication_mutation_graph": Configuration(
        {
            "mutation_proba": th.distributions.Beta(2, 3),
            "divergence_proba": th.distributions.Beta(3, 2),
        },
        generators.duplication_mutation_graph,
        localization=1,
    ),
    # Duplication-mutation with complementarity (DMC) model for PPI networks.
    # - divergence_proba: ~0.6 from fitting to yeast/fly/human PPI, Beta(3,2) mean=0.6
    # References:
    # - Li et al. (2015) https://doi.org/10.1089/cmb.2015.0072
    # - Ispolatov et al. (2005) https://doi.org/10.1103/PhysRevE.71.061911
    "duplication_complementation_graph": Configuration(
        {
            "interaction_proba": th.distributions.Beta(2, 3),
            "divergence_proba": th.distributions.Beta(3, 2),
        },
        generators.duplication_complementation_graph,
    ),
    # random_geometric_graph: (nx.random_geometric_graph, {"dim": 2}),
    # waxman_graph: (nx.waxman_graph, {}),
    # web_graph: (generators.web_graph, {"dist_degree_new": np.arange(3) == 2}),
    # planted_partition_graph: (_planted_partition_graph, {"num_groups": 2}),
    # Small-world properties emerge at p ~ 0.01-0.1. Beta(1,9) mean=0.1.
    # Reference: Watts & Strogatz (1998) https://doi.org/10.1038/30918
    "watts_strogatz_graph": Configuration(
        {"p": th.distributions.Beta(1, 9)}, nx.watts_strogatz_graph, {"k": 5}
    ),
    # latent_space_graph: (_latent_space_graph, {"dim": 2}),
    # Start with two connected nodes as described in 10.1155/2008/190836 for numerical experiments.
    # partial_duplication_graph: (nx.partial_duplication_graph, {"n": 2}),
    # thresholded_random_geometric_graph: (nx.thresholded_random_geometric_graph, {}),
    # geographical_threshold_graph: (nx.geographical_threshold_graph, {}),
    # degree_attachment_graph: (generators.degree_attachment_graph, {"m": 4}),
    # rank_attachment_graph: (generators.rank_attachment_graph, {"m": 4}),
    # Social network formation via random and network-based meetings.
    # Reference: Jackson & Rogers (2007) https://doi.org/10.1257/aer.97.3.890
    "jackson_rogers_graph": Configuration(
        {
            "pr": th.distributions.Beta(1, 1),
            "pn": th.distributions.Beta(1, 1),
        },
        generators.jackson_rogers_graph,
        {"mr": 4},
        localization=None,
    ),
    # "surfer_graph": Configuration(
    #     {"hop_proba": th.distributions.Beta(1, 1)},
    #     generators.surfer_graph,
    # ),
    # "long_surfer_graph": Configuration(
    #     {"hop_proba": th.distributions.Beta(4, 2)},
    #     generators.surfer_graph,
    # ),
    # Citation network with preferential attachment. gamma=1 is linear PA.
    # Empirical estimates suggest gamma ~ 0.8-1.0 for most networks.
    # Reference: Krapivsky et al. (2000) https://doi.org/10.1103/PhysRevLett.85.4629
    "gn_graph": Configuration(
        {"gamma": th.distributions.Beta(1, 1)},
        _gn_graph,
    ),
    # Extended range gamma in [0, 2] for sub/superlinear attachment.
    # gamma < 1: sublinear (stretched exponential degree dist)
    # gamma > 1: superlinear (winner-take-all dynamics)
    # Reference: Krapivsky et al. (2000) https://doi.org/10.1103/PhysRevLett.85.4629
    "gn_graph02": Configuration(
        {"gamma": th.distributions.Uniform(0, 2)},
        _gn_graph,
    ),
    # "latent_space_graph": Configuration(
    #     {"bias": th.distributions.Normal(1, 1), "scale": th.distributions.Gamma(2, 2)},
    #     generators.latent_space_graph,
    #     {"num_dims": 2},
    # ),
}
