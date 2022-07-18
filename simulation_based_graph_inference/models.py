import inspect
import networkx as nx
import torch as th
import torch_geometric as tg
import torch_scatter as ts
import typing
import warnings
from . import generators
from .util import to_edge_index


warnings.filterwarnings("ignore", message="Lazy modules are a new feature")


class DistributionModule(th.nn.Module):
    """
    Module to obtain distributions.

    Args:
        distribution_cls: Distribution class to create.
        params: Mapping of modules to parameter names.
        squeeze: Squeeze the last parameter dimension.
    """
    def __init__(self, distribution_cls, *, squeeze: bool = True, **params) -> None:
        super().__init__()
        self.distribution_cls = distribution_cls
        if not isinstance(params, th.nn.Module) and isinstance(params, typing.Mapping):
            params = th.nn.ModuleDict(params)
        self.params = params
        self.squeeze = squeeze

    def forward(self, x):
        # Obtain parameters and transform to the constraints of arguments.
        params = {}
        for key, module in self.params.items():
            y = module(x)
            constraint = self.distribution_cls.arg_constraints[key]
            transform = th.distributions.transform_to(constraint)
            y = transform(y)
            if self.squeeze:
                y = y.squeeze(dim=-1)
            params[key] = y
        return self.distribution_cls(**params)


class Normalize(th.nn.Module):
    """
    Normalize the output of a graph convolutional layer.

    Args:
        gcn: Graph convolutional layer.
        mode: Normalization mode.
    """
    MODES = {"mean_degree+1"}

    def __init__(self, gcn: th.nn.Module, mode: str = "mean_degree+1") -> None:
        super().__init__()
        self.gcn = gcn
        if mode not in self.MODES:  # pragma: no cover
            raise ValueError(f"{mode} is not one of {', '.join(self.MODES)}")
        self.mode = mode

    def forward(self, x: th.Tensor, batch) -> th.Tensor:
        # Apply the GCN transform.
        x = self.gcn(x, edge_index=batch.edge_index)
        # Check if the normalization has already been computed and apply it.
        norm_key = f"__Normalize:{self.mode}"
        if (norm := batch.get(norm_key)) is not None:
            return x / norm

        if self.mode == "mean_degree+1":
            # Get the number of nodes and number of edges per graph.
            key = "__Normalize:connections_per_graph"
            if (connections_per_graph := batch.get(key)) is None:
                batch[key] = connections_per_graph = \
                    tg.utils.degree(batch.batch[batch.edge_index[0]], batch.num_graphs)
            key = "__Normalize:nodes_per_graph"
            if (nodes_per_graph := batch.get(key)) is None:
                batch[key] = nodes_per_graph = tg.utils.degree(batch.batch, batch.num_graphs)

            # Evaluate the normalization.
            norm = (connections_per_graph / nodes_per_graph)[batch.batch, None] + 1.
        else:  # pragma: no cover
            raise ValueError(self.mode)

        batch[norm_key] = norm
        return x / norm


def get_prior_and_kwargs(generator: typing.Callable) -> typing.Tuple[
        typing.Mapping[str, th.distributions.Distribution],
        typing.Mapping[str, typing.Any]
        ]:
    """
    Get a prior for the given generator with sensible defaults.

    Args:
        generator: Graph generator for which to get a prior.

    Returns:
        prior: Mapping from parameter names to distributions.
    """
    if generator is generators.duplication_complementation:
        return {
            "interaction_proba": th.distributions.Uniform(0, 1),
            "divergence_proba": th.distributions.Uniform(0, 1),
        }, {}
    elif generator is generators.duplication_mutation:
        return {
            "mutation_proba": th.distributions.Uniform(0, 1),
            "deletion_proba": th.distributions.Uniform(0, 1),
        }, {}
    elif generator is generators.poisson_random_attachment:
        return {
            "rate": th.distributions.Gamma(2, 1),
        }, {}
    elif generator is generators.redirection:
        return {
            "redirection_proba": th.distributions.Uniform(0, 1),
        }, {
            "max_num_connections": 2,
        }
    elif generator is generators.geometric:
        return {
            "scale": th.distributions.Uniform(0, 1),
        }, {
            "kernel": lambda x, scale: x < float(scale),
        }
    else:
        raise ValueError(f"{generator.__name__} is not a known generator")  # pragma: no cover


def get_parameterized_posterior_density_estimator(generator) \
        -> typing.Mapping[str, typing.Tuple[int, typing.Callable]]:
    """
    Get factorized posterior density estimators.

    Args:
        generator: Graph generator for which to get a prior.

    Returns:
        estimator: Mapping from parameter names to a module that returns a
            :class:`torch.distributions.Distribution`.
    """
    if generator is generators.duplication_complementation:
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
    elif generator is generators.duplication_mutation:
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
    elif generator is generators.poisson_random_attachment:
        return {
            "rate": DistributionModule(
                th.distributions.Gamma, concentration=th.nn.LazyLinear(1),
                rate=th.nn.LazyLinear(1),
            )
        }
    elif generator is generators.redirection:
        return {
            "redirection_proba": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
        }
    elif generator is generators.geometric:
        return {
            "scale": DistributionModule(
                th.distributions.Beta, concentration0=th.nn.LazyLinear(1),
                concentration1=th.nn.LazyLinear(1),
            ),
        }
    else:  # pragma: no cover
        raise ValueError(f"{generator.__name__} is not a known generator")


def generate_data(generator: typing.Callable, num_nodes: int,
                  prior: typing.Mapping[str, th.distributions.Distribution],
                  dtype=th.long, **kwargs) -> tg.data.Data:
    """
    Generate a graph in :mod:`torch_geometric` data format.

    Args:
        generator: Generator to obtain a synthetic graph.
        num_nodes: Number of nodes in the synthetic graph.
        prior: Prior to sample from.
        dtype: Data type of the `edge_index` tensor.
        **kwargs: Keyword arguments passed to the generator.

    Returns:
        data: Synthetic graph in :mod:`torch_geometric` data format.
    """
    params = {key: dist.sample() for key, dist in prior.items()}
    graph: nx.Graph = generator(num_nodes, **params, **kwargs)
    if len(graph) != num_nodes:  # pragma: no cover
        raise ValueError(f"expected {num_nodes} but {generator} generated {len(graph)}")
    edge_index = to_edge_index(graph, dtype=dtype)
    return tg.data.Data(edge_index=edge_index, num_nodes=num_nodes,
                        **{key: param[None] for key, param in params.items()})


def create_dense_nn(units: typing.Iterable[int], activation: th.nn.Module, final_activation: bool) \
        -> th.nn.Sequential:
    """
    Get a dense neural network with a given activation between layers.

    Args:
        units: Number of units per layer.
        activation: Activation function between layers.
        final_activation: Whether to add an activation to the last layer.

    Returns:
        dense: Dense neural network conforming to the inputs.
    """
    layers = []
    for num in units:
        layers.append(th.nn.LazyLinear(num))
        layers.append(activation)
    if not final_activation:
        layers.pop(-1)
    return th.nn.Sequential(*layers)


class Model(th.nn.Module):
    """
    Model for conditional posterior density estimation for networks. The model comprises four
    steps:

    1. graph convolutional layers are applied to constant features to obtain node-level
       representations.
    2. the hidden representation of all layers are concatenated and mean-pooled stratified by graph
       to obtain an initial graph-level representation.
    3. a dense network transforms the initial graph-level representation to a final graph-level
       representation.
    4. parameter-specific networks transform the final graph-level representation to posterior
       density estimators.

    Args:
        conv: Sequence of convolutional modules. If a model has an attribute `hidden`, its
            representation is not included in the concatenated representation.
        dense: Dense network that transforms initial to final graph representations.
        dists: Mapping of modules that evaluate distributions keyed by parameter names.
    """
    def __init__(self, conv: typing.Iterable[th.nn.Module], dense: th.nn.Module,
                 dists: typing.Mapping[str, th.nn.Module]) -> None:
        super().__init__()
        if not isinstance(conv, th.nn.Module) and isinstance(conv, typing.Iterable):
            conv = th.nn.ModuleList(conv)
        self.conv = conv
        if not isinstance(dense, th.nn.Module) and isinstance(dense, typing.Iterable):
            dense = th.nn.Sequential(*dense)
        self.dense = dense
        if not isinstance(dists, th.nn.Module) and isinstance(dists, typing.Mapping):
            dists = th.nn.ModuleDict(dists)
        self.dists = dists

    def evaluate_graph_features(self, batch):
        # Return trivial features if there are no convolutional layers.
        if self.conv is None:
            return th.ones((batch.num_graphs, 1))

        # Apply the convolutions to constant features.
        x = th.ones((batch.num_nodes, 1))
        xs = []
        for conv in self.conv:
            if "batch" in inspect.signature(conv.forward).parameters:
                x = conv(x, batch=batch)
            else:
                x = conv(x, edge_index=batch.edge_index)
            if not getattr(conv, "hidden", False):
                xs.append(x)
        x = th.concat(xs, dim=1)

        # Validate the resulting features.
        num_conv_features = x.shape[-1]
        if x.shape != (shape := (batch.num_nodes, num_conv_features)):  # pragma: no cover
            raise ValueError(f"expected feature shape {shape} (num_nodes, num_features) but got "
                             f"{x.shape}")

        # Mean-pool stratified by graph.
        x = ts.scatter(x, batch.batch, dim=0, reduce='mean')
        if x.shape != (shape := (batch.num_graphs, num_conv_features)):  # pragma: no cover
            raise ValueError(f"expected feature shape {shape} (num_graphs, num_features) but got "
                             f"{x.shape}")
        return x

    def forward(self, batch) -> typing.Mapping[str, th.distributions.Distribution]:
        x = self.evaluate_graph_features(batch)
        x = self.dense(x)
        return {key: module(x) for key, module in self.dists.items()}
