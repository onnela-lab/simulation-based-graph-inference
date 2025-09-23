import inspect
import torch as th
import torch_geometric as tg
from torch_geometric.data import Data
import torch_scatter as ts
import typing
import warnings
from .util import clustering_coefficient


warnings.filterwarnings("ignore", message="Lazy modules are a new feature")


class DistributionModule(th.nn.Module):
    """
    Module to obtain distributions.

    Args:
        distribution_cls: Distribution class to create.
        params: Mapping of modules to parameter names.
        squeeze: Squeeze the last parameter dimension.
        transforms: Transformations to apply to the distribution.
    """

    def __init__(
        self,
        distribution_cls: typing.Callable,
        *,
        squeeze: bool = True,
        transforms: typing.Iterable = None,
        **params,
    ):
        super().__init__()
        self.distribution_cls = distribution_cls
        if not isinstance(params, th.nn.Module) and isinstance(params, typing.Mapping):
            params = th.nn.ModuleDict(params)
        self.params = params
        self.squeeze = squeeze
        self.transforms = transforms

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
        distribution = self.distribution_cls(**params)
        if self.transforms:
            distribution = th.distributions.TransformedDistribution(
                distribution, self.transforms
            )
        return distribution


class Residual(th.nn.Module):
    """
    A residual graph convolutional layer.
    """

    def __init__(self, module, method: str = "identity") -> None:
        super().__init__()
        self._module = module
        self._method = method
        if self._method == "scalar":
            self._scalar = th.nn.Parameter(th.ones([]), requires_grad=True)

    def forward(self, x: th.Tensor, edge_index: th.LongTensor, **kwargs) -> th.Tensor:
        y = self._module(x, edge_index=edge_index, **kwargs)
        if self._method == "identity":
            y = y + x
        elif self._method == "scalar":
            y = y + self._scalar * x
        else:
            raise NotImplementedError
        return y


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
                batch[key] = connections_per_graph = tg.utils.degree(
                    batch.batch[batch.edge_index[0]], batch.num_graphs
                )
            key = "__Normalize:nodes_per_graph"
            if (nodes_per_graph := batch.get(key)) is None:
                batch[key] = nodes_per_graph = tg.utils.degree(
                    batch.batch, batch.num_graphs
                )

            # Evaluate the normalization.
            norm = (connections_per_graph / nodes_per_graph)[batch.batch, None] + 1.0
        else:  # pragma: no cover
            raise ValueError(self.mode)

        batch[norm_key] = norm
        return x / norm


class InsertClusteringCoefficient(th.nn.Module):
    """
    Insert the clustering coefficient as a feature.
    """

    def forward(self, x: th.Tensor, batch: Data) -> th.Tensor:
        num_nodes, _ = x.shape
        try:
            y = getattr(batch, "clustering_coefficient")
        except AttributeError:
            y = clustering_coefficient(batch.edge_index, num_nodes)
        return th.hstack([x, y.to(x)[:, None]])


def create_dense_nn(
    units: typing.Iterable[int], activation: th.nn.Module, final_activation: bool
) -> th.nn.Sequential:
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

    def __init__(
        self,
        conv: typing.Iterable[th.nn.Module],
        dense: th.nn.Module,
        dists: typing.Mapping[str, th.nn.Module],
    ) -> None:
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
            parameters = inspect.signature(conv.forward).parameters
            if "batch" in parameters:
                x = conv(x, batch=batch)
            elif "edge_index" in parameters:
                x = conv(x, edge_index=batch.edge_index)
            else:
                x = conv(x)
            if not getattr(conv, "hidden", False):
                xs.append(x)
        x = th.concat(xs, dim=1)

        # Validate the resulting features.
        num_conv_features = x.shape[-1]
        if x.shape != (
            shape := (batch.num_nodes, num_conv_features)
        ):  # pragma: no cover
            raise ValueError(
                f"expected feature shape {shape} (num_nodes, num_features) but got "
                f"{x.shape}"
            )

        # Mean-pool stratified by graph.
        x = ts.scatter(x, batch.batch, dim=0, reduce="mean")
        if x.shape != (
            shape := (batch.num_graphs, num_conv_features)
        ):  # pragma: no cover
            raise ValueError(
                f"expected feature shape {shape} (num_graphs, num_features) but got "
                f"{x.shape}"
            )
        return x

    def forward(
        self, batch
    ) -> typing.Tuple[typing.Mapping[str, th.distributions.Distribution], th.Tensor]:
        """
        Evaluate posterior density estimates and latent features.

        Args:
            batch: Batch of graphs to apply the model to.

        Returns:
            dists: Mapping from parameter names to mean-field posterior density estimates.
            features: Graph-level features after convolutional and dense transformation.
        """
        x = self.evaluate_graph_features(batch)
        x = self.dense(x)
        return {key: module(x) for key, module in self.dists.items()}, x
