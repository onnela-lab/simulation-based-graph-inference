from datetime import datetime
import pickle
import torch as th
import torch_geometric as tg
import torch_scatter as ts
from tqdm import tqdm
import typing
from .util import get_parser, get_prior
from ..convert import to_edge_index
from ..data import SimulatedDataset
from .. import generators
from ..graph import Graph


GENERATORS = [
    "generate_duplication_mutation_complementation",
    "generate_duplication_mutation_random",
    "generate_poisson_random_attachment",
    "generate_redirection",
]


class Model(th.nn.Module):
    """
    Model for conditional posterior density estimation for networks.

    Args:
        depth: Depth of the simple graph isomorphism feature extractor.
        dists: Mapping of `(num_el, dist)` keyed by parameter name. `num_el` is the number of
            elements (per batch) passed to `dist` which returns a
            :class:`th.distributions.Distribution`.
        hidden_units: Number of units per hidden layer.
        activation: Nonlinear activation function applied between layers.
    """
    def __init__(self, depth: int, dists: typing.Mapping[str, typing.Tuple[int, typing.Callable]],
                 hidden_units: typing.Tuple[int] = (64, 64),
                 activation: typing.Optional[th.nn.Module] = None):
        super().__init__()
        self.depth = depth
        self.dists = dists
        self.activation = activation or th.nn.Tanh()
        self.conv = tg.nn.GINConv(lambda x: x)

        # Dense network to transform the features from the sGIN.
        layers = []
        previous = max(self.depth, 1)
        for num_hidden in hidden_units:
            layers.append(th.nn.Linear(previous, num_hidden))
            layers.append(self.activation)
            previous = num_hidden
        self.dense = th.nn.Sequential(*layers)

        # Mapping from dense layer to parameters that we can pass into distributions.
        self.params = th.nn.ModuleDict({key: th.nn.Linear(num_hidden, num_el)
                                        for key, (num_el, _) in self.dists.items()})

    def evaluate_features(self, batch):
        """
        Evaluate representations of graphs using mean-pooled hidden layers of simple graph
        isomorphism networks.
        """
        # We use constant features if there are no layers. Not useful except for demonstrating
        # uninformative features extracted by zero-depth networks.
        if not self.depth:
            return th.ones((batch.num_graphs, 1))

        # Successively construct features by appling sGIN layers.
        x = th.ones((batch.num_nodes, 1))
        xs = []
        for _ in range(self.depth):
            x = self.conv(x, batch.edge_index)
            xs.append(x)
        x = th.concat(xs, dim=1)
        assert x.shape == (batch.num_nodes, self.depth)
        # Mean-pool stratified by graph.
        x = ts.scatter(x, batch.batch, dim=0, reduce='mean')
        assert x.shape == (batch.num_graphs, self.depth)
        # Normalise by mean degree for each graph in the batch to bring the representations of
        # layers to vaguely similar scales.
        return x / x[:, 0, None] ** th.arange(self.depth)

    def forward(self, batch):
        # Extract features, transform using the dense network, then estimate the distributions.
        x = self.evaluate_features(batch)
        x = self.dense(x)
        dists = {}
        for key, (num_el, dist) in self.dists.items():
            y = self.params[key](x)
            assert y.shape[-1] == num_el, f"expected {num_el} elements but got {y.shape[-1]}"
            dists[key] = dist(y)
        return dists


def evaluate_parameterized_beta(x: th.Tensor) -> th.distributions.Beta:
    a, b = x.exp().split(1, dim=-1)
    return th.distributions.Beta(a.squeeze(dim=-1), b.squeeze(dim=-1))


def evaluate_parameterized_gamma(x: th.Tensor) -> th.distributions.Gamma:
    a, b = x.exp().split(1, dim=-1)
    return th.distributions.Gamma(a.squeeze(dim=-1), b.squeeze(dim=-1))


def generate(generator: typing.Callable, num_nodes: int, prior: typing.Callable) -> tg.data.Data:
    """
    Generate a graph with the desired number of nodes using the given generator.
    """
    graph = Graph(strict=False)
    params = {key: dist.sample() for key, dist in prior.items()}
    generator(num_nodes, **params, graph=graph)
    assert graph.get_num_nodes() == num_nodes
    edge_index = to_edge_index(graph)
    return tg.data.Data(edge_index=edge_index, num_nodes=num_nodes,
                        **{key: param[None] for key, param in params.items()})


def __main__(args: typing.Optional[list[str]] = None) -> None:
    parser = get_parser(100)
    parser.add_argument("--batch_size", "-b", help="batch size for each optimization step",
                        type=int, default=32)
    parser.add_argument("--steps_per_epoch", "-e", help="number of optimization steps per epoch",
                        type=int, default=32)
    parser.add_argument("--patience", "-p", type=int, default=50, help="number of consecutive "
                        "epochs without loss improvement after which to terminate training")
    parser.add_argument("--test", "-t", help="path at which to store one epoch of test data, "
                        "including evaluation")
    parser.add_argument("num_sGIN_layers", help="number of feature extraction layers", type=int)
    args = parser.parse_args(args)

    # Set up the generator and model.
    generator = getattr(generators, args.generator)
    prior = get_prior(generator)
    if generator is generators.generate_duplication_mutation_complementation:
        dists = {
            "interaction_proba": (2, evaluate_parameterized_beta),
            "divergence_proba": (2, evaluate_parameterized_beta),
        }
    elif generator is generators.generate_duplication_mutation_random:
        dists = {
            "mutation_proba": (2, evaluate_parameterized_beta),
            "deletion_proba": (2, evaluate_parameterized_beta),
        }
    elif generator is generators.generate_poisson_random_attachment:
        dists = {
            "rate": (2, evaluate_parameterized_gamma),
        }
    elif generator is generators.generate_redirection:
        dists = {
            "redirection_proba": (2, evaluate_parameterized_beta),
        }
    else:
        raise ValueError(args.model)  # pragma: no cover
    model = Model(args.num_sGIN_layers, dists)

    # Prepare the dataset and optimizer.
    dataset = SimulatedDataset(generate, (generator, args.num_nodes, prior),
                               length=args.batch_size * args.steps_per_epoch)
    loader = tg.loader.DataLoader(dataset, batch_size=args.batch_size)
    optimizer = th.optim.Adam(model.parameters(), lr=0.01)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, factor=0.5, cooldown=20, min_lr=1e-6
    )

    losses = []
    best_loss = float('inf')
    num_bad_epochs = 0

    start = datetime.now()
    with tqdm() as progress:
        while num_bad_epochs < args.patience:
            epoch_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                dists = model(batch)
                # Verify that the batch dimension is the same for all distributions.
                assert all(dist.batch_shape == (args.batch_size,) for dist in dists.values())
                loss = - sum(dist.log_prob(batch[key]).mean() for key, dist in dists.items())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            losses.append(epoch_loss)
            progress.update()
            scheduler.step(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
            progress.set_description_str(f"bad epochs: {num_bad_epochs}; loss: {epoch_loss:.3f}; "
                                         f"best loss: {best_loss:.3f}")

    if args.test:
        # Full batch evaluation.
        loader = tg.loader.DataLoader(dataset, batch_size=len(dataset))
        with th.no_grad():
            for i, batch in enumerate(loader):
                dists = model(batch)
                log_prob = sum(dist.log_prob(batch[key]) for key, dist in dists.items())
        assert i == 0, "got more than one evaluation batch"
        assert log_prob.shape == (len(dataset),)
        # Store the distributions, batch parameters, and the evaluated log probability.
        end = datetime.now()
        result = {
            "start": start,
            "end": end,
            "duration": (end - start).total_seconds(),
            "dists": dists,
            "params": {key: batch[key] for key in dists},
            "log_prob": log_prob,
            "prior": prior,
            "batch": batch,
        }
        with open(args.test, "wb") as fp:
            pickle.dump(result, fp)


if __name__ == "__main__":  # pragma: no cover
    __main__()
