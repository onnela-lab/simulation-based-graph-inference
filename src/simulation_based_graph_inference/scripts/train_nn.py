import contextlib
from datetime import datetime
import os
import pickle
import torch as th
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import typing
from .util import get_parser
from ..data import BatchedDataset
from ..util import ensure_long_edge_index
from .. import config, models


@contextlib.contextmanager
def null_context(*args, **kwargs):
    yield


def run_epoch(model: models.Model, loader: DataLoader, epsilon: float,
              optimizer: th.optim.Optimizer = None, max_num_batches: int = None) -> dict:
    """
    Run one epoch for the model using data from the given loader.

    Args:
        model: Model to estimate posterior densities.
        loader: Data loader yielding batches of graphs.
        epsilon: L2 penalty parameter for latent represenations after convolutional layer.
        optimizer: Optimizer to train the model. The model is not trained if no optimizer is given.
        max_num_batches: Maximum number of optimization steps for this epoch.

    Returns:
        epoch_loss: Mean negative log-probability loss evaluated on data provided by the loader.
        dists: Posterior density estimates for the last batch keyed by parameter name.
    """
    epoch_loss = 0
    num_batches = 0
    batch = None
    for batch in loader:
        # Reset gradients if we're training the model.
        if optimizer:
            optimizer.zero_grad()

        # Apply the model to get posterior estimates and evaluate the loss.
        with (null_context() if optimizer else th.no_grad()):
            features: th.Tensor
            dists, features = model(batch)
            assert features.ndim == 2
            assert features.shape[0] == batch.num_graphs
            assert batch.num_graphs == loader.batch_size or num_batches == len(loader) - 1
            assert all(dist.batch_shape == (batch.num_graphs,) for dist in dists.values())
            log_prob = sum(dist.log_prob(batch[key]) for key, dist in dists.items())
            # Evaluate negative log probability loss plus small regularization on latent state. This
            # regularization can't just apply to the norm, however. Otherwise, the model will simply
            # learn to make the embeddings tiny and then scale up in the density estimator. So we
            # want to regularize such that the embeddings are approximately unit vectors. We achieve
            # this "soft" constraint by including a penalty `epsilon * (1 - norm) ** 2` averaged
            # over the batch.
            loss = - log_prob.mean() \
                + epsilon * (features.square().sum(axis=-1) - 1).square().mean()

        # Update the model weights if desired.
        if optimizer:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        if max_num_batches and num_batches >= max_num_batches:
            break

    if batch is None:  # pragma: no cover
        raise RuntimeError("did not process any batch")

    return {
        "batch": batch,
        "dists": dists,
        "log_prob": log_prob,
        "epoch_loss": epoch_loss / num_batches,
        "num_batches": num_batches,
        "features": features,
    }


def dense_from_str(layer: str, activation: typing.Callable, final_activation: bool) -> th.nn.Module:
    return models.create_dense_nn(map(int, layer.split(',')), activation, final_activation)


def __main__(argv: typing.Optional[list[str]] = None) -> None:
    if "CI" not in os.environ:  # pragma: no cover
        th.set_num_threads(1)
        th.set_num_interop_threads(1)
    parser = get_parser(100)
    parser.add_argument("--batch_size", "-b", help="batch size for each optimization step",
                        type=int, default=32)
    parser.add_argument("--steps_per_epoch", "-e", help="number of optimization steps per epoch",
                        type=int, default=32)
    parser.add_argument("--patience", "-p", type=int, default=25, help="number of consecutive "
                        "epochs without loss improvement after which to terminate training")
    parser.add_argument("--result", help="path at which to store evaluation on test set",
                        required=True)
    parser.add_argument("--conv", help="sequence of graph isomorphism convolutional layers "
                        "separated by underscores; e.g. `simple_8,8_norm` denotes a three-layer "
                        "network comprising a simple layer, a two-layer transform, and a "
                        "normalized simple layer; if starting with `file:`, the convolutional "
                        "layers will be loaded from a previous result", required=True)
    parser.add_argument("--dense", help="sequence of number of hidden units for the dense "
                        "graph-level transform; if starting with `file:`, the dense layers will be "
                        "loaded from a previous result", required=True)
    parser.add_argument("--test", help="path to test set", required=True)
    parser.add_argument("--train", help="path to training set", required=True)
    parser.add_argument("--max_num_epochs", help="maximum number of epochs to run", type=int)
    parser.add_argument("--epsilon", help="L2 penalty for latent representations", type=float,
                        default=0)
    args = parser.parse_args(argv)

    # Set up the convoluational network for node-level representations.
    activation = th.nn.Tanh()
    if args.conv.startswith("none"):
        conv = None
    elif args.conv.startswith("file:"):
        # This must be a previously-saved result file.
        with open(args.conv.removeprefix("file:"), "rb") as fp:
            conv: th.nn.Module = pickle.load(fp)["conv"]
        if conv is not None:
            for parameter in conv.parameters():
                parameter.requires_grad = False
    else:
        conv = []
        for layer in args.conv.split('_'):
            # Each layer operates independently on a feature representation of shape
            # `(num_nodes, num_features)`.
            if layer == "simple":
                conv.append(tg.nn.GINConv(th.nn.Identity()))
            elif layer == "norm":
                conv.append(models.Normalize(tg.nn.GINConv(th.nn.Identity())))
            elif layer == "insert-clustering":
                conv.append(models.InsertClusteringCoefficient()),
            elif layer.startswith("dropout"):
                _, proba = layer.split("-")
                conv.append(th.nn.Dropout(float(proba)))
            elif layer.startswith("res"):
                _, method, layer = layer.split("-")
                nn = dense_from_str(layer, activation, True)
                conv.append(models.Residual(tg.nn.GINConv(nn), method=method))
            else:
                nn = dense_from_str(layer, activation, True)
                conv.append(tg.nn.GINConv(nn))

    # Set up the dense network for transforming graph-level representations.
    if args.dense.startswith("file:"):
        # This must be a previously-saved result file.
        with open(args.conv.removeprefix("file:"), "rb") as fp:
            dense: th.nn.Module = pickle.load(fp)["dense"]
        for parameter in dense.parameters():
            parameter.requires_grad = False
    else:
        dense = models.create_dense_nn(map(int, args.dense.split(',')), activation, True)

    configuration = config.GENERATOR_CONFIGURATIONS[args.configuration]

    # Set up the parameterized distributions and model.
    dists = configuration.create_estimator()
    model = models.Model(conv, dense, dists)

    # Prepare the datasets and optimizer. Only shuffle the training set.
    dataset = BatchedDataset(args.train, transform=ensure_long_edge_index, shuffle=True,
                             num_concurrent=4)
    train_dataset, validation_dataset = dataset.bootstrap_split()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
    optimizer = th.optim.Adam((param for param in model.parameters() if param.requires_grad),
                              lr=0.01)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, cooldown=20, min_lr=1e-6
    )

    losses = {}
    best_loss = float("inf")
    num_bad_epochs = 0
    epoch = 0

    start = datetime.now()
    with tqdm() as progress:
        while num_bad_epochs < args.patience and (args.max_num_epochs is None
                                                  or epoch < args.max_num_epochs):
            # Run one training epoch and evaluate the validation loss.
            train_loss = run_epoch(model, train_loader, args.epsilon, optimizer,
                                   args.steps_per_epoch)["epoch_loss"]
            validation_loss = run_epoch(model, validation_loader, args.epsilon)["epoch_loss"]
            losses.setdefault("train", []).append(train_loss)
            losses.setdefault("validation", []).append(validation_loss)
            progress.update()
            scheduler.step(validation_loss)

            if validation_loss + 1e-6 < best_loss:
                best_loss = validation_loss
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
            progress.set_description_str(
                f"bad epochs: {num_bad_epochs}; loss: {validation_loss:.3f}; "
                f"best loss: {best_loss:.3f}"
            )
            epoch += 1

    # Evaluate on the test set using full batch evaluation.
    dataset = BatchedDataset(args.test, transform=ensure_long_edge_index)
    model.eval()
    result = run_epoch(model, DataLoader(dataset, len(dataset)), epsilon=0)
    assert result["num_batches"] == 1, "got more than one evaluation batch"
    assert result["log_prob"].shape == (len(dataset),)

    # Store the distributions, batch parameters, and the evaluated log probability.
    end = datetime.now()
    result = {
        "args": vars(args),
        "configuration": configuration,
        "start": start,
        "end": end,
        "duration": (end - start).total_seconds(),
        "dists": result["dists"],
        "params": {key: result["batch"][key] for key in result["dists"]},
        "log_prob": result["log_prob"],
        "features": result["features"],
        "num_epochs": epoch,
        "losses": {key: th.as_tensor(value) for key, value in losses.items()},
        "conv": args.conv if args.conv.startswith("file:") else model.conv,
        "dense": args.dense if args.dense.startswith("file:") else model.dense,
    }
    with open(args.result, "wb") as fp:
        pickle.dump(result, fp)


if __name__ == "__main__":  # pragma: no cover
    __main__()
