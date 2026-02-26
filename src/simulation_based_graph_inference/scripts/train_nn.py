import contextlib
import copy
from datetime import datetime
import numpy as np
import pickle
import torch as th
from torch_geometric import nn as tgnn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import typing
from typing import cast
from .util import get_parser
from ..data import BatchedDataset
from ..util import ensure_long_edge_index
from .. import config, models


@contextlib.contextmanager
def null_context(*args, **kwargs):
    yield


def run_epoch(
    model: models.Model,
    loader: DataLoader,
    epsilon: float,
    optimizer: th.optim.Optimizer | None = None,
) -> dict:
    """
    Run one epoch for the model using data from the given loader.

    Args:
        model: Model to estimate posterior densities.
        loader: Data loader yielding batches of graphs.
        epsilon: L2 penalty parameter for latent represenations after convolutional layer.
        optimizer: Optimizer to train the model. The model is not trained if no optimizer is given.

    Returns:
        epoch_loss: Mean negative log-probability loss evaluated on data provided by the loader.
        dists: Posterior density estimates for the last batch keyed by parameter name.
    """
    epoch_loss = 0
    num_batches = 0
    batch = None
    dists = None
    log_prob = None
    features: th.Tensor | None = None
    num_items = 0
    for batch in loader:
        # Reset gradients if we're training the model.
        if optimizer:
            optimizer.zero_grad()

        # Apply the model to get posterior estimates and evaluate the loss.
        with null_context() if optimizer else th.no_grad():
            dists, features = model(batch)
            assert dists, (
                "There must be at least one output distribution for parameters."
            )
            assert features.ndim == 2
            assert features.shape[0] == batch.num_graphs
            assert (
                batch.num_graphs == loader.batch_size or num_batches == len(loader) - 1
            )
            assert all(
                dist.batch_shape == (batch.num_graphs,) for dist in dists.values()
            )
            log_prob = cast(
                th.Tensor, sum(dist.log_prob(batch[key]) for key, dist in dists.items())
            )
            # Evaluate negative log probability loss plus small regularization on latent state. This
            # regularization can't just apply to the norm, however. Otherwise, the model will simply
            # learn to make the embeddings tiny and then scale up in the density estimator. So we
            # want to regularize such that the embeddings are approximately unit vectors. We achieve
            # this "soft" constraint by including a penalty `epsilon * (1 - norm) ** 2` averaged
            # over the batch.
            loss: th.Tensor = (
                -log_prob.mean()
                + epsilon * (features.square().sum(axis=-1) - 1).square().mean()  # pyright: ignore[reportCallIssue]
            )

        # Update the model weights if desired.
        if optimizer:
            loss.backward()
            optimizer.step()

        epoch_loss += batch.num_graphs * loss.item()
        num_batches += 1
        num_items += batch.num_graphs

    if not num_batches:  # pragma: no cover
        raise RuntimeError("did not process any batch")

    return {
        "batch": batch,
        "dists": dists,
        "log_prob": log_prob,
        "epoch_loss": epoch_loss / num_items,
        "num_batches": num_batches,
        "num_items": num_items,
        "features": features,
    }


def dense_from_str(
    layer: str,
    activation: typing.Callable,
    final_activation: bool,
    use_layer_norm: bool = False,
) -> th.nn.Module:
    return models.create_dense_nn(
        map(int, layer.split(",")), activation, final_activation, use_layer_norm
    )


def __main__(argv: typing.Optional[list[str]] = None) -> None:
    try:
        th.set_num_threads(1)
        th.set_num_interop_threads(1)
    except RuntimeError as ex:
        if "number of interop threads" not in str(ex):
            raise
    parser = get_parser()
    parser.add_argument(
        "--batch_size",
        "-b",
        help="batch size for each optimization step",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--patience",
        "-p",
        type=int,
        default=25,
        help="number of consecutive "
        "epochs without loss improvement after which to terminate training",
    )
    parser.add_argument(
        "--result", help="path at which to store evaluation on test set", required=True
    )
    parser.add_argument(
        "--conv",
        help="sequence of graph isomorphism convolutional layers "
        "separated by underscores; e.g. `simple_8,8_norm` denotes a three-layer "
        "network comprising a simple layer, a two-layer transform, and a "
        "normalized simple layer; if starting with `file:`, the convolutional "
        "layers will be loaded from a previous result",
        required=True,
    )
    parser.add_argument(
        "--dense",
        help="sequence of number of hidden units for the dense "
        "graph-level transform; if starting with `file:`, the dense layers will be "
        "loaded from a previous result",
        required=True,
    )
    parser.add_argument("--test", help="path to test set", required=True)
    parser.add_argument("--train", help="path to training set", required=True)
    parser.add_argument("--validation", help="path to validation set", required=True)
    parser.add_argument(
        "--max_num_epochs", help="maximum number of epochs to run", type=int
    )
    parser.add_argument(
        "--epsilon", help="L2 penalty for latent representations", type=float, default=0
    )
    parser.add_argument(
        "--pooling",
        help="pooling strategy for GNN layer outputs: 'concat' (default, concatenate all) or 'last' (use only last layer)",
        type=str,
        default="concat",
        choices=["concat", "last"],
    )
    parser.add_argument(
        "--init-scale",
        help="scale factor for all Linear layer weights after initialization (default: 1.0)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--final-activation",
        help="whether to apply activation function after the final layer in dense networks (default: True)",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
    )
    args = parser.parse_args(argv)

    # Set up the convoluational network for node-level representations.
    activation = th.nn.ReLU()
    use_layer_norm = True
    conv: list[th.nn.Module] | th.nn.ModuleList | None
    if args.conv.startswith("none"):
        conv = None
    elif args.conv.startswith("file:"):
        # This must be a previously-saved result file.
        with open(args.conv.removeprefix("file:"), "rb") as fp:
            conv = pickle.load(fp)["conv"]
        if conv is not None:
            assert isinstance(conv, th.nn.ModuleList)
            for parameter in conv.parameters():
                parameter.requires_grad = False
    else:
        conv = []
        for layer in args.conv.split("_"):
            # Each layer operates independently on a feature representation of shape
            # `(num_nodes, num_features)`.
            if layer == "simple":
                conv.append(tgnn.GINConv(th.nn.Identity()))
            elif layer == "norm":
                conv.append(models.Normalize(tgnn.GINConv(th.nn.Identity())))
            elif layer == "insert-clustering":
                conv.append(models.InsertClusteringCoefficient())
            elif layer.startswith("dropout"):
                _, proba = layer.split("-")
                conv.append(th.nn.Dropout(float(proba)))
            elif layer.startswith("res"):
                _, method, layer = layer.split("-")
                nn = dense_from_str(
                    layer, activation, args.final_activation, use_layer_norm
                )
                conv.append(models.Residual(tgnn.GINConv(nn), method=method))
            else:
                nn = dense_from_str(
                    layer, activation, args.final_activation, use_layer_norm
                )
                conv.append(tgnn.GINConv(nn))

    # Set up the dense network for transforming graph-level representations.
    if args.dense.startswith("file:"):
        # This must be a previously-saved result file.
        with open(args.conv.removeprefix("file:"), "rb") as fp:
            dense: th.nn.Module = pickle.load(fp)["dense"]
        for parameter in dense.parameters():
            parameter.requires_grad = False
    else:
        # Parse dense specification, supporting residual blocks
        dense_layers = []
        for block in args.dense.split("_"):
            if block.startswith("res-"):
                # Parse residual block: "res-scalar-8,8" -> residual around 2-layer MLP
                _, method, layer = block.split("-", 2)
                nn = dense_from_str(
                    layer, activation, args.final_activation, use_layer_norm
                )
                dense_layers.append(models.DenseResidual(nn, method=method))
            else:
                # Regular dense layers
                nn = dense_from_str(
                    block, activation, args.final_activation, use_layer_norm
                )
                dense_layers.append(nn)

        # Combine into sequential or just use single module
        if len(dense_layers) == 1:
            dense = dense_layers[0]
        else:
            dense = th.nn.Sequential(*dense_layers)

    configuration = config.GENERATOR_CONFIGURATIONS[args.configuration]

    # Set up the parameterized distributions and model.
    dists = configuration.create_estimator()
    if isinstance(conv, list):
        conv = th.nn.ModuleList(conv)
    model = models.Model(conv, dense, dists, pooling=args.pooling)

    # Apply weight scaling if specified (only for newly created models)
    if args.init_scale != 1.0 and not args.dense.startswith("file:"):
        # Create dummy batch to materialize lazy layers
        # Minimal batch: 1 graph with 2 nodes and 1 edge
        from torch_geometric.data import Batch, Data

        dummy_graph = Data(
            edge_index=th.tensor([[0], [1]], dtype=th.long),
            num_nodes=2,
        )
        dummy_batch = Batch.from_data_list([dummy_graph])

        # Materialize all lazy layers with dummy forward pass
        with th.no_grad():
            model(dummy_batch)

        # Now scale all Linear layer weights
        models.scale_linear_weights(model, args.init_scale)

    # Prepare the datasets and optimizer. Only shuffle the training set.
    train_dataset = BatchedDataset(
        args.train, transform=ensure_long_edge_index, shuffle=True, num_concurrent=4
    )
    validation_dataset = BatchedDataset(
        args.validation,
        transform=ensure_long_edge_index,
        shuffle=False,
        num_concurrent=1,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)  # pyright: ignore[reportArgumentType]
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)  # pyright: ignore[reportArgumentType]
    optimizer = th.optim.Adam(
        (param for param in model.parameters() if param.requires_grad), lr=0.01
    )
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, cooldown=20, min_lr=1e-6
    )

    losses = {}
    best_loss = float("inf")
    best_conv = None
    best_dense = None
    num_bad_epochs = 0
    epoch = 0

    start = datetime.now()
    params_printed = False  # Track whether we've printed parameter count
    with tqdm() as progress:
        while num_bad_epochs < args.patience and (
            args.max_num_epochs is None or epoch < args.max_num_epochs
        ):
            # Run one training epoch and evaluate the validation loss.
            train_loss = run_epoch(model, train_loader, args.epsilon, optimizer)[
                "epoch_loss"
            ]
            assert np.isfinite(train_loss), f"Loss is not finite: {train_loss}"

            # Print parameter count after first epoch (when lazy layers are initialized)
            if not params_printed:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                print(f"\nTarget: {args.result}")
                print(
                    f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
                )
                params_printed = True
            validation_loss = run_epoch(model, validation_loader, args.epsilon)[
                "epoch_loss"
            ]
            losses.setdefault("train", []).append(train_loss)
            losses.setdefault("validation", []).append(validation_loss)
            progress.update()
            scheduler.step(validation_loss)

            if validation_loss + 1e-6 < best_loss:
                best_loss = validation_loss
                best_conv = copy.deepcopy(model.conv)
                best_dense = copy.deepcopy(model.dense)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
            progress.set_description_str(
                f"bad epochs: {num_bad_epochs}; loss: {validation_loss:.3f}; "
                f"best loss: {best_loss:.3f}"
            )
            epoch += 1

    # Evaluate on the test set using full batch evaluation with both last and best models.
    dataset = BatchedDataset(args.test, transform=ensure_long_edge_index)
    test_loader = DataLoader(dataset, len(dataset))  # pyright: ignore[reportArgumentType]
    model.eval()

    # Evaluate with last model (save references before potential swap).
    last_conv = model.conv
    last_dense = model.dense
    eval_start = datetime.now()
    last_result = run_epoch(model, test_loader, epsilon=0)
    eval_end = datetime.now()
    last_eval_duration = (eval_end - eval_start).total_seconds()
    print(f"Evaluation (last model) on test set took {eval_end - eval_start}.")
    assert last_result["num_batches"] == 1, "got more than one evaluation batch"
    assert last_result["log_prob"].shape == (len(dataset),)

    # Evaluate with best model.
    if best_conv is not None and best_dense is not None:
        model.conv = best_conv
        model.dense = best_dense
        eval_start = datetime.now()
        best_result = run_epoch(model, test_loader, epsilon=0)
        eval_end = datetime.now()
        best_eval_duration = (eval_end - eval_start).total_seconds()
        print(f"Evaluation (best model) on test set took {eval_end - eval_start}.")
        assert best_result["num_batches"] == 1, "got more than one evaluation batch"
        assert best_result["log_prob"].shape == (len(dataset),)
    else:
        # No best model saved (e.g., loaded from file), use last as best.
        best_result = last_result
        best_eval_duration = last_eval_duration

    # Store the distributions, batch parameters, and the evaluated log probability.
    end = datetime.now()
    result = {
        "args": vars(args),
        "configuration": configuration,
        "start": start,
        "end": end,
        "duration": (end - start).total_seconds(),
        "eval_duration": last_eval_duration,
        "best_eval_duration": best_eval_duration,
        # Results from last model.
        "dists": last_result["dists"],
        "params": {key: last_result["batch"][key] for key in last_result["dists"]},
        "log_prob": last_result["log_prob"],
        "features": last_result["features"],
        # Results from best model.
        "best_dists": best_result["dists"],
        "best_params": {key: best_result["batch"][key] for key in best_result["dists"]},
        "best_log_prob": best_result["log_prob"],
        "best_features": best_result["features"],
        # Training info.
        "num_epochs": epoch,
        "losses": {key: th.as_tensor(value) for key, value in losses.items()},
        "conv": args.conv if args.conv.startswith("file:") else last_conv,
        "dense": args.dense if args.dense.startswith("file:") else last_dense,
        "best_conv": args.conv if args.conv.startswith("file:") else best_conv,
        "best_dense": args.dense if args.dense.startswith("file:") else best_dense,
    }
    with open(args.result, "wb") as fp:
        pickle.dump(result, fp)


if __name__ == "__main__":  # pragma: no cover
    __main__()
