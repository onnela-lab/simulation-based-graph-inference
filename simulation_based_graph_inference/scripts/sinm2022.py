from datetime import datetime
import pickle
import torch as th
import torch_geometric as tg
from tqdm import tqdm
import typing
from .util import get_parser
from ..data import SimulatedDataset
from .. import generators, models


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
    parser.add_argument("conv", help="sequence of graph isomorphism convolutional layers separated "
                        "by underscores; e.g. `simple_8,8_norm` denotes a three-layer network "
                        "comprising a simple layer, a two-layer transform, and a normalized simple "
                        "layer")
    parser.add_argument("dense", help="sequence of number of hidden units for the dense "
                        "graph-level transform")
    args = parser.parse_args(args)

    # Set up the generator and model.
    generator = getattr(generators, args.generator)
    prior = models.get_prior(generator)

    # Set up the convoluational network for node-level representations.
    activation = th.nn.Tanh()
    if args.conv == "none":
        conv = None
    else:
        conv = []
        for layer in args.conv.split('_'):
            if layer == "simple":
                conv.append(tg.nn.GINConv(lambda x: x))
            elif layer == "norm":
                conv.append(models.Normalize(tg.nn.GINConv(lambda x: x)))
            else:
                nn = models.create_dense_nn(map(int, layer.split(',')), activation, True)
                conv.append(tg.nn.GINConv(nn))

    # Set up the dense network for transforming graph-level representations.
    dense = models.create_dense_nn(map(int, args.dense.split(',')), activation, True)

    # Set up the parameterized distributions and model.
    dists = models.get_parameterized_posterior_density_estimator(generator)
    model = models.Model(conv, dense, dists)

    # Prepare the dataset and optimizer.
    dataset = SimulatedDataset(models.generate_data, (generator, args.num_nodes, prior),
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
