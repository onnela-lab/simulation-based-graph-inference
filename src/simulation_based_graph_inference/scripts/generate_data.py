from datetime import datetime
import torch as th
import typing
from .util import get_parser
from ..data import BatchedDataset
from .. import config, data


def __main__(argv: typing.Optional[list[str]] = None) -> None:
    parser = get_parser()
    parser.add_argument(
        "--num_nodes", "-n", help="number of nodes", type=int, required=True
    )
    parser.add_argument(
        "--batch_size", type=int, help="number of samples per batch", required=True
    )
    parser.add_argument(
        "--num_batches", type=int, help="number of batches", required=True
    )
    parser.add_argument("--directory", help="path to store the dataset", required=True)
    parser.add_argument(
        "--dtype",
        help="dtype for edge indices",
        default="int16",
        choices=["int16", "int32", "int64"],
    )
    parser.add_argument(
        "--no_clustering",
        help="do not precompute clustering coefficients",
        action="store_true",
    )
    args = parser.parse_args(argv)

    # Make sure we can store the samples.
    dtype = getattr(th, args.dtype)
    imax = th.iinfo(dtype).max
    if args.num_nodes > imax + 1:
        raise ValueError(f"cannot represent {args.num_nodes} using {dtype}")

    # Set up the generator and model.
    generator_config = config.GENERATOR_CONFIGURATIONS[args.configuration]

    # Sample all parameters upfront for reproducibility and easier debugging.
    num_samples = args.batch_size * args.num_batches
    all_params = generator_config.sample_params(th.Size([num_samples]))

    # Create an iterator that yields params for each sample.
    params_iter = iter(
        [{k: v[i] for k, v in all_params.items()} for i in range(num_samples)]
    )

    def generate_with_params():
        return data.generate_data(
            generator_config,
            args.num_nodes,
            dtype=dtype,
            clustering=not args.no_clustering,
            params=next(params_iter),
        )

    # Prepare the dataset.
    start = datetime.now()
    meta = BatchedDataset.generate(
        args.directory,
        args.batch_size,
        args.num_batches,
        generate_with_params,
        progress=True,
    )
    duration = datetime.now() - start
    print(
        f"saved {meta['length']} samples of {args.configuration} to {args.directory} in "
        f"{duration}"
    )


if __name__ == "__main__":  # pragma: no cover
    __main__()
