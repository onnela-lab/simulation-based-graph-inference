from datetime import datetime
import torch as th
from tqdm import tqdm
import typing
from .util import get_parser
from ..data import PersistentDataset
from .. import generators, models


def __main__(args: typing.Optional[list[str]] = None) -> None:
    parser = get_parser(100)
    parser.add_argument("--num_samples", type=int, help="number of samples in the dataset",
                        required=True)
    parser.add_argument("--directory", help="path to store the dataset", required=True)
    parser.add_argument("--dtype", help="dtype for edge indices", default="int16",
                        choices=["int16", "int32", "int64"])
    args = parser.parse_args(args)

    # Make sure we can store the samples.
    dtype = getattr(th, args.dtype)
    imax = th.iinfo(dtype).max
    if args.num_nodes > imax + 1:
        raise ValueError(f"cannot represent {args.num_nodes} using {dtype}")

    # Set up the generator and model.
    generator = getattr(generators, args.generator)
    prior = models.get_prior(generator)

    # Prepare the dataset.
    start = datetime.now()
    with tqdm(total=args.num_samples) as progress:
        def _target(*args, **kwargs):
            result = models.generate_data(*args, **kwargs)
            progress.update()
            return result

        PersistentDataset(args.directory, args.num_samples, _target,
                          (generator, args.num_nodes, prior), {"dtype": dtype})
    duration = datetime.now() - start
    print(f"saved {args.num_samples} samples of {args.generator} to {args.directory} in "
          f"{duration}")


if __name__ == "__main__":  # pragma: no cover
    __main__()
