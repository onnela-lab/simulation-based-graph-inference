import itertools as it
import json
import numbers
import pathlib
import shutil
import torch as th
from tqdm import tqdm
import typing


class PersistentDataset(th.utils.data.Dataset):
    """
    Persistent dataset for simulated graphs.

    Note:
        Multiple iterations over the dataset will return the same results unless the dataset was
        :meth:`reset`.

    Args:
        root: Directory for storing simulated graphs.
        length: Number of graphs to generate (loaded from `[root]/length` if not given).
        func: Callable to generate graphs (can be omitted for a pre-generated dataset).
        args: Positional arguments passed to `func`.
        kwargs: Keyword arguments passed to `func`.
        progress: Whether to show a progress bar or a callable that can wrap an iterable.
        transform: Callable to transform an item in the dataset.
    """
    def __init__(self, root: str, length: int = None, func: typing.Callable = None,
                 args: typing.Iterable = None, kwargs: typing.Mapping = None,
                 progress: bool = False, transform: typing.Callable = None) -> None:
        self.root = pathlib.Path(root)

        # Load the length if it is missing and validate it.
        if not length:
            try:
                with open(self.root / "length") as fp:
                    length = int(fp.read().strip())
            except FileNotFoundError as ex:
                raise ValueError(f"length cannot be loaded from the root directory `{self.root}` "
                                 "and must be given") from ex
        if not isinstance(length, numbers.Integral) or length <= 0:
            raise ValueError("length must be a positive integer")

        self.length = length
        self.func = func
        self.args = args or []
        self.kwargs = kwargs or {}
        self.transform = transform

        # Prepare the iterator for loading data.
        self.root.mkdir(parents=True, exist_ok=True)
        steps = range(self.length)
        if isinstance(progress, typing.Callable):
            steps = progress(steps)
        elif progress:
            steps = tqdm(steps)

        # Generate the data.
        for i in steps:
            path = self._get_path(i)
            if path.is_file():
                continue
            if self.func is None:
                raise RuntimeError(f"func to generate data must be given because element {i} does "
                                   f"not exist at {path}")
            data = self.func(*self.args, **self.kwargs)
            th.save(data, path)

        with open(self.root / "length", "w") as fp:
            fp.write(str(self.length))

    def _get_path(self, index: int) -> pathlib.Path:
        return self.root / f"{index}.pt"

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index >= self.length:
            raise StopIteration
        item = th.load(self._get_path(index))
        if self.transform:
            item = self.transform(item)
        return item

    def reset(self):
        """
        Reset the dataset by deleting the root directory.
        """
        shutil.rmtree(self.root)


class BatchedDataset(th.utils.data.IterableDataset):
    """
    Dataset to load items from batches of data. If `num_concurrent == 1` and `not shuffle`, the
    elements are yielded in order.

    Args:
        root: Directory from which to load data (must contain a file `meta.json` that contains
            metadata).
        num_concurrent: Number of concurrent batches to load.
        shuffle: Whether to shuffle batches and elements in each batch.
    """
    def __init__(self, root: str, num_concurrent: int = 1, shuffle: bool = False):
        self.root = pathlib.Path(root)
        self.num_concurrent = num_concurrent
        self.shuffle = shuffle

        with open(self.root / "meta.json") as fp:
            self.meta = json.load(fp)

    def __len__(self):
        return self.meta["length"]

    def __iter__(self):
        filenames = self.meta["filenames"]
        if self.shuffle:
            filenames = [filenames[i] for i in th.randperm(len(filenames))]

        iterators = []
        while True:
            # Populate concurrent iterators.
            while len(iterators) < self.num_concurrent and filenames:
                filename = filenames.pop(0)
                batch = th.load(self.root / filename)
                if self.shuffle:
                    batch = [batch[i] for i in th.randperm(len(batch))]
                iterators.append(iter(batch))

            # We are done if there are no more iterators.
            if not iterators:
                return

            # Iterate over the elements in batches, and only retain them if they are not exhausted.
            next_iterators = []
            for iterator in iterators:
                try:
                    yield next(iterator)
                    next_iterators.append(iterator)
                except StopIteration:
                    pass
            iterators = next_iterators

    @classmethod
    def generate(
            cls, root: pathlib.Path, batch_size: int, num_batches: int, func: typing.Callable,
            args: typing.Iterable = None, kwargs: typing.Mapping = None, progress: bool = False) \
            -> None:
        """
        Generate a batched dataset.

        Args:
            root: Directory to store the batched data.
            batch_size: Number of elements per batch.
            num_batches: Number of batches.
            func: Callable to generate elements for each batch.
            args: Positional arguments passed to `func`.
            kwargs: Keyword arguments passed to `func`.
            progress: Show a progress bar.
        """
        root = pathlib.Path(root)
        root.mkdir(parents=True, exist_ok=True)
        args = args or []
        kwargs = kwargs or {}
        meta = {
            "num_batches": num_batches,
            "batch_size": batch_size,
            "length": num_batches * batch_size,
        }

        # Set up the batch iterator.
        batches = range(num_batches)
        if isinstance(progress, typing.Callable):
            batches = progress(batches)
        elif progress:
            batches = tqdm(batches)

        for i in batches:
            filename = f"{i}.pt"
            batch = [func(*args, **kwargs) for _ in range(batch_size)]
            th.save(batch, root / filename)
            meta.setdefault("filenames", []).append(filename)

        with open(root / "meta.json", "w") as fp:
            json.dump(meta, fp)

        return meta


class SimulatedDataset(th.utils.data.IterableDataset):
    """
    Dataset that yields synthetic data from a simulator.

    Note:
        Multiple iterations over the dataset will return different results.

    Args:
        simulator: Callable to generate data.
        args: Positional arguments for the simulator.
        kwargs: Keyword arguments for the simulator.
        length: Maximum number of simulations.
    """
    def __init__(self, simulator: typing.Callable, args: typing.Iterable = None,
                 kwargs: typing.Mapping = None, length: int = None):
        super().__init__()
        self.simulator = simulator
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.length = length

    def __len__(self):
        if self.length is None:
            raise TypeError('length of `SimulatedDataset` has not been specified')
        return self.length

    def __iter__(self):
        num_simulations = 0
        while self.length is None or num_simulations < self.length:
            simulation = self.simulator(*self.args, **self.kwargs)
            num_simulations += 1
            yield simulation


class InterleavedDataset(th.utils.data.IterableDataset):
    """
    Meta dataset that interleaves different datasets.

    Args:
        datasets: Datasets to interleave.
        longest: If `True`, yield all elements of all datasets even if some are exhausted. If
            `False`, yield elements until one or more datasets are exhausted.
    """
    def __init__(self, datasets: typing.Iterable[th.utils.data.Dataset], longest: bool = False):
        self.datasets = datasets
        self.longest = longest

    def __len__(self):
        if self.longest:
            return sum(len(dataset) for dataset in self.datasets)
        else:
            return len(self.datasets) * min(len(dataset) for dataset in self.datasets)

    def __iter__(self):
        for batch in it.zip_longest(*self.datasets, fillvalue=StopIteration):
            filtered_batch = [element for element in batch if element is not StopIteration]
            if not self.longest and len(filtered_batch) < len(batch):
                return
            for element in filtered_batch:
                yield element
