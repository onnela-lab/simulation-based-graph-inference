from __future__ import annotations
from datetime import datetime
import itertools as it
import json
import networkx as nx
import numpy as np
import pathlib
import torch as th
import torch_geometric as tg
from tqdm import tqdm
import typing
from . import config
from .util import to_edge_index


def generate_data(generator_config: config.Configuration, num_nodes: int,
                  dtype=th.long) -> tg.data.Data:
    """
    Generate a graph in :mod:`torch_geometric` data format.

    Args:
        generator_config: Generator to obtain a synthetic graph.
        num_nodes: Number of nodes in the synthetic graph.
        dtype: Data type of the `edge_index` tensor.

    Returns:
        data: Synthetic graph in :mod:`torch_geometric` data format.
    """
    params = generator_config.sample_params()
    graph: nx.Graph = generator_config.sample_graph(num_nodes, **params)
    if len(graph) != num_nodes:  # pragma: no cover
        raise ValueError(f"expected {num_nodes} but {generator_config} generated {len(graph)}")
    edge_index = to_edge_index(graph, dtype=dtype)
    return tg.data.Data(edge_index=edge_index, num_nodes=num_nodes,
                        **{key: param[None] for key, param in params.items()})


class BatchedDataset(th.utils.data.IterableDataset):
    """
    Dataset to load items from batches of data. If `num_concurrent == 1` and `not shuffle`, the
    elements are yielded in order.

    Args:
        root: Directory from which to load data (must contain a file `meta.json` that contains
            metadata).
        num_concurrent: Number of concurrent batches to load.
        shuffle: Whether to shuffle batches and elements in each batch.
        transform: Transform applied to every element.
        index_batches: Indices to select for each batch.
    """
    def __init__(self, root: str, num_concurrent: int = 1, shuffle: bool = False,
                 transform: typing.Callable = None, index_batches: list[th.Tensor] = None) -> None:
        self.root = pathlib.Path(root)
        self.num_concurrent = num_concurrent
        self.shuffle = shuffle
        self.transform = transform
        self.index_batches = index_batches

        with open(self.root / "meta.json") as fp:
            self.meta = json.load(fp)

        if self.index_batches is not None and len(self.index_batches) != self.meta["num_batches"]:
            raise ValueError(f"dataset has {self.meta['num_batches']} batches, but "
                             f"{len(self.index_batches)} index batches were supplied")

    def __len__(self):
        if self.index_batches is None:
            return self.meta["length"]
        else:
            return sum(len(index_batch) for index_batch in self.index_batches)

    def __iter__(self):
        if self.index_batches is None:
            index = th.arange(self.meta["batch_size"])
            batches = [(filename, index) for filename in self.meta["filenames"]]
        else:
            batches = [(filename, index) for filename, index in
                       zip(self.meta["filenames"], self.index_batches)]
        if self.shuffle:
            batches = [batches[i] for i in th.randperm(self.meta["num_batches"])]

        iterators = []
        while True:
            # Populate concurrent iterators.
            while len(iterators) < self.num_concurrent and batches:
                filename, index = batches.pop(0)
                batch = th.load(self.root / filename)
                if self.shuffle:
                    index = index[th.randperm(len(index))]
                batch = [batch[i] for i in index]
                iterators.append(iter(batch))

            # We are done if there are no more iterators.
            if not iterators:
                return

            # Iterate over the elements in batches, and only retain them if they are not exhausted.
            next_iterators = []
            for iterator in iterators:
                try:
                    element = next(iterator)
                    if self.transform:
                        element = self.transform(element)
                    yield element
                    next_iterators.append(iterator)
                except StopIteration:
                    pass
            iterators = next_iterators

    def _unravel_indices(self, indices: th.Tensor) -> list[th.Tensor]:
        """
        Transform indices over the entire dataset to indices within each batch.
        """
        batch_size = self.meta["batch_size"]
        return [indices[th.div(indices, th.as_tensor(batch_size), rounding_mode="floor") == batch]
                % batch_size for batch in range(self.meta["num_batches"])]

    def bootstrap_split(self, num_concurrent: int = None, shuffle: bool = None,
                        transform: typing.Callable = None) -> tuple[BatchedDataset, BatchedDataset]:
        """
        Create a bootstrapped dataset by sampling without replacement and the out-of-bag dataset.

        Args:
            num_concurrent: Number of concurrent batches to load (defaults to value of parent).
            shuffle: Whether to shuffle batches and elements in each batch (defaults to value of
                parent).
            transform: Transform applied to every element (defaults to value of parent).

        Returns:
            bootstrap: Bootstrapped dataset obtained by sampling without replacement.
            out_of_bag: Out-of-bag dataset of instances not included in the bootstrapped dataset.
        """
        if self.index_batches is not None:
            raise NotImplementedError("cannot bootstrap datasets with index batches")
        n = self.meta["length"]
        selected = th.randint(n, size=[n])
        not_selected = th.asarray(np.setdiff1d(th.arange(n), selected))
        kwargs = {
            "num_concurrent": self.num_concurrent if num_concurrent is None else num_concurrent,
            "shuffle": self.shuffle if shuffle is None else self.shuffle,
            "transform": self.transform if transform is None else transform,
        }
        return BatchedDataset(self.root, **kwargs, index_batches=self._unravel_indices(selected)), \
            BatchedDataset(self.root, **kwargs, index_batches=self._unravel_indices(not_selected))

    @classmethod
    def generate(
            cls, root: pathlib.Path, batch_size: int, num_batches: int, func: typing.Callable,
            args: typing.Iterable = None, kwargs: typing.Mapping = None, progress: bool = False) \
            -> dict:
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
        start = datetime.now()
        meta = {
            "num_batches": num_batches,
            "batch_size": batch_size,
            "length": num_batches * batch_size,
            "start": start.isoformat(),
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

        end = datetime.now()
        meta["end"] = end.isoformat()
        meta["duration"] = (end - start).total_seconds()
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
