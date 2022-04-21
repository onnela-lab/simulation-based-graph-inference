import itertools as it
import torch as th
import typing


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
