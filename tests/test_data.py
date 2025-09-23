import itertools as it
import numpy as np
import pathlib
import pytest
from simulation_based_graph_inference import data
import torch as th
from torch import distributions as dists
from torch.utils.data import DataLoader, TensorDataset


def test_simulated_dataset():
    batch_size = 16
    num_batches = 4
    num_simulations = 0

    def _simulate():
        nonlocal num_simulations
        num_simulations += 1
        return dists.Normal(0, 1).sample([7])

    dataset = data.SimulatedDataset(_simulate)
    loader = DataLoader(dataset, batch_size=batch_size)
    for i, batch in enumerate(loader):
        assert batch.shape == (batch_size, 7)
        if i > num_batches:
            break
    assert num_simulations == batch_size * (num_batches + (loader.prefetch_factor or 2))
    with pytest.raises(TypeError):
        len(dataset)


@pytest.mark.parametrize('num_iterations', [1, 7])
def test_simulated_dataset_with_length(num_iterations):
    batch_size = 16
    dataset = data.SimulatedDataset(dists.Normal(0, 1).sample, ([7],), length=37)
    loader = DataLoader(dataset, batch_size=batch_size)
    for _ in range(num_iterations):
        batch_sizes = [batch.shape[0] for batch in loader]
        assert batch_sizes == [16, 16, 5]
        assert len(dataset) == dataset.length


@pytest.mark.parametrize('longest', [False, True])
def test_interleaved_dataset(longest):
    datasets = [TensorDataset(i * th.ones((10 + i, 3)), i * th.ones(10 + i)) for i in range(5)]
    interleaved = data.InterleavedDataset(datasets, longest)

    if longest:
        assert len(interleaved) == 60
    else:
        assert len(interleaved) == 50

    for i, (X, y) in enumerate(interleaved):
        if i < 50:
            np.testing.assert_allclose(X, i % len(datasets))


@pytest.mark.parametrize("num_concurrent", [1, 3])
@pytest.mark.parametrize("progress", [True, False, lambda x: x])
@pytest.mark.parametrize("shuffle", [True, False])
def test_batched_dataset_generate(tmpwd: str, progress, num_concurrent: int, shuffle: bool):
    sequence = it.count()
    iterator = iter(sequence)
    meta = data.BatchedDataset.generate(tmpwd, 3, 7, next, [iterator], progress=progress)

    assert len(meta["filenames"]) == meta["num_batches"]
    assert all((pathlib.Path(tmpwd) / filename).is_file() for filename in meta["filenames"])

    dataset = data.BatchedDataset(tmpwd, num_concurrent, shuffle)
    if shuffle or num_concurrent > 1:
        assert set(dataset) == set(range(21))
    else:
        assert list(dataset) == list(range(21))

    # Make sure we can iterate multiple times.
    if shuffle:
        assert set(dataset) == set(dataset) and set(dataset)
    else:
        assert list(dataset) == list(dataset) and list(dataset)

    # Check we get a complaint if the indices are wrong.
    with pytest.raises(ValueError):
        data.BatchedDataset(tmpwd, num_concurrent, shuffle, index_batches=[None for _ in range(5)])


@pytest.mark.parametrize("num_concurrent", [1, 3])
@pytest.mark.parametrize("shuffle", [True, False])
def test_batched_dataset_bootstrap_split(num_concurrent, shuffle, tmpwd: str):
    sequence = it.count()
    iterator = iter(sequence)
    num_batches = 97
    batch_size = 89
    meta = data.BatchedDataset.generate(tmpwd, batch_size, num_batches, next, [iterator])
    assert meta["batch_size"] == batch_size
    assert meta["num_batches"] == num_batches

    dataset = data.BatchedDataset(tmpwd, num_concurrent, shuffle)
    assert len(dataset) == num_batches * batch_size
    bootstrap, out_of_bag = dataset.bootstrap_split()

    num_oob = 0
    for i in range(num_batches):
        bs_indices = set(map(int, bootstrap.index_batches[i]))
        oob_indices = set(map(int, out_of_bag.index_batches[i]))
        # All data used somehow.
        assert bs_indices | oob_indices == set(range(batch_size))
        # Now overlap.
        assert not bs_indices & oob_indices
        # Unique out of bag samples.
        assert len(oob_indices) == len(out_of_bag.index_batches[i])
        num_oob += len(oob_indices)

    # We expect exp(-1) = 0.368 to be "outside the bag" (see
    # https://en.wikipedia.org/wiki/Bootstrap_aggregating#Description_of_the_technique).
    assert 0.3 < num_oob / len(dataset) < 0.4

    # Check that iteration gives the right number of elements.
    assert sum(1 for _ in bootstrap) == len(dataset)
    assert sum(1 for _ in out_of_bag) == num_oob
    assert len(out_of_bag) == num_oob
