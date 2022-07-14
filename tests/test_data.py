import numpy as np
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
    assert num_simulations == batch_size * (num_batches + loader.prefetch_factor)
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


def test_persistent_dataset(tmpwd: str):
    # First step to generate the dataset.
    length = 10
    dataset = data.PersistentDataset(tmpwd, length, th.randn, [3, 4])
    assert len(dataset) == length
    result = th.vstack(list(dataset))
    assert result.shape == (length * 3, 4)

    # Do it again.
    dataset = data.PersistentDataset(tmpwd)
    assert len(dataset) == length
    other = result = th.vstack(list(dataset))
    np.testing.assert_array_equal(result, other)


def test_persistent_dataset_uninitialized(tmpwd: str):
    with pytest.raises(ValueError):
        data.PersistentDataset(tmpwd)


def test_invalid_length(tmpwd: str):
    with pytest.raises(ValueError):
        data.PersistentDataset(tmpwd, 0)
    with pytest.raises(ValueError):
        data.PersistentDataset(tmpwd, "3")


def test_missing_func(tmpwd: str):
    with pytest.raises(RuntimeError):
        data.PersistentDataset(tmpwd, 3)


def test_reset(tmpwd: str):
    dataset = data.PersistentDataset("data", 3, th.randn, [5])
    assert dataset.root.is_dir()
    dataset.reset()
    assert not dataset.root.is_dir()


@pytest.mark.parametrize("progress", [True, False, lambda x: x])
def test_progress(tmpwd: str, progress):
    data.PersistentDataset("data", 3, th.randn, [5], progress=progress)
