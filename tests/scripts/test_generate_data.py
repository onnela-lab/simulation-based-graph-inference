from cook import dict2args
import pytest
from simulation_based_graph_inference.data import BatchedDataset
from simulation_based_graph_inference.scripts import generate_data
from torch_geometric.loader import DataLoader


def test_generate_data(tmpwd: str):
    num_nodes = 97
    generate_data.__main__(
        dict2args(
            configuration="redirection_graph",
            directory=tmpwd,
            num_batches=3,
            batch_size=2,
            num_nodes=num_nodes,
        )
    )
    dataset = BatchedDataset(tmpwd)
    loader = DataLoader(dataset, batch_size=3)  # type: ignore[arg-type]
    batch_size = 3
    for batch in loader:
        assert batch.num_nodes == num_nodes * batch_size
        assert batch.clustering_coefficient.shape == (num_nodes * batch_size,)


def test_generate_data_dtype_too_small(tmpwd: str):
    with pytest.raises(ValueError):
        generate_data.__main__(
            dict2args(
                configuration="redirection_graph",
                directory=tmpwd,
                num_batches=2,
                batch_size=3,
                num_nodes=100_000,
                dtype="int16",
            )
        )
