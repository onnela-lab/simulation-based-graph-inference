from doit_interface import dict2args
import pytest
from simulation_based_graph_inference.scripts import generate_data


def test_sinm2022_data_dtype(tmpwd: str):
    generate_data.__main__(dict2args(configuration="poisson_random_attachment_graph", directory=".",
                                     num_batches=3, batch_size=2))
    with pytest.raises(ValueError):
        generate_data.__main__(dict2args(
            configuration="poisson_random_attachment_graph", directory=".", num_batches=2,
            batch_size=3, num_nodes=100_000, dtype="int16",
        ))
