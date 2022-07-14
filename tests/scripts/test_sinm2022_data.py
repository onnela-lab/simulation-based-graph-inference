from doit_interface import dict2args
import pytest
from simulation_based_graph_inference.scripts import sinm2022_data


def test_sinm2022_data_dtype(tmpwd: str):
    sinm2022_data.__main__(dict2args(generator="generate_poisson_random_attachment", directory=".",
                                     num_samples=3))
    with pytest.raises(ValueError):
        sinm2022_data.__main__(dict2args(
            generator="generate_poisson_random_attachment", directory=".", num_samples=3,
            num_nodes=100_000, dtype="int16",
        ))
