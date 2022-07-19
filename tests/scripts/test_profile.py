from doit_interface import dict2args
import pytest
from simulation_based_graph_inference import config
from simulation_based_graph_inference.scripts import profile


@pytest.fixture(params=config.Configuration)
def generator_configuration(request) -> config.Configuration:
    return request.param


def test_profile(generator_configuration: config.Configuration):
    profile.__main__(dict2args(configuration=generator_configuration.name, num_samples=1,
                               num_nodes=10))
