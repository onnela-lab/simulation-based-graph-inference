from doit_interface import dict2args
import pytest
from simulation_based_graph_inference.scripts import profile, util


@pytest.fixture(params=util.GENERATOR_CONFIGURATIONS)
def generator_configuration(request):
    return request.param


def test_profile(generator_configuration: str):
    profile.__main__(dict2args(configuration=generator_configuration, num_samples=1, num_nodes=10))
