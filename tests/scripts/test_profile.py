import pytest
from simulation_based_graph_inference.scripts import profile, util


@pytest.fixture(params=util.GENERATORS)
def generator(request):
    return request.param


def test_profile(generator: str):
    profile.__main__([generator, "--num_samples=1", "--num_nodes=10"])
