import numpy as np
import pytest
from simulation_based_graph_inference.graph import Graph
from simulation_based_graph_inference import generators
import torch as th


@pytest.fixture
def graph():
    graph = Graph()
    yield graph
    # Always convert to edge index to find any bugs in generators.
    graph.to_edge_index()


def test_generate_poisson_random_attachment(graph: Graph):
    rate = th.distributions.Gamma(4, 1).sample()
    generators.generate_poisson_random_attachment(100, rate, graph)
    assert graph.get_num_nodes() == 100


def test_seed():
    edge_indices = []
    for _ in range(2):
        generators.set_seed(42)
        graph = generators.generate_poisson_random_attachment(10, 3)
        edge_indices.append(graph.to_edge_index())
    np.testing.assert_array_equal(*edge_indices)
