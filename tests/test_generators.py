import numpy as np
import pytest
from simulation_based_graph_inference.convert import to_edge_index
from simulation_based_graph_inference.graph import Graph
from simulation_based_graph_inference import generators
import torch as th


@pytest.fixture
def num_nodes():
    return 100


@pytest.fixture
def graph(num_nodes):
    graph = Graph()
    yield graph
    assert graph.get_num_nodes() == num_nodes
    # Always convert to edge index to find any bugs in generators.
    to_edge_index(graph)


def test_generate_poisson_random_attachment(graph: Graph, num_nodes: int):
    rate = th.distributions.Gamma(4, 1).sample()
    generators.generate_poisson_random_attachment(num_nodes, rate, graph)


def test_generate_duplication_complementation(graph: Graph, num_nodes: int):
    interaction_proba, divergence_proba = th.distributions.Uniform(0, 1).sample([2])
    generators.generate_duplication_mutation_complementation(num_nodes, interaction_proba,
                                                             divergence_proba, graph)


def test_generate_duplication_random(graph: Graph, num_nodes: int):
    mutation_proba, deletion_proba = th.distributions.Uniform(0, 1).sample([2])
    generators.generate_duplication_mutation_random(num_nodes, mutation_proba, deletion_proba,
                                                    graph)


def test_seed():
    edge_indices = []
    for _ in range(2):
        generators.set_seed(42)
        graph = generators.generate_poisson_random_attachment(10, 3)
        edge_indices.append(to_edge_index(graph))
    np.testing.assert_array_equal(*edge_indices)
