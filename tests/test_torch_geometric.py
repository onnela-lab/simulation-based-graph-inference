import pytest
from simulation_based_graph_inference import convert
from simulation_based_graph_inference import generators
from simulation_based_graph_inference.graph import Graph
import torch as th
import torch_geometric.nn


@pytest.fixture
def graph():
    return generators.generate_redirection(100, 4, .5)


@pytest.fixture
def adjacency(graph):
    return convert.to_adjacency(graph)


@pytest.fixture
def edge_index(graph):
    return convert.to_edge_index(graph)


def test_gin(graph: Graph, adjacency: th.Tensor, edge_index: th.Tensor):
    x = th.randn(graph.get_num_nodes(), 3)
    y = torch_geometric.nn.GINConv(lambda x: x)(x, edge_index)
    z = (adjacency + th.eye(adjacency.shape[0])) @ x
    th.testing.assert_allclose(y, z)


def test_gcn(graph: Graph, adjacency: th.Tensor, edge_index: th.Tensor):
    x = th.randn(graph.get_num_nodes(), 3)
    conv = torch_geometric.nn.GCNConv(3, 4)
    y = conv(x, edge_index)
    matrix = (adjacency + th.eye(adjacency.shape[0]))
    degree = matrix.sum(axis=0)
    matrix = matrix / (degree[:, None] * degree).sqrt()
    z = matrix @ x @ conv.lin.weight.T
    th.testing.assert_allclose(y, z)
