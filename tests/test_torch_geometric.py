import networkx as nx
import pytest
from simulation_based_graph_inference import generators
from simulation_based_graph_inference import util
import torch as th
import torch_geometric.nn


@pytest.fixture
def graph() -> nx.Graph:
    return generators.redirection_graph(100, 4, 0.5)


@pytest.fixture
def adjacency(graph: nx.Graph):
    return th.as_tensor(nx.to_numpy_array(graph))


@pytest.fixture
def edge_index(graph: nx.Graph):
    return util.to_edge_index(graph)


def test_gin(graph: nx.Graph, adjacency: th.Tensor, edge_index: th.Tensor):
    x = th.randn(len(graph), 3)
    y = torch_geometric.nn.GINConv(lambda x: x)(x, edge_index)
    z = (adjacency + th.eye(adjacency.shape[0])) @ x
    th.testing.assert_close(y, z)


def test_gcn(graph: nx.Graph, adjacency: th.Tensor, edge_index: th.Tensor):
    x = th.randn(len(graph), 3)
    conv = torch_geometric.nn.GCNConv(3, 4)
    y = conv(x, edge_index)
    matrix = adjacency + th.eye(adjacency.shape[0])
    degree = matrix.sum(axis=0)
    matrix = matrix / (degree[:, None] * degree).sqrt()
    z = matrix @ x @ conv.lin.weight.T
    th.testing.assert_close(y, z)
