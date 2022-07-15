import networkx as nx
import numpy as np
import pytest
from simulation_based_graph_inference.graph import Graph
from simulation_based_graph_inference.convert import to_edge_index, to_networkx, to_adjacency
import torch as th


def _get_graph_and_edges(networkx: bool = False) -> tuple[Graph, set]:
    nodes = [0, 1, 2, 3]
    edges = {(0, 1), (1, 2), (2, 3), (0, 3)}
    if networkx:
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
    else:
        graph = Graph()
        graph.add_nodes(nodes)
        graph.add_edges(edges)
    return graph, edges


@pytest.mark.parametrize("networkx", [False, True])
def test_to_edge_index(networkx: bool):
    graph, edges = _get_graph_and_edges(networkx)
    edge_index = to_edge_index(graph)
    num_edges = graph.number_of_edges() if networkx else graph.get_num_edges()
    assert edge_index.shape == (2, 2 * num_edges)
    # Check that directed edge indices have been created for all edges.
    edge_index_set = {(int(u), int(v)) for u, v in edge_index.T}
    directed_edges = edges | {(v, u) for u, v in edges}
    assert directed_edges == edge_index_set


def test_to_edge_index_invalid_shape():
    graph, edges = _get_graph_and_edges()
    edge_index = th.zeros(1)
    with pytest.raises(ValueError):
        to_edge_index(graph, edge_index)


def test_to_networkx():
    graph, edges = _get_graph_and_edges()
    nxgraph = to_networkx(graph)
    assert nxgraph.number_of_nodes() == graph.get_num_nodes()
    assert edges == set(nxgraph.edges)


def test_to_adjacency():
    graph, _ = _get_graph_and_edges()
    adjacency = to_adjacency(graph)
    nxgraph = to_networkx(graph)
    nxadjacency = nx.to_numpy_array(nxgraph)
    np.testing.assert_array_equal(adjacency, nxadjacency)

    with pytest.raises(ValueError):
        to_adjacency(graph, th.zeros(1))


@pytest.mark.parametrize("edge", [(0, 100_000), (100_000, 0)])
def test_invalid_dtype(edge):
    graph = Graph()
    graph.add_nodes(edge)
    graph.add_edge(*edge)
    with pytest.raises(ValueError):
        to_edge_index(graph, dtype=th.int16)
