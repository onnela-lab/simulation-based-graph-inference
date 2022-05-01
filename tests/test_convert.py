import pytest
from simulation_based_graph_inference.graph import Graph
from simulation_based_graph_inference.convert import to_edge_index, to_networkx
import torch as th


def _get_graph_and_edges() -> tuple[Graph, set]:
    graph = Graph()
    graph.add_nodes([0, 1, 2, 3])
    edges = {(0, 1), (1, 2), (2, 3), (0, 3)}
    graph.add_edges(edges)
    return graph, edges


def test_to_edge_index():
    graph, edges = _get_graph_and_edges()
    edge_index = to_edge_index(graph)
    assert edge_index.shape == (2, 2 * graph.get_num_edges())
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
