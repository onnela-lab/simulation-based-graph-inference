from simulation_based_graph_inference.graph import Graph
import pytest


def test_empty_graph():
    graph = Graph()
    assert graph.get_num_nodes() == 0
    assert graph.get_num_edges() == 0


def test_add_node():
    graph = Graph()
    graph.add_node(3)
    assert graph.get_num_nodes() == 1
    graph.add_node(7)
    assert graph.get_num_nodes() == 2
    graph.add_node(3)
    assert graph.get_num_nodes() == 2


def test_remove_missing_node():
    graph = Graph()
    with pytest.raises(IndexError):
        graph.remove_node(7)


def test_remove_node():
    graph = Graph()
    graph.add_nodes({0, 1, 2})
    graph.add_edges({(0, 1), (1, 2), (2, 0)})
    assert graph.get_num_nodes() == 3
    assert graph.get_num_edges() == 3

    graph.remove_node(0)
    assert graph.get_num_nodes() == 2
    assert graph.get_num_edges() == 1


def test_add_edge_missing_node():
    graph = Graph()
    with pytest.raises(IndexError):
        graph.add_edge(0, 1)


def test_add_edge_and_neighbors():
    graph = Graph()
    graph.add_nodes({0, 1, 2})
    graph.add_edges({(0, 1), (0, 2)})

    assert graph.get_neighbors(0) == {1, 2}
    assert graph.get_neighbors(1) == {0}
    assert graph.get_neighbors(2) == {0}


def test_neighbors_of_missing_node():
    graph = Graph()
    with pytest.raises(IndexError):
        graph.get_neighbors(0)


def test_to_edge_index():
    graph = Graph()
    graph.add_nodes([0, 1, 2, 3])
    edges = {(0, 1), (1, 2), (2, 3), (3, 0)}
    graph.add_edges(edges)
    edge_index = graph.to_edge_index()
    assert edge_index.shape == (2, 2 * graph.get_num_edges())
    # Check that directed edge indices have been created for all edges.
    edge_index_set = {(int(u), int(v)) for u, v in edge_index.T}
    directed_edges = edges | {(v, u) for u, v in edges}
    assert directed_edges == edge_index_set


def test_no_self_loops():
    graph = Graph()
    graph.add_node(0)
    with pytest.raises(RuntimeError):
        graph.add_edge(0, 0)
