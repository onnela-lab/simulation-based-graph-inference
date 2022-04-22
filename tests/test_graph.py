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


def test_no_self_loops():
    graph = Graph()
    graph.add_node(0)
    with pytest.raises(ValueError):
        graph.add_edge(0, 0)


def test_copy():
    graph = Graph()
    graph.add_nodes({0, 1, 2})
    graph.add_edges({(0, 1), (0, 2)})

    # Check that nodes aren't affected.
    other = Graph(graph)
    graph.remove_node(1)
    assert 1 in other
    other.remove_node(2)
    assert 2 in graph

    # Check that edges aren't affected.
    other = Graph(graph)
    graph.remove_edge(0, 2)
    assert (0, 2) in other


def test_repr():
    graph = Graph()
    graph.add_nodes({0, 1, 2})
    graph.add_edges({(0, 1), (1, 2)})
    assert repr(graph) == "Graph(num_nodes=3, num_edges=2)"


def test_iter():
    graph = Graph()
    nodes = {0, 1, 2}
    graph.add_nodes(nodes)
    assert set(graph) == nodes
