from simulation_based_graph_inference.graph import Graph, normalize_node_labels, extract_subgraph, \
    extract_neighborhood, extract_neighborhood_subgraph
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


def test_normalize_node_labels():
    offset = 7
    nodes = {0, 1, 2}
    edges = {(0, 1), (1, 2)}
    graph = Graph()
    graph.add_nodes({offset + i for i in nodes})
    graph.add_edges({(u + offset, v + offset) for u, v in edges})

    other = normalize_node_labels(graph)
    assert other.nodes == nodes
    assert other.edges == edges


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


def test_extract_subgraph():
    # Create a five-node ring graph.
    graph = Graph()
    graph.add_nodes({0, 1, 2, 3, 4})
    graph.add_edges({(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)})

    # Get a subgraph with the first three nodes which should be a line.
    nodes = {0, 1, 2}
    subgraph = extract_subgraph(graph, nodes)
    assert subgraph.nodes == nodes
    assert subgraph.edges == {(0, 1), (1, 2)}


def _line_graph(num_nodes: int) -> Graph:
    graph = Graph()
    graph.add_nodes(range(num_nodes))
    graph.add_edges({(i, i + 1) for i in range(num_nodes - 1)})
    return graph


@pytest.mark.parametrize('depth', [0, 1, 2, 3])
def test_extract_neighborhood(depth):
    num_nodes = 11
    graph = _line_graph(num_nodes)

    # Extract the neighborhood starting from the middle.
    seed = num_nodes // 2
    neighborhood = extract_neighborhood(graph, {seed}, depth)
    expected_neighborhood = set(range(seed - depth, seed + depth + 1))
    assert neighborhood == expected_neighborhood


@pytest.mark.parametrize('depth', [0, 1, 2, 3])
def test_extract_neighborhood_multiseed(depth):
    num_nodes = 11
    graph = _line_graph(num_nodes)

    # Seed at the left and right.
    neighborhood = extract_neighborhood(graph, {0, num_nodes - 1}, depth)
    expected_neighborhood = set(range(1 + depth)) | {num_nodes - 1 - i for i in range(1 + depth)}
    assert neighborhood == expected_neighborhood


@pytest.mark.parametrize('depth', [0, 1, 2, 3])
def test_extract_neighborhood_subgraph(depth):
    num_nodes = 11
    graph = _line_graph(num_nodes)

    # Extract the subgraph from the middle.
    seed = num_nodes // 2
    subgraph = extract_neighborhood_subgraph(graph, {seed}, depth)
    expected_neighborhood = set(range(seed - depth, seed + depth + 1))
    assert set(subgraph) == expected_neighborhood
