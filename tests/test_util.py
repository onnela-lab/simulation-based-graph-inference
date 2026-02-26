import networkx as nx
import numpy as np
import pytest
from simulation_based_graph_inference import generators
from simulation_based_graph_inference import util
import torch as th
from torch_geometric.utils.convert import from_networkx


def test_plot_generated_graph():
    util._plot_generated_graph(generators.random_attachment_graph, 4)


@pytest.mark.parametrize(
    "fail, args",
    [
        (False, (0.5, 0, 1)),  # General check.
        (True, (-1, 1.1, 2)),  # Outside to the left.
        (False, (0, 0, 1)),  # Left bound inclusive.
        (True, (0, 0, 1, False)),  # Left bound exclusive.
        (False, (-1000, None, 1)),  # Unbounded left.
        (True, (2, None, 1)),  # Unbounded left, outside to the right.
        (True, (2.5, 1.1, 2)),  # Outside to the right.
        (False, (1, 0, 1)),  # Right bound inclusive.
        (True, (1, 0, 1, True, False)),  # Right bound inclusive.
        (False, (1000, 10, None)),  # Unbounded right.
        (True, (3, 10, None)),  # Unbounded right, outside to the left.
    ],
)
def test_assert_interval(fail, args):
    if fail:
        with pytest.raises(ValueError):
            util.assert_interval("var", *args)
    else:
        util.assert_interval("var", *args)


def _get_graph_and_edges() -> tuple[nx.Graph, set]:
    nodes = [0, 1, 2, 3]
    edges = {(0, 1), (1, 2), (2, 3), (0, 3)}
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph, edges


def test_to_edge_index():
    graph, edges = _get_graph_and_edges()
    edge_index = util.to_edge_index(graph)
    assert edge_index.shape == (2, 2 * len(graph))
    # Check that directed edge indices have been created for all edges.
    edge_index_set = {(int(u), int(v)) for u, v in edge_index.T}
    directed_edges = edges | {(v, u) for u, v in edges}
    assert directed_edges == edge_index_set


def test_to_edge_index_invalid_shape():
    graph, _ = _get_graph_and_edges()
    edge_index = th.zeros(1)
    with pytest.raises(ValueError):
        util.to_edge_index(graph, edge_index)


def test_to_edge_index_loops():
    graph = nx.Graph()
    graph.add_edge(0, 0)
    with pytest.raises(ValueError):
        util.to_edge_index(graph)


@pytest.mark.parametrize("edge", [(0, 100_000), (100_000, 0)])
def test_invalid_dtype(edge):
    graph = nx.Graph()
    graph.add_edge(*edge)
    with pytest.raises(RuntimeError):
        util.to_edge_index(graph, dtype=th.int16)


def test_assert_normalized_node_labels():
    graph = nx.Graph()
    util.assert_normalized_nodel_labels(graph)

    graph.add_node(0)
    util.assert_normalized_nodel_labels(graph)

    graph.add_nodes_from({1, 2})
    util.assert_normalized_nodel_labels(graph)

    graph.add_node(7)
    with pytest.raises(ValueError):
        util.assert_normalized_nodel_labels(graph)
    graph.remove_node(7)

    graph.remove_node(0)
    with pytest.raises(ValueError):
        util.assert_normalized_nodel_labels(graph)


def test_randint_invalid():
    with pytest.raises(TypeError):
        util.randint(None, 4)


@pytest.mark.parametrize("size", [None, 3, (5, 7)])
def test_random_sequence(size):
    rng1 = np.random.RandomState(0)
    rng2 = np.random.RandomState(0)
    sequence = util.random_sequence(rng1.normal, 3, 2, size=size, batch_size=10)

    for _ in range(1000):
        np.testing.assert_allclose(rng2.normal(3, 2, size=size), next(sequence))


@pytest.mark.parametrize("directed", [False, True])
def test_clustering_coefficient(directed: bool) -> None:
    if directed:
        pytest.xfail("directed doesn't seem to work; not a problem for our models")
    graph = nx.erdos_renyi_graph(100, 0.1, directed=directed)
    clustering = nx.clustering(graph)
    assert isinstance(clustering, dict)
    expected = th.as_tensor([value for _, value in sorted(clustering.items())])
    data = from_networkx(graph)
    actual = util.clustering_coefficient(data.edge_index, data.num_nodes)
    np.testing.assert_allclose(actual, expected)
