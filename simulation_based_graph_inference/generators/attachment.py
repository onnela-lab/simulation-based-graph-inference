import networkx as nx
import numbers
import numpy as np
from scipy import special
import typing
from ..util import assert_interval, assert_normalized_nodel_labels


def random_attachment_graph(
        num_nodes: int, m: typing.Union[int, typing.Callable], add_isolated_nodes: bool = False,
        graph: nx.Graph = None, rng: np.random.Generator = None) -> nx.Graph:
    """
    Grow a graph with Poisson-distributed number of stubs for new nodes that are randomly attached
    to existing nodes.

    Args:
        num_nodes: Final number of nodes.
        m: Number of nodes to connect to or a callable that returns the number of nodes to connect
            to.
        add_isolated_nodes: Whether to add new nodes even if they are isolated.
        graph: Seed graph to modify in place. If `None`, a new graph is created.
        rng: Random number generator.

    Returns:
        graph: Generated graph with `num_nodes` nodes.

    .. plot::

        _plot_generated_graph(generators.random_attachment_graph, 4)
    """
    assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    if isinstance(m, numbers.Integral):
        assert_interval("m", m, 0, None, inclusive_low=False)
    elif not isinstance(m, typing.Callable):
        raise ValueError("m must be an integer or callable that returns an integer")
    rng = rng or np.random
    graph = graph or nx.Graph()
    assert_normalized_nodel_labels(graph)

    for node in range(len(graph), num_nodes):
        graph.add_node(node)
        # Sample the degree and obtain neighbors.
        degree = min(node, m if isinstance(m, numbers.Integral) else m())
        neighbors = rng.choice(node, int(degree), replace=False)
        graph.add_edges_from((node, neighbor) for neighbor in neighbors)
        if add_isolated_nodes:
            graph.add_node(node)

    return graph


def degree_attachment_graph(num_nodes: int, m: int, power: float, graph: nx.Graph = None,
                            rng: np.random.Generator = None) -> nx.Graph:
    r"""
    Grow a graph with power degree preferential attachment as described by
    `Krapivsky et al. (2000) <https://doi.org/10.1103/PhysRevLett.85.4629>`_. New nodes are
    connected to neighbors with probability :math:`\propto k^\alpha`, where :math:`k` is the degree
    and :math:`\alpha` is the power exponent.

    Args:
        num_nodes: Final number of nodes.
        m: Number of nodes to connect to.
        power: Power for degree preferential attachment.
        graph: Seed graph to modify in place. If `None`, a new graph is created.
        rng: Random number generator.

    Returns:
        graph: Generated graph with `num_nodes` nodes.
    """
    assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    assert_interval("m", m, 0, None, inclusive_low=False)
    assert_interval("power", power, 0, None)
    rng = rng or np.random
    if not graph:
        graph = nx.Graph()
        graph.add_edge(0, 1)
    if not len(graph.edges):
        raise ValueError("graph must have at least one edge")
    assert_normalized_nodel_labels(graph)

    degrees = [graph.degree[node] for node in sorted(graph)]
    while (node := len(graph)) < num_nodes:
        # Choose neighbors.
        log_proba = power * np.log(degrees)
        proba = special.softmax(log_proba)
        degree = min(m, node)
        neighbors = rng.choice(node, size=degree, p=proba, replace=False)

        # Add edges and update degrees.
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)
            degrees[neighbor] += 1
        degrees.append(degree)

    return graph


def rank_attachment_graph(num_nodes: int, m: int, power: float, graph: nx.Graph = None,
                          rng: np.random.Generator = None) -> nx.Graph:
    r"""
    Grow a graph with power rank preferential attachment as described by
    `Fortunato et al. (2006) <https://doi.org/10.1103/PhysRevLett.96.218701>`_. New nodes are
    connected to neighbors with probability :math:`\propto r^-\alpha`, where :math:`r` is their rank
    and :math:`\alpha` is the power exponent. In this implementation, the rank is simply the order
    of addition of nodes.

    Args:
        num_nodes: Final number of nodes.
        m: Number of nodes to connect to.
        power: Power for rank preferential attachment.
        graph: Seed graph to modify in place. If `None`, a new graph is created.
        rng: Random number generator.

    Returns:
        graph: Generated graph with `num_nodes` nodes.
    """
    assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    assert_interval("m", m, 0, None, inclusive_low=False)
    assert_interval("power", power, 0, None)
    rng = rng or np.random
    if not graph:
        graph = nx.Graph()
        graph.add_node(0)
    assert_normalized_nodel_labels(graph)

    ranks = [node + 1 for node in sorted(graph)]
    while (node := len(graph)) < num_nodes:
        # Choose neighbors.
        log_proba = - power * np.log(ranks)
        proba = special.softmax(log_proba)
        degree = min(m, node)
        neighbors = rng.choice(node, size=degree, p=proba, replace=False)

        # Add edges and add the rank of the new node.
        graph.add_edges_from((node, neighbor) for neighbor in neighbors)
        ranks.append(node + 1)

    return graph
