import networkx as nx
import numpy as np
from ..util import assert_interval, assert_normalized_nodel_labels


def poisson_random_attachment_graph(num_nodes: int, rate: float, graph: nx.Graph = None,
                                    rng: np.random.Generator = None) -> nx.Graph:
    """
    Grow a graph with Poisson-distributed number of stubs for new nodes that are randomly attached
    to existing nodes.

    Args:
        num_nodes: Final number of nodes.
        rate: Poisson rate for generating stubs.
        graph: Seed graph to modify in place. If `None`, a new graph is created.
        rng: Random number generator.

    Returns:
        graph: Generated graph with `num_nodes` nodes.

    .. plot::

        _plot_generated_graph(generators.poisson_random_attachment_graph, 4)
    """
    assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    assert_interval("rate", rate, 0, None, inclusive_low=False)
    rng = rng or np.random
    graph = graph or nx.Graph()
    assert_normalized_nodel_labels(graph)

    for node in range(len(graph), num_nodes):
        graph.add_node(node)
        # Sample the degree and obtain neighbors.
        degree = min(node, rng.poisson(rate))
        neighbors = rng.choice(node, degree)
        graph.add_edges_from((node, neighbor) for neighbor in neighbors)

    return graph
