import networkx as nx
import numpy as np
from ..util import assert_interval, assert_normalized_nodel_labels, randint


def copy_graph(
    num_nodes: int,
    copy_proba: float,
    graph: nx.Graph = None,
    rng: np.random.Generator = None,
) -> nx.Graph:
    """
    Generate a graph by random attachment with probabilistic copying as described by
    `Lambiotte et al. (2016) <https://doi.org/10.1103/PhysRevLett.117.218301>`_.

    Args:
        num_nodes: Final number of nodes.
        copy_proba: Probability that new nodes are connected to each neighbor of the seed node.
        graph: Seed graph to modify in place. If `None`, a new graph is created.
        rng: Random number generator.

    Returns:
        graph: Generated graph with `num_nodes` nodes.

    .. plot::

       _plot_generated_graph(generators.copy_graph, 0.5)
    """
    assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    copy_proba = assert_interval("copy_proba", copy_proba, 0, 1, dtype=float)
    rng = rng or np.random
    graph = graph or nx.Graph()
    if not len(graph):
        graph.add_node(0)
    assert_normalized_nodel_labels(graph)

    for node in range(graph.number_of_nodes(), num_nodes):
        seed = randint(rng, node)
        edges = [
            (node, neighbor)
            for neighbor in graph.neighbors(seed)
            if rng.binomial(1, copy_proba)
        ]
        edges.append((node, seed))
        graph.add_edges_from(edges)

    return graph
