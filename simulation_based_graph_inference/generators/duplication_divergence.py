import networkx as nx
import numpy as np
from ..util import assert_interval, assert_normalized_nodel_labels, randint


def duplication_complementation(
        num_nodes: int, interaction_proba: float, divergence_proba: float, graph: nx.Graph = None,
        rng: np.random.Generator = None) -> nx.Graph:
    """
    Generate a protein interaction graph as described by
    `Vazquez et al. (2003) <https://doi.org/10.1159/000067642>`_.

    Args:
        num_nodes: Final number of nodes.
        interaction_proba: Probability to connect the duplicated node to the original node.
        divergence_proba: Probability that one of the connections between each neighbors and either
            the duplicated or original node is removed.
        graph: Seed graph to modify in place. If `None`, a new graph is created.
        rng: Random number generator.

    Returns:
        graph: Generated graph with `num_nodes` nodes.

    The growth process proceeds in three stages at each step:

    1. A node :math:`i` is chosen at random and duplicated to obtain node :math:`i'`, including all
       of its connections.
    2. With probability `interaction_proba` (often denoted :math:`p`) the two nodes :math:`i` and
       :math:`i'` are connected.
    3. For each of the neighbors :math:`j`, we choose one of the edges :math:`(i, j)` and
       :math:`(i', j)` and remove it with probability `divergence_proba` (often denoted :math:`q`).

    Note:
        We use an equivalent but more efficient growth process here. For each of the neighbors
        :math:`j`, we chose either the original node or the duplicated node. If we chose the
        original node, we remove the edge :math:`(i, j)` with probability `divergence_proba`. If we
        chose the duplicated node, we create the edge :math:`(i', j)` with probability
        `1 - divergence proba`.

    .. plot::

       _plot_generated_graph(generators.duplication_complementation, 0.5, 0.2)
    """
    assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    assert_interval("divergence_proba", divergence_proba, 0, 1)
    assert_interval("interaction_proba", interaction_proba, 0, 1)
    rng = rng or np.random
    graph = graph or nx.Graph()
    assert_normalized_nodel_labels(graph)
    # Ensure there is at least one node in the graph.
    if not len(graph):
        graph.add_node(0)

    for node in range(len(graph), num_nodes):
        # Pick one of the nodes and duplicate it.
        source = randint(rng, node)
        graph.add_node(node)
        # Create or remove connections with neighbors.
        for neighbor in list(graph.neighbors(source)):
            if rng.binomial(1, divergence_proba):
                if rng.binomial(1, 0.5):
                    graph.remove_edge(source, neighbor)
                    graph.add_edge(node, neighbor)
            else:
                graph.add_edge(node, neighbor)
        # Create a connection between the nodes.
        if rng.binomial(1, interaction_proba):
            graph.add_edge(source, node)

    return graph


def duplication_mutation(
        num_nodes: int, mutation_proba: float, deletion_proba: float, graph: nx.Graph = None,
        rng: np.random.Generator = None) -> nx.Graph:
    r"""
    Generate a protein interaction graph as described by
    `Sole et al. (2002) <https://doi.org/10.1142/S021952590200047X>`_.

    Args:
        num_nodes: Final number of nodes.
        mutation_proba: Parameter such that random connections between the duplicated node and all
            other nodes in the network are created with probability
            `mutation_proba / current_num_nodes`.
        deletion_proba: Probability that, for each neighbor, the connection with the duplicated node
            is removed.
        graph: Seed graph to modify in place. If `None`, a new graph is created.
        rng: Random number generator.

    Returns:
        graph: Generated graph with `num_nodes` nodes.

    The growth process proceeds in three stages at each step:

    1. A node :math:`i` is chosen at random and duplicated to obtain node :math:`i'`, including all
       of its connections.
    2. For each neighbor :math:`j`, the connection to the new node :math:`i'` is removed with
       probability `deletion_proba` (often denoted :math:`\delta`).
    3. New connections from the new node :math:`i'` to any nodes in the network are created with
       probability `mutation_proba / num_nodes` (often denoted :math:`\beta`).

    Note:
        We use an equivalent but more efficient growth process here. In the second step, for each of
        the neighbors :math:`j`, we create an edge to :math:`i'` with probability
        `1 - deletion_proba`. In the third step, we sample the number of additional edges `extra`
        from a binomial random variable with `num_nodes` trials and probability
        `mutation_proba / num_nodes`. We then sample connect the new node to `extra` distinct
        existing nodes.

    .. plot::

       _plot_generated_graph(generators.duplication_mutation, 0.5, 0.2)
    """
    assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    assert_interval("mutation_proba", mutation_proba, 0, 1)
    assert_interval("deletion_proba", deletion_proba, 0, 1)
    rng = rng or np.random
    graph = graph or nx.Graph()
    assert_normalized_nodel_labels(graph)

    # Ensure there is at least one node in the graph.
    if not len(graph):
        graph.add_node(0)

    while (node := len(graph)) < num_nodes:
        # First pick the additional neighbors, we sample one additional one as the seed.
        num_extra_connections = rng.binomial(node, mutation_proba / node)
        source, *random_neighbors = rng.choice(node, num_extra_connections + 1)

        # Create connections to neighbors of source node.
        for neighbor in graph.neighbors(source):
            if not rng.binomial(1, deletion_proba):
                graph.add_edge(node, neighbor)

        graph.add_edges_from((node, neighbor) for neighbor in random_neighbors)

    return graph
