import networkx as nx
import numpy as np
import typing
from ..util import assert_interval, assert_normalized_nodel_labels, randint


def redirection_graph(
    num_nodes: int,
    max_num_connections: int,
    redirection_proba: float,
    graph: typing.Optional[nx.Graph] = None,
    rng: typing.Optional[np.random.Generator] = None,
) -> nx.Graph:
    """
    Generate a graph by random attachment with probabilistic redirection as described by
    `Krapivsky et al. (2001) <https://doi.org/10.1103/PhysRevE.63.066123>`_.

    Args:
        num_nodes: Final number of nodes.
        max_num_connections: Maximum number of connections added to new nodes.
        redirection_proba: Probability that connections are redirected to a neighbor.
        graph: Seed graph to modify in place. If `None`, a new graph is created.
        rng: Random number generator.

    Returns:
        graph: Generated graph with `num_nodes` nodes.

    The growth process proceeds in three stages at each step:

    1. We sample `min(num_connections, num_current_nodes)` nodes from the graph as potential
       neighbors.
    2. For each potential neighbor, we replace it by one of its randomly chosen neighbors with
       proability `redirection_proba`.
    3. We create a new node and connect it to the chosen neighbors.

    Note:
        The model differs slightly from the reference description in two ways: first, connections
        are undirected. Second, we may create more than one edge per new node.

        If `redirection_proba` is zero, the model reduces to a random growth model. If
        `redirection_proba` is one, the model is equivalent to a linear preferential attachment
        model.

    .. plot::

       _plot_generated_graph(generators.redirection_graph, 3, 0.5)
    """
    assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    max_num_connections = assert_interval(
        "max_num_connections",
        max_num_connections,
        0,
        None,
        inclusive_low=False,
        dtype=int,
    )
    redirection_proba = assert_interval(
        "redirection_proba", redirection_proba, 0, 1, dtype=float
    )
    rng = rng or np.random.default_rng()
    graph = graph or nx.Graph()
    if not len(graph):
        graph.add_node(0)
    assert_normalized_nodel_labels(graph)

    for node in range(len(graph), num_nodes):
        # Sample new neighbors.
        candidates = rng.choice(node, min(max_num_connections, node), replace=False)

        # Redirect for each neighbor with some probability.
        neighbors = []
        for candidate in candidates:
            if rng.binomial(1, redirection_proba):
                candidate_neighbors = list(graph.neighbors(candidate))
                # We can only redirect if there are neighbors.
                if len(candidate_neighbors):
                    candidate = rng.choice(candidate_neighbors)
            neighbors.append(candidate)

        graph.add_edges_from((node, candidate) for candidate in neighbors)

    return graph


def surfer_graph(
    num_nodes: int,
    hop_proba: float,
    graph: typing.Optional[nx.Graph] = None,
    rng: typing.Optional[np.random.Generator] = None,
) -> nx.Graph:
    """
    Generate a random surfer graph as described by
    `Vazquez (2003) <https://doi.org/10.1103/PhysRevE.67.056104>`__.

    Args:
        num_nodes: Final number of nodes.
        hop_proba: Probability that the walker hops to a neighbor.
        graph: Seed graph to modify in place. If `None`, a new graph is created.
        rng: Random number generator.

    Returns:
        graph: Generated graph with `num_nodes` nodes.
    """
    assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    hop_proba = assert_interval("hop_proba", hop_proba, 0, 1, dtype=float)
    rng = rng or np.random.default_rng()
    graph = graph or nx.Graph()
    if not len(graph):
        graph.add_node(0)
    assert_normalized_nodel_labels(graph)

    for source in range(len(graph), num_nodes):
        seed = randint(rng, source)
        targets = {seed}
        while rng.binomial(1, hop_proba):
            candidates = list(graph.neighbors(seed))
            if not candidates:
                break
            seed = rng.choice(candidates)
            if seed in targets:
                break
            targets.add(seed)
        graph.add_edges_from((source, target) for target in targets)

    return graph
