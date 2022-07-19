import networkx as nx
import numpy as np
from ..util import assert_interval, assert_normalized_nodel_labels, randint


def _sample_by_degree(rng: np.random.Generator, edges: list) -> int:
    """
    Sample a node proportional to its degree.
    """
    u, v = edges[randint(rng, len(edges))]
    return u if rng.binomial(1, 0.5) else v


def web_graph(num_nodes: int, proba_new: float, proba_uniform_new: float, proba_uniform_old1: float,
              dist_degree_new: np.ndarray, proba_uniform_old2: float = None,
              dist_degree_old: np.ndarray = None, graph: nx.Graph = None, rng:
              np.random.Generator = None) -> nx.Graph:
    """
    Generate a web graph according to the undirected model of
    `Cooper et al. (2003) <https://doi.org/10.1002/rsa.10084>`_.

    Args:
        num_nodes: Final number of nodes.
        proba_new: Probability that a new node is added at each step.
        proba_uniform_new: Probability that neighbors of new nodes are chosen uniformly at random
            (as opposed to proportional to degree).
        proba_uniform_old1: Probability that old nodes are chosen uniformly at random (as opposed to
            proportional to degree).
        dist_degree_new: Distribution of number of edges added to new nodes.
        proba_uniform_old2: Probability that neighbors of old nodes are chosen uniformly at random
            (as opposed to proportional to degree). Defaults to `proba_uniform_old1`.
        dist_degree_old: Distribution of number of edges added to old nodes. Defaults to
            `dist_degree_new`.
        graph: Seed graph to modify in place. If `None`, a new graph is created.
        rng: Random number generator.

    Returns:
        graph: Generated graph with `num_nodes` nodes.
    """
    # Normalize all parameter values.
    num_nodes = assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    proba_new = assert_interval("proba_new", proba_new, 0, 1, inclusive_low=False, dtype=float)
    proba_uniform_new = assert_interval("proba_uniform_new", proba_uniform_new, 0, 1, dtype=float)
    proba_uniform_old1 = assert_interval("proba_uniform_old1", proba_uniform_old1, 0, 1,
                                         dtype=float)
    proba_uniform_old2 = assert_interval(
        "proba_uniform_old2", proba_uniform_old1 if proba_uniform_old2 is None else
        proba_uniform_old2, 0, 1, dtype=float)
    if dist_degree_old is None:
        dist_degree_old = dist_degree_new
    rng = rng or np.random
    graph = assert_normalized_nodel_labels(graph)
    if not len(graph):
        graph.add_node(0)

    # We keep track of the edgelist for easier sampling proportional to degree.
    edges = list(graph.edges)

    while len(graph) < num_nodes:
        # We add a new node if the random sample tells us to or if there is only one node in the
        # graph because we are not allowed self-loops.
        if rng.binomial(1, proba_new) or len(graph) < 2:
            node = len(graph)
            dist_degree = dist_degree_new
            proba_uniform = proba_uniform_new
        else:
            if rng.binomial(1, proba_uniform_old1):
                node = randint(rng, len(graph))
            else:
                node = _sample_by_degree(rng, edges)
            dist_degree = dist_degree_new
            proba_uniform = proba_uniform_old2

        # Sample degree and connect the node to sampled neighbors.
        degree = rng.choice(len(dist_degree), p=dist_degree)
        assert degree > 0, "must create at least one edge at each step"
        for _ in range(degree):
            while True:
                if rng.binomial(1, 1 - proba_uniform) and edges:
                    neighbor = _sample_by_degree(rng, edges)
                else:
                    neighbor = randint(rng, len(graph))
                # Ensure we don't create any self-loops.
                if neighbor != node:
                    break
            graph.add_edge(node, neighbor)
            edges.append((node, neighbor))

        # Add the node. This is a no-op for existing nodes. This *should* not be necessary because
        # each node adds at least one edge, but let's be sure.
        graph.add_node(node)

    return graph
