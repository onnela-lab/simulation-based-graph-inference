# cython: cdivision = True
from cython.operator cimport dereference
from libcpp.iterator cimport back_inserter
from libcpp.utility cimport move
from libcpp.vector cimport vector as vector_t
from .graph cimport count_t, node_t, node_set_t, Graph
from .libcpp.algorithm cimport sample
from .libcpp.random cimport mt19937, random_device, poisson_distribution, bernoulli_distribution, \
    binomial_distribution
from .util import assert_interval

__PYI_HEADER = """
from .graph import Graph
"""
__PYI_TYPEDEFS = {
    "count_t": "int",
    "double": "float",
    "Graph": "Graph",
    "mt19937": {
        "result_type": "int",
    },
}

cdef random_device rd
cdef mt19937 random_engine = mt19937(rd())


def set_seed(seed: mt19937.result_type) -> None:
    """
    Set the `mt19937` random number generator seed.

    Args:
        seed: Seed value.
    """
    random_engine.seed(seed)


def generate_poisson_random_attachment(num_nodes: count_t, rate: double, graph: Graph = None) \
        -> Graph:
    """
    Grow a graph with Poisson-distributed number of stubs for new nodes that are randomly attached
    to existing nodes.

    Args:
        num_nodes: Final number of nodes.
        rate: Poisson rate for generating stubs.
        graph: Seed graph to modify in place. If `None`, a new graph is created.

    Returns:
        graph: Generated graph with `num_nodes` nodes.

    .. plot::

        _plot_generated_graph(generators.generate_poisson_random_attachment, 4)
    """
    cdef:
        count_t degree
        vector_t[node_t] neighbors
        poisson_distribution[count_t] connection_distribution = poisson_distribution[count_t](rate)

    assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    assert_interval("rate", rate, 0, None, inclusive_low=False)
    graph = graph or Graph()

    for node in range(graph.get_num_nodes(), num_nodes):
        # Sample the degree and obtain neighbors.
        degree = min(node, connection_distribution(random_engine))
        sample(graph.nodes.begin(), graph.nodes.end(), back_inserter(neighbors), degree,
               move(random_engine))
        # Add the node and connections.
        graph.add_node(node)
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)
        # Reset the neighbors.
        neighbors.clear()

    return graph


def generate_duplication_mutation_complementation(num_nodes: count_t, interaction_proba: double,
        divergence_proba: double, graph: Graph = None) -> Graph:
    """
    Generate a protein interaction graph as described by
    `Vazquez et al. (2003) <https://doi.org/10.1159/000067642>`_.

    Args:
        num_nodes: Final number of nodes.
        interaction_proba: Probability to connect the duplicated node to the original node.
        divergence_proba: Probability that one of the connections between each neighbors and either
            the duplicated or original node is removed.
        graph: Seed graph to modify in place. If `None`, a new graph is created.

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

       _plot_generated_graph(generators.generate_duplication_mutation_complementation, 0.5, 0.2)
    """
    cdef:
        node_t source
        bernoulli_distribution dist_original = bernoulli_distribution(0.5)
        bernoulli_distribution dist_interaction = bernoulli_distribution(interaction_proba)
        bernoulli_distribution dist_divergence = bernoulli_distribution(divergence_proba)

    assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    assert_interval("divergence_proba", divergence_proba, 0, 1)
    assert_interval("interaction_proba", interaction_proba, 0, 1)
    graph = graph or Graph()
    # Ensure there is at least one node in the graph.
    if not graph.get_num_nodes():
        graph.add_node(0)

    for node in range(graph.get_num_nodes(), num_nodes):
        # Pick one of the nodes and duplicate it.
        sample(graph.nodes.begin(), graph.nodes.end(), &source, 1, move(random_engine))
        graph.add_node(node)
        # Create or remove connections with neighbors.
        for neighbor in dereference(graph._get_neighbors_ptr(source)):
            if dist_original(random_engine):
                if dist_divergence(random_engine):
                    graph.remove_edge(source, neighbor)
            else:
                if not dist_divergence(random_engine):
                    graph.add_edge(node, neighbor)
        # Create a connection between the nodes.
        if dist_interaction(random_engine):
            graph.add_edge(source, node)

    return graph


def generate_duplication_mutation_random(num_nodes: count_t, mutation_proba: double,
        deletion_proba: double, graph: Graph = None) -> Graph:
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

       _plot_generated_graph(generators.generate_duplication_mutation_random, 0.5, 0.2)
    """
    cdef:
        node_t source, node
        count_t num_extra_connections
        binomial_distribution[count_t] dist_num_extra_connections
        bernoulli_distribution dist_delete = bernoulli_distribution(deletion_proba)
        vector_t[node_t] neighbors

    assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    assert_interval("mutation_proba", mutation_proba, 0, 1)
    assert_interval("deletion_proba", deletion_proba, 0, 1)
    graph = graph or Graph()

    # Ensure there is at least one node in the graph.
    if not graph.get_num_nodes():
        graph.add_node(0)

    for node in range(graph.get_num_nodes(), num_nodes):
        # First pick the additional neighbors.
        dist_num_extra_connections = binomial_distribution[count_t](node, mutation_proba / node)
        num_extra_connections = dist_num_extra_connections(random_engine)
        sample(graph.nodes.begin(), graph.nodes.end(), back_inserter(neighbors),
               num_extra_connections, move(random_engine))

        # Pick one of the nodes and duplicate it.
        sample(graph.nodes.begin(), graph.nodes.end(), &source, 1, move(random_engine))
        graph.add_node(node)

        # Create connections to neighbors of source node.
        for neighbor in dereference(graph._get_neighbors_ptr(source)):
            if not dist_delete(random_engine):
                graph.add_edge(node, neighbor)

        # Create connections to random nodes.
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)
        neighbors.clear()

    return graph


def generate_redirection(num_nodes: count_t, num_connections: count_t, redirection_proba: double,
                         graph: Graph = None) -> Graph:
    """
    Generate a graph by random attachment with probabilistic redirection as described by
    `Krapivsky et al. (2001) <https://doi.org/10.1103/PhysRevE.63.066123>`_.

    Args:
        num_nodes: Final number of nodes.
        redirection_proba: Probability that connections are redirected to a neighbor.
        graph: Seed graph to modify in place. If `None`, a new graph is created.

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

       _plot_generated_graph(generators.generate_redirection, 3, 0.5)
    """
    cdef:
        int i
        node_t node, neighbor
        node_set_t* neighbors_ptr
        vector_t[node_t] neighbors
        bernoulli_distribution dist_redirect = bernoulli_distribution(redirection_proba)

    assert_interval("num_nodes", num_nodes, 0, None, inclusive_low=False)
    assert_interval("num_connections", num_connections, 0, None, inclusive_low=False)
    assert_interval("redirection_proba", redirection_proba, 0, 1)
    graph = graph or Graph()

    for node in range(graph.get_num_nodes(), num_nodes):
        # Sample new neighbors.
        sample(graph.nodes.begin(), graph.nodes.end(), back_inserter(neighbors),
               min(num_connections, node), move(random_engine))

        # Redirect for each neighbor with some probability.
        i = 0
        for neighbor in neighbors:
            if dist_redirect(random_engine):
                neighbors_ptr = graph._get_neighbors_ptr(neighbor)
                sample(neighbors_ptr.begin(), neighbors_ptr.end(), &neighbor, 1,
                       move(random_engine))
                neighbors[i] = neighbor
            i += 1

        # Add the node and connect neighbors.
        graph.add_node(node)
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)
        neighbors.clear()

    return graph
