from libcpp.iterator cimport back_inserter
from libcpp.utility cimport move
from libcpp.vector cimport vector
from .graph cimport count_t, node_t, Graph
from .libcpp.algorithm cimport sample
from .libcpp.random cimport mt19937, random_device, poisson_distribution

cdef random_device rd
cdef mt19937 random_engine = mt19937(rd())


def set_seed(seed: mt19937.result_type) -> None:
    """
    Set the `mt19937` random number generator seed.

    Args:
        seed: Seed value.
    """
    random_engine.seed(seed)


def generate_poisson_random_attachment(num_nodes: count_t, rate: double, graph: Graph = None) -> Graph:
    """
    Grow a graph with Poisson-distributed number of stubs for new nodes that are randomly attached
    to existing nodes.

    Args:
        num_nodes: Final number of nodes.
        rate: Poisson rate for generating stubs.
        graph: Seed graph to modify in place. If `None`, a new graph is created.

    Returns:
        graph: Generated graph with `num_nodes` nodes.
    """
    cdef:
        count_t degree
        vector[node_t] neighbors
        poisson_distribution[count_t] connection_distribution = poisson_distribution[count_t](rate)

    graph = graph or Graph()

    for node in range(graph.get_num_nodes(), num_nodes):
        # Sample the degree and obtain neighbors.
        degree = min(node, connection_distribution(random_engine))
        sample(graph.nodes.begin(), graph.nodes.end(), back_inserter(neighbors), degree, move(random_engine))
        # Add the node and connections.
        graph.add_node(node)
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)
        # Reset the neighbors.
        neighbors.clear()

    return graph
