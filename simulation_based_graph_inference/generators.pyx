from .graph cimport Graph


def generate_poisson_random_attachment(num_nodes: size_t, rate: double, graph: Graph = None):
    if graph is None:
        graph = Graph()
    _generate_poisson_random_attachment(graph._this, num_nodes, rate)
    return graph
