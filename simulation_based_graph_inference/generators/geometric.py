import numpy as np
from scipy.spatial.distance import cdist
import typing
from ..graph import Graph


def geometric(num_nodes: int, kernel: typing.Callable = None, num_dims: int = 2,
              graph: Graph = None, **kwargs) -> Graph:
    """
    Generate a (soft) random geometric graph in the unit hypercube.

    Args:
        num_nodes: Number of nodes in the graph.
        kernel: Kernel that maps distances to connection probabilities.
        num_dims: Number of dimensions of the random geometric graph.

    Returns:
        graph: Generated graph.
    """
    if graph is None:
        graph = Graph()
    if num_nodes <= 0:
        raise ValueError
    if kernel is None:
        kernel = lambda x, scale, **kwargs: x < float(scale)  # noqa: E731
    graph.add_nodes(range(num_nodes))
    x = np.random.uniform(0, 1, (num_nodes, num_dims))
    dist = cdist(x, x)
    adjacency = kernel(dist, **kwargs) < np.random.uniform(0, 1, dist.shape)
    np.fill_diagonal(adjacency, 0)
    for i, j in np.transpose(np.where(adjacency)):
        graph.add_edge(i, j)
    return graph
