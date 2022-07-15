import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
import typing


def geometric(num_nodes: int, kernel: typing.Callable = None, num_dims: int = 2,
              graph: nx.Graph = None, rng: np.random.Generator = None, **kwargs) -> nx.Graph:
    """
    Generate a (soft) random geometric graph in the unit hypercube.

    Args:
        num_nodes: Number of nodes in the graph.
        kernel: Kernel that maps distances to connection probabilities.
        num_dims: Number of dimensions of the random geometric graph.
        graph: Ignored.
        rng: Random number generator.
        **kwargs: Keyword arguments passed to the kernel function.

    Returns:
        graph: Generated graph.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    rng = rng or np.random
    if kernel is None:
        kernel = lambda x, scale: x < float(scale)  # noqa: E731

    x = np.random.uniform(0, 1, (num_nodes, num_dims))
    dist = cdist(x, x)
    adjacency = kernel(dist, **kwargs) < np.random.uniform(0, 1, dist.shape)
    np.fill_diagonal(adjacency, 0)
    return nx.from_numpy_array(adjacency)
