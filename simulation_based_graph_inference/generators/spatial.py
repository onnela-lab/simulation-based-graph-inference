import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit


def latent_space_graph(num_nodes: int, bias: float, scale: float, num_dims: int) -> nx.Graph:
    """
    Generate a latent space graph.

    Args:
        num_nodes: Number of nodes.
        bias: Kernel bias.
        scale: Latent space scale.
        num_dims: Number of latent space dimensions.

    Returns:
        graph: Latent space graph.
    """
    x = np.random.normal(0, scale, (num_nodes, num_dims))
    proba = expit(bias - pdist(x))
    adjacency = squareform(np.random.binomial(1, proba))
    return nx.from_numpy_array(adjacency)
