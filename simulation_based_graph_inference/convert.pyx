# cython: boundscheck = False

from cython cimport integral
import networkx as nx
import torch as th
from .graph cimport Graph

__PYI_HEADER = """
import networkx
import torch
from .graph import Graph
"""
__PYI_TYPEDEFS = {
    'Graph': 'Graph',
    'th': 'torch',
    'nx': 'networkx',
}


def _to_edge_index(graph: Graph, integral[:, :] out, dtype):
    cdef long i = 0
    cdef long imax = th.iinfo(dtype).max
    # Fill the memory.
    for pair in graph.neighbor_map:
        if pair.first > imax:
            raise ValueError(f"cannot represent node {pair.first} with {dtype}")
        for neighbor in pair.second:
            if neighbor > imax:
                raise ValueError(f"cannot represent node {neighbor} with {dtype}")
            out[0, i] = pair.first
            out[1, i] = neighbor
            i += 1


def to_edge_index(graph: Graph, edge_index: th.Tensor = None, dtype=th.long) -> th.Tensor:
    """
    Convert a graph to a :mod:`torch_geometric` edge index.

    Args:
        graph: Graph to convert.
        edge_index: Preallocated tensor with shape `(2, 2 * num_edges)`. Defaults to a newly
            allocated tensor.

    Returns:
        edge_index: Tensor with shape `(2, 2 * num_edges)` encoding the edges.

    Raises:
        ValueError: If the preallocated `edge_index` has the wrong shape.
    """
    # Prepare memory.
    expected_shape = (2, 2 * graph.get_num_edges())
    if edge_index is None:
        edge_index = th.empty(expected_shape, dtype=dtype)
    elif edge_index.shape != expected_shape:
        raise ValueError(f"expected shape {expected_shape} but got {edge_index.shape}")

    _to_edge_index(graph, edge_index.numpy(), dtype)
    return edge_index


def to_adjacency(graph: Graph, adjacency: th.Tensor = None) -> th.Tensor:
    """
    Convert a graph to a square adjacency matrix.

    Args:
        graph: Graph to convert.
        adjacency: Preallocated tensor with shape `(num_nodes, num_nodes)`. Defaults to a newly
            allocated tensor.

    Returns:
        adjacency: Tensor with shape `(num_nodes, num_nodes)` encoding the edges.

    Raises:
        ValueError: If the preallocated `adjacency` has the wrong shape.
    """
    cdef:
        long[:, :] out
    # Prepare memory.
    expected_shape = (graph.get_num_nodes(), graph.get_num_nodes())
    if adjacency is None:
        adjacency = th.zeros(expected_shape, dtype=th.long)
    elif adjacency.shape != expected_shape:
        raise ValueError(f"expected shape {expected_shape} but got {adjacency.shape}")
    out = adjacency.numpy()

    # Fill the memory.
    for pair in graph.neighbor_map:
        for neighbor in pair.second:
            out[pair.first, neighbor] = 1

    return adjacency


def to_networkx(graph: Graph) -> nx.Graph:
    """
    Convert a graph to an undirected :mod:`networkx` graph.

    Args:
        graph: Graph to convert.

    Returns:
        nxgraph: Undirected :mod:`networkx` graph.
    """
    nxgraph = nx.Graph()
    nxgraph.add_nodes_from(graph.nodes)
    nxgraph.add_edges_from(tuple(map(int, edge)) for edge in to_edge_index(graph).T)
    return nxgraph
