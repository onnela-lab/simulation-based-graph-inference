# cython: boundscheck = False
# cython: embedsignature = True
# distutils: extra_compile_args=-Wall
# distutils: extra_compile_args=-Wno-c++11-extensions

import torch as th


cdef class Graph:
    """
    Graph comprising nodes connected by edges.
    """
    def __init__(self):
        pass

    def add_node(self, node: node_t):
        self._this.add_node(node)

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    def remove_node(self, node):
        self._this.remove_node(node)

    def get_num_nodes(self):
        return self._this.get_num_nodes()

    def add_edge(self, node1: node_t, node2: node_t):
        self._this.add_edge(node1, node2)

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(*edge)

    def remove_edge(self, node1, node2):
        self._this.remove_edge(node1, node2)

    def get_degree(self, node):
        return self._this.get_degree(node)

    def get_neighbors(self, node):
        return self._this.get_neighbors(node)

    def get_num_edges(self):
        return self._this.get_num_edges()

    def to_edge_index(self, edge_index=None):
        cdef:
            long[:, :] out
            long i = 0
        # Prepare memory.
        if edge_index is None:
            edge_index = th.empty((2, 2 * self.get_num_nodes()), dtype=th.long)
        out = edge_index.numpy()

        # Fill the memory.
        for pair in self._this.neighbor_map:
            for neighbor in pair.second:
                out[0, i] = pair.first
                out[1, i] = neighbor
                i += 1

        return edge_index
