# cython: boundscheck = False

from cython.operator cimport dereference
import torch as th


cdef class Graph:
    """
    Graph comprising nodes connected by edges.
    """
    cpdef count_t get_num_nodes(self):
        return self.nodes.size()

    cpdef count_t get_num_edges(self):
        cdef int num_edges = 0
        for pair in self.neighbor_map:
            num_edges += pair.second.size()
        return num_edges // 2

    cpdef void add_node(self, node: node_t):
        self.nodes.insert(node)

    cpdef void add_nodes(self, nodes: node_set_t):
        for node in nodes:
            self.add_node(node)

    cpdef bint has_node(self, node_t node):
        return self.nodes.find(node) != self.nodes.end()

    cpdef int assert_node_exists(self, node_t node) except -1:
        if not self.has_node(node):
            raise IndexError(f"node {node} does not exist")

    cpdef int remove_node(self, node: node_t) except -1:
        self.assert_node_exists(node)
        # Remove any edges targeting or originating from this node.
        it = self.neighbor_map.find(node)
        if it != self.neighbor_map.end():
            for neighbor in dereference(it).second:
                self._remove_directed_edge(neighbor, node)
            self.neighbor_map.erase(it)

        self.nodes.erase(node)

    cpdef int add_edge(self, node1: node_t, node2: node_t) except -1:
        """
        Add a connection between `node1` and `node2`.
        """
        self._add_directed_edge(node1, node2)
        self._add_directed_edge(node2, node1)

    cpdef int add_edges(self, edges: edge_list_t) except -1:
        for edge in edges:
            self.add_edge(edge.first, edge.second)

    cpdef int _add_directed_edge(self, source: node_t, target: node_t) except -1:
        if source == target:
            raise ValueError(f"loop for node {source} is not allowed")
        self.assert_node_exists(source)
        self.assert_node_exists(target)
        # The element does not exist if the lower bound is the `end` or the key of the lower bound
        # doesn't match the key we care about (see https://stackoverflow.com/a/101980/1150961).
        lb = self.neighbor_map.lower_bound(source)
        if lb == self.neighbor_map.end() or source != dereference(lb).first:
            lb = self.neighbor_map.insert(lb, pair_t[node_t, node_set_t](source, node_set_t()))
        dereference(lb).second.insert(target)

    cpdef void _remove_directed_edge(self, source: node_t, target: node_t):
        it = self.neighbor_map.find(source)
        if it != self.neighbor_map.end():
            dereference(it).second.erase(target)

    cpdef void remove_edge(self, node1: node_t, node2: node_t):
        self._remove_directed_edge(node1, node2)
        self._remove_directed_edge(node2, node1)

    def get_neighbors(self, node: node_t):
        self.assert_node_exists(node)
        it = self.neighbor_map.find(node)
        if it == self.neighbor_map.end():
            return set()
        return dereference(it).second

    def to_edge_index(self, edge_index=None):
        cdef:
            long[:, :] out
            long i = 0
        # Prepare memory.
        expected_shape = (2, 2 * self.get_num_edges())
        if edge_index is None:
            edge_index = th.empty(expected_shape, dtype=th.long)
        elif edge_index.shape != expected_shape:
            raise ValueError(f"expected shape {expected_shape} but got {edge_index.shape}")
        out = edge_index.numpy()

        # Fill the memory.
        for pair in self.neighbor_map:
            for neighbor in pair.second:
                out[0, i] = pair.first
                out[1, i] = neighbor
                i += 1

        return edge_index

    def get_neighbor_map(self):
        return self._this.neighbor_map
