from cython.operator cimport dereference, preincrement
from libcpp.algorithm cimport sort
from libcpp.queue cimport queue as queue_t
from libcpp.utility cimport pair as pair_t

cdef node_set_t EMPTY_NODE_SET


__PYI_HEADER = """
from __future__ import annotations
import typing
"""
__PYI_TYPEDEFS = {
    'node_t': 'int',
    'edge_list_t': 'typing.Iterable[typing.Tuple[int, int]]',
    'void': 'None',
    'node_set_t': 'typing.Set[int]',
    'count_t': 'int',
    'double': 'float',
    'Graph': 'Graph',
    'bint': 'bool',
    'th': 'torch',
}


cdef class Graph:
    """
    Graph comprising nodes connected by edges.

    Args:
        other: If given, create a copy of the graph and ignore the `strict` flag.
        strict: Whether to use strict validation of arguments. If `True`, method arguments are
            validated at the cost of some performance. If `False`, method arguments are not
            validated for improved performance with the risk of unexpected behavior.

    Attributes:
        edges (edge_set_t): Set of undirected edges.
        neighbor_map (neighbor_map_t): Mapping from nodes to their neighbors.
        nodes (node_set_t): Nodes in the graph.
    """
    def __init__(self, other: Graph = None, strict: bint = True):
        if other:
            self.nodes = other.nodes
            self.neighbor_map = other.neighbor_map
            self.strict = other.strict
        else:
            self.strict = strict

    def __contains__(self, value):
        if isinstance(value, int):
            return self.has_node(value)
        elif isinstance(value, tuple):
            return self.has_edge(*value)
        else:
            raise TypeError(value)

    def __iter__(self):
        for node in self.nodes:
            yield node

    @property
    def strict(self):
        return self.strict

    @strict.setter
    def strict(self, value: bint):
        self.strict = value

    @property
    def neighbor_map(self):
        return self.neighbor_map

    @property
    def nodes(self):
        return self.nodes

    @property
    def edges(self):
        edges = set()
        for pair in self.neighbor_map:
            for neighbor in pair.second:
                if pair.first < neighbor:
                    edges.add((pair.first, neighbor))
        return edges

    cpdef count_t get_num_nodes(self):
        """
        Get the number of nodes.
        """
        return self.nodes.size()

    cpdef count_t get_num_edges(self):
        """
        Get the number of edges.
        """
        cdef int num_edges = 0
        for pair in self.neighbor_map:
            num_edges += pair.second.size()
        return num_edges // 2

    cpdef void add_node(self, node: node_t):
        """
        Add a node.
        """
        self.nodes.insert(node)

    cpdef void add_nodes(self, nodes: node_set_t):
        """
        Add multiple nodes.
        """
        for node in nodes:
            self.add_node(node)

    cpdef bint has_node(self, node: node_t):
        """
        Check whether a node exists.
        """
        return self.nodes.find(node) != self.nodes.end()

    cdef inline int assert_node_exists(self, node_t node) except -1:
        """
        Assert that a node exists.

        Note:
            This is a no-op if :attr:`strict` is `False`.

        Raises:
            IndexError: If the node does not exist.
        """
        if self.strict and not self.has_node(node):
            raise IndexError(f"node {node} does not exist")

    cpdef int remove_node(self, node: node_t) except -1:
        """
        Remove a node.

        Raises:
            IndexError: If the node does not exist.
        """
        # Remove any edges targeting or originating from this node.
        it = self.neighbor_map.find(node)
        if it != self.neighbor_map.end():
            for neighbor in dereference(it).second:
                self._remove_directed_edge(neighbor, node)
            self.neighbor_map.erase(it)

        # Delete the node and check whether it existed in the first place.
        if self.strict and not self.nodes.erase(node):
            raise IndexError(f"node {node} does not exist")

    cpdef bint has_edge(self, node1: node_t, node2: node_t):
        """
        Check whether an edge exists.
        """
        it = self.neighbor_map.find(node1)
        if it == self.neighbor_map.end():
            return False
        return dereference(it).second.find(node2) != dereference(it).second.end()

    cpdef int add_edge(self, node1: node_t, node2: node_t) except -1:
        """
        Add an edge between two nodes.

        Raises:
            ValueError: If the two nodes are equal because loops are not allowed.
            IndexError: If either of the two nodes does not exist.
        """
        if self.strict and node1 == node2:
            raise ValueError(f"loop for node {node1} is not allowed")
        self.assert_node_exists(node1)
        self.assert_node_exists(node2)
        self._add_directed_edge(node1, node2)
        self._add_directed_edge(node2, node1)

    cpdef int _add_directed_edge(self, source: node_t, target: node_t) except -1:
        """
        Add a directed edge from a source to a target node.

        Notes:
            This method does not validate inputs even if :attr:`strict` is `True`.
        """
        it = self.neighbor_map.insert(pair_t[node_t, node_set_t](source, node_set_t()))
        dereference(it.first).second.insert(target)

    cpdef int add_edges(self, edges: edge_list_t) except -1:
        """
        Add edges between pairs of nodes.

        Raises:
            ValueError: If the nodes of any pair are equal because loops are not allowed.
            IndexError: If either of the two nodes of any pair does not exist.
        """
        for edge in edges:
            self.add_edge(edge.first, edge.second)

    cpdef int _remove_directed_edge(self, source: node_t, target: node_t) except -1:
        """
        Remove the edge from a source to a target node.

        Raises:
            IndexError: If the edge does not exist.
        """
        cdef bint exists = False
        it = self.neighbor_map.find(source)
        if it != self.neighbor_map.end():
            exists = dereference(it).second.erase(target)
        if self.strict and not exists:
            raise IndexError("edge from {source} to {target} does not exist")

    cpdef int remove_edge(self, node1: node_t, node2: node_t) except -1:
        """
        Remove an edge between two nodes.

        Raises:
            IndexError: If the edge does not exist.
        """
        self._remove_directed_edge(node1, node2)
        self._remove_directed_edge(node2, node1)

    cdef node_set_t* _get_neighbors_ptr(self, node: node_t) except NULL:
        """
        Get a pointer to the neighbors of a node.

        Args:
            node: Node for which to get neighbors.

        Returns:
            neighbors: Neighbors of the node.

        Raises:
            IndexError: If the node does not exist.
        """
        self.assert_node_exists(node)
        it = self.neighbor_map.find(node)
        if it == self.neighbor_map.end():
            return &EMPTY_NODE_SET
        return &dereference(it).second

    def get_neighbors(self, node: node_t) -> node_set_t:
        """
        Get the neighbors of a node.

        Args:
            node: Node for which to get neighbors.

        Returns:
            neighbors: Neighbors of the node.

        Raises:
            IndexError: If the node does not exist.
        """
        return dereference(self._get_neighbors_ptr(node))

    def __repr__(self):
        return f"Graph(num_nodes={self.get_num_nodes()}, num_edges={self.get_num_edges()})"


def normalize_node_labels(graph: Graph) -> Graph:
    """
    Create a copy of the graph with reset node indices such that they are consecutive in the range
    `[0, num_nodes)`.

    Args:
        graph: Graph whose nodes to normalize.

    Returns:
        relabeled: Graph with normalized node labels.
    """
    cdef node_t i = 0
    cdef node_set_t neighbors
    cdef unordered_map_t[node_t, node_t] mapping
    cdef Graph other = Graph()

    # Construct the mapping.
    for node in graph.nodes:
        mapping[node] = i
        other.nodes.insert(i)
        i += 1

    # Create the relabeled edges.
    for pair in graph.neighbor_map:
        neighbors = node_set_t()
        for neighbor in pair.second:
            neighbors.insert(mapping[neighbor])
        other.neighbor_map[mapping[pair.first]] = neighbors

    return other


cpdef Graph extract_subgraph(graph: Graph, node_set_t &nodes):
    """
    Extract a subgraph comprising the specified nodes.

    Args:
        graph: Graph from which to extract a subgraph.
        nodes: Nodes that define the subgraph.

    Returns:
        subgraph: Graph comprising only the specified nodes and their connections.
    """
    cdef Graph other = Graph()

    # Ensure all nodes exist and copy over the nodes.
    for node in nodes:
        graph.assert_node_exists(node)
        other.nodes.insert(node)

    # Copy over all edges for which both nodes are in the desired set.
    for node in nodes:
        for neighbor in dereference(graph._get_neighbors_ptr(node)):
            if nodes.find(neighbor) != nodes.end():
                other.add_edge(node, neighbor)

    return other


cpdef assert_normalized_nodel_labels(graph: Graph):
    """
    Assert that node labels are consecutive starting at zero.

    Args:
        graph: Graph whose node labels to check.

    Note:
        This operation is relatively expensive because it makes a copy of the unordered node set.

    Raises:
        ValueError: If the node labels are not normalized.
    """
    cdef node_t previous
    cdef node_list_t nodes
    if graph.get_num_nodes() == 0:
        return

    nodes.assign(graph.nodes.begin(), graph.nodes.end())
    sort(nodes.begin(), nodes.end())

    it = nodes.begin()
    previous = dereference(it)
    if previous != 0:
        raise ValueError(f"normalized node labels must start at 0")
    preincrement(it)

    while it != nodes.end():
        if dereference(it) - previous != 1:
            raise ValueError(f"expected normalized node labels but got {previous} and {dereference(it)}")
        previous = dereference(it)
        preincrement(it)


cpdef node_set_t extract_neighborhood(graph: Graph, node_set_t &nodes, depth: count_t = 1):
    """
    Extract nodes within the neighborhood of seed nodes at a given depth.

    Args:
        graph: Graph from which to extract the neighborhood.
        nodes: Seed nodes for the neighborhood extraction.
        depth: Depth of the neighborhood.

    Returns:
        neighborhood: Nodes within the neighborhood at a given depth, including seed nodes.
    """
    cdef:
        node_set_t neighborhood
        queue_t[pair_t[node_t, count_t]] queue

    # Population the initial queue.
    for node in nodes:
        queue.push(pair_t[node_t, count_t](node, 0))

    # Process elements of the queue.
    while not queue.empty():
        pair = queue.front()
        queue.pop()
        neighborhood.insert(pair.first)
        if pair.second < depth:
            for node in dereference(graph._get_neighbors_ptr(pair.first)):
                queue.push(pair_t[node_t, count_t](node, pair.second + 1))

    return neighborhood


cpdef Graph extract_neighborhood_subgraph(graph: Graph, node_set_t &nodes, depth: count_t = 1):
    """
    Extract a subgraph comprising nodes within the neighborhood of seed nodes at a given depth.

    Args:
        graph: Graph from which to extract the neighborhood.
        nodes: Seed nodes for the neighborhood extraction.
        depth: Depth of the neighborhood.

    Returns:
        subgraph: Subgraph comprising nodes within the neighborhood at a given depth, including seed
            nodes.
    """
    return extract_subgraph(graph, extract_neighborhood(graph, nodes, depth))
