from libcpp.map cimport map as map_t
from libcpp.set cimport set as set_t
from libcpp.utility cimport pair as pair_t
from libcpp.vector cimport vector as vector_t


ctypedef int node_t
ctypedef int count_t
ctypedef pair_t[node_t, node_t] edge_t
ctypedef vector_t[edge_t] edge_list_t
ctypedef set_t[node_t] node_set_t
ctypedef map_t[node_t, node_set_t] neighbor_map_t

cdef class Graph:
    cdef neighbor_map_t neighbor_map
    cdef node_set_t nodes
    cpdef count_t get_num_nodes(self)
    cpdef count_t get_num_edges(self)
    cpdef void add_node(self, node_t)
    cpdef void add_nodes(self, node_set_t)
    cpdef int remove_node(self, node_t) except -1
    cpdef bint has_node(self, node_t)
    cpdef int assert_node_exists(self, node_t) except -1
    cpdef int add_edge(self, node_t, node_t) except -1
    cpdef int add_edges(self, edge_list_t) except -1
    cpdef int _add_directed_edge(self, node_t, node_t) except -1
    cpdef void remove_edge(self, node_t, node_t)
    cpdef void _remove_directed_edge(self, node_t, node_t)
    '''
    cpdef void add_node(Graph, node_t) except +
    cpdef void remove_node(node_t) except +
    cpdef void add_edge(node_t, node_t) except +
    cpdef void remove_edge(node_t, node_t) except +
    cpdef size_t get_degree(node_t) except +
    cpdef node_set_t get_neighbors(node_t) except +'''
