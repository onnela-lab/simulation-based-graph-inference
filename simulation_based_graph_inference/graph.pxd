from libcpp.unordered_map cimport unordered_map as unordered_map_t
from libcpp.unordered_set cimport unordered_set as unordered_set_t
from libcpp.utility cimport pair as pair_t
from libcpp.vector cimport vector as vector_t


ctypedef int node_t
ctypedef int count_t
ctypedef pair_t[node_t, node_t] edge_t
ctypedef vector_t[edge_t] edge_list_t
ctypedef unordered_set_t[node_t] node_set_t
ctypedef unordered_map_t[node_t, node_set_t] neighbor_map_t
ctypedef vector_t[node_t] node_list_t

cdef class Graph:
    cdef neighbor_map_t neighbor_map
    cdef node_set_t nodes
    cdef bint strict

    cpdef count_t get_num_nodes(self)
    cpdef void add_node(self, node_t)
    cpdef void add_nodes(self, node_set_t)
    cpdef int remove_node(self, node_t) except -1
    cpdef bint has_node(self, node_t)
    cdef inline int assert_node_exists(self, node_t) except -1

    cpdef count_t get_num_edges(self)
    cpdef int add_edge(self, node_t, node_t) except -1
    cpdef int add_edges(self, edge_list_t) except -1
    cpdef int _add_directed_edge(self, node_t, node_t) except -1
    cpdef int remove_edge(self, node_t, node_t) except -1
    cpdef int _remove_directed_edge(self, node_t, node_t) except -1
    cpdef bint has_edge(self, node_t, node_t)
