from libcpp.set cimport set as set_t
from libcpp.map cimport map as map_t

ctypedef long node_t
ctypedef set_t[node_t] node_set_t
ctypedef map_t[node_t, node_set_t] neighbor_map_t

cdef extern from "graph.hpp" namespace "SimulationBasedGraphInference":
    cdef cppclass _Graph "SimulationBasedGraphInference::Graph":
        Graph() except +
        neighbor_map_t neighbor_map
        size_t get_num_nodes() except +
        size_t get_num_edges() except +
        void add_node(node_t) except +
        void remove_node(node_t) except +
        void add_edge(node_t, node_t) except +
        void remove_edge(node_t, node_t) except +
        size_t get_degree(node_t) except +
        node_set_t get_neighbors(node_t) except +


cdef class Graph:
    cdef _Graph _this
