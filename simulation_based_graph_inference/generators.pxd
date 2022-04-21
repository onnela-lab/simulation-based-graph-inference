from .graph cimport _Graph
from libc.stdint cimport uint_fast32_t


cdef extern from "generators.hpp" namespace "SimulationBasedGraphInference":
    _Graph& _generate_poisson_random_attachment "SimulationBasedGraphInference::generate_poisson_random_attachment" (_Graph&, size_t, float)
