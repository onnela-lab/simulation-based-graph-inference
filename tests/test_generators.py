from simulation_based_graph_inference.graph import Graph
from simulation_based_graph_inference import generators


def test_generate_poisson_random_attachment():
    graph: Graph = generators.generate_poisson_random_attachment(100, 4.5)
    assert graph.get_num_nodes() == 100
    assert 4 < graph.get_num_edges() / 100 < 5
