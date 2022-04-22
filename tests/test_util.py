from simulation_based_graph_inference import generators
from simulation_based_graph_inference import util


def test_plot_generated_graph():
    util._plot_generated_graph(generators.generate_poisson_random_attachment, 4)
