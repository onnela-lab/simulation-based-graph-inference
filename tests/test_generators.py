import networkx as nx
import numpy as np
import pytest
from simulation_based_graph_inference import generators
import typing


@pytest.mark.parametrize("num_nodes", [-1, 0, 50, 200])
@pytest.mark.parametrize("generator, args", [
    (generators.poisson_random_attachment_graph, (4,)),
    (generators.duplication_complementation_graph, (.7, .3)),
    (generators.duplication_complementation_graph, (0.5, 0.5, True)),
    (generators.duplication_mutation_graph, (.6, .2)),
    (generators.duplication_mutation_graph, (0.5, 0.5, True)),
    (generators.redirection_graph, (4, .3)),
    (generators.web_graph, (0.5, 0.5, 0.5, np.arange(3) / 3)),
])
@pytest.mark.parametrize("rng", [None, np.random.default_rng()])
def test_generator(num_nodes: int, generator: typing.Callable, args: list, rng):
    # Check validation against non-positive number of nodes.
    if num_nodes <= 0:
        with pytest.raises(ValueError):
            generator(num_nodes, *args)
    else:
        graph: nx.Graph = generator(num_nodes, *args, rng=rng)
        assert graph.number_of_nodes() == num_nodes


@pytest.mark.parametrize("generator, args", [
    (generators.poisson_random_attachment_graph, (-1,)),
    (generators.duplication_complementation_graph, (-.7, .3)),
    (generators.duplication_complementation_graph, (.7, 1.1)),
    (generators.duplication_mutation_graph, (1.6, .2)),
    (generators.duplication_mutation_graph, (.6, -.2)),
    (generators.redirection_graph, (0, .3)),
    (generators.redirection_graph, (4, 1.3)),
    (generators.web_graph, (0, 0.5, 0.5, np.arange(3) / 3)),
    (generators.web_graph, (0.5, 1.1, 0.5, np.arange(3) / 3)),
    (generators.web_graph, (0.5, 0.5, -0.1, np.arange(3) / 3)),
])
def test_generator_invalid_parameters(generator: typing.Callable, args: list):
    with pytest.raises(ValueError):
        generator(20, *args)
