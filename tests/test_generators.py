import pytest
from simulation_based_graph_inference.graph import Graph
from simulation_based_graph_inference import generators
import typing


@pytest.mark.parametrize("num_nodes", [-1, 0, 50, 200])
@pytest.mark.parametrize("generator, args", [
    (generators.generate_poisson_random_attachment, (4,)),
    (generators.generate_duplication_mutation_complementation, (.7, .3)),
    (generators.generate_duplication_mutation_random, (.6, .2)),
    (generators.generate_redirection, (4, .3)),
])
def test_generator(num_nodes: int, generator: typing.Callable, args: list):
    # Check validation against non-positive number of nodes.
    if num_nodes <= 0:
        with pytest.raises(ValueError):
            generator(num_nodes, *args)
    else:
        graph: Graph = generator(num_nodes, *args)
        assert graph.get_num_nodes() == num_nodes


@pytest.mark.parametrize("generator, args", [
    (generators.generate_poisson_random_attachment, (-1,)),
    (generators.generate_duplication_mutation_complementation, (-.7, .3)),
    (generators.generate_duplication_mutation_complementation, (.7, 1.1)),
    (generators.generate_duplication_mutation_random, (1.6, .2)),
    (generators.generate_duplication_mutation_random, (.6, -.2)),
    (generators.generate_redirection, (0, .3)),
    (generators.generate_redirection, (4, 1.3)),
])
def test_generator_invalid_parameters(generator: typing.Callable, args: list):
    with pytest.raises(ValueError):
        generator(20, *args)


def test_seed():
    edge_sets = []
    for _ in range(2):
        generators.set_seed(42)
        graph = generators.generate_poisson_random_attachment(10, 3)
        edge_sets.append(graph.edges)
    assert edge_sets[0] == edge_sets[1]
